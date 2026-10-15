"""Trainable CuneiMAE encoder + three-scale backbone for the existing detector.

Checkpoint architecture: mae_vit_base_patch16, 12 blocks, width 768, 12 heads.
The detector still uses FOUR feature levels; it builds the fourth itself.
"""

import argparse
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from timm.models.vision_transformer import Block


ARCH = {
    "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12,
    "mlp_ratio": 4, "pos_grid": [14, 14], "norm_eps": 1e-6,
}
ADAPTER = {
    "version": 1, "channels": 256, "backbone_strides": [8, 16, 32],
    "detector_strides": [8, 16, 32, 64],
    "resampling": "bilinear_up2_crop__avgpool2_ceil",
}


class CuneiMAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Names match the original MAE checkpoint exactly.
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 768, 16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768), requires_grad=False)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            Block(768, 12, mlp_ratio=4, qkv_bias=True, norm_layer=norm)
            for _ in range(12)
        ])
        self.norm = norm(768)
        # The existing HF gradient_checkpointing_enable() toggles this flag too.
        self.gradient_checkpointing = False

    def load_pretrained(self, path):
        with torch.serialization.safe_globals([argparse.Namespace]):
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        args = saved.get("args")
        if args is not None and getattr(args, "model", None) != "mae_vit_base_patch16":
            raise ValueError("Expected a mae_vit_base_patch16 checkpoint")
        source = saved["model"]
        prefixes = ("patch_embed.", "blocks.", "norm.")
        weights = {
            key: value for key, value in source.items()
            if key in ("cls_token", "pos_embed") or key.startswith(prefixes)
        }
        # Do not silently accept missing/mismatched encoder weights.
        self.load_state_dict(weights, strict=True)
        return {"encoder_tensors": len(weights), "epoch": saved.get("epoch")}

    def _position_embeddings(self, valid, dtype):
        """Interpolate pretrained positions per image, not per padded batch."""
        batch, height, width = valid.shape
        base = self.pos_embed[:, 1:].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        positions = []
        for index in range(batch):
            h = int(valid[index, :, 0].sum().item())
            w = int(valid[index, 0, :].sum().item())
            if not h or not w:
                raise ValueError("Each image must contain a non-empty valid rectangle")
            pos = F.interpolate(base.float(), size=(h, w), mode="bicubic", align_corners=False)
            positions.append(F.pad(pos, (0, width - w, 0, height - h)))
        patches = torch.cat(positions).flatten(2).transpose(1, 2).to(dtype)
        cls = self.pos_embed[:, :1].expand(batch, -1, -1).to(dtype)
        return torch.cat((cls, patches), dim=1)

    def forward(self, pixels, pixel_mask):
        # External preprocessing/labels retain their original sizes.
        batch, _, height, width = pixels.shape
        pad_h, pad_w = (-height) % 16, (-width) % 16
        pixels = F.pad(pixels, (0, pad_w, 0, pad_h))
        padded_mask = F.pad(pixel_mask.bool(), (0, pad_w, 0, pad_h), value=False)
        # A partially valid boundary patch is retained; fully padded patches are masked.
        valid = F.max_pool2d(padded_mask[:, None].float(), 16, 16)[:, 0].bool()
        feature = self.patch_embed.proj(pixels)
        hp, wp = feature.shape[-2:]
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + self._position_embeddings(valid, tokens.dtype)
        key_valid = torch.cat((valid.new_ones(batch, 1), valid.flatten(1)), dim=1)
        attention_mask = key_valid[:, None, None, :]
        # Dense raster order throughout: no random masking or token shuffling.
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = activation_checkpoint(block, tokens, attention_mask, use_reentrant=False)
            else:
                tokens = block(tokens, attn_mask=attention_mask)
        tokens = self.norm(tokens)[:, 1:]
        feature = tokens.transpose(1, 2).reshape(batch, 768, hp, wp)
        return feature.masked_fill(~valid[:, None], 0)


class CuneiMAEBackbone(nn.Module):
    intermediate_channel_sizes = [256, 256, 256]

    def __init__(self):
        super().__init__()
        self.encoder = CuneiMAEEncoder()
        self.projection = nn.Sequential(nn.Conv2d(768, 256, 1), nn.GroupNorm(32, 256))
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

    def forward(self, pixels, pixel_mask):
        height, width = pixels.shape[-2:]
        p16 = self.projection(self.encoder(pixels, pixel_mask))
        p8 = F.interpolate(p16, scale_factor=2, mode="bilinear", align_corners=False)
        # Match ceil(H/stride), ceil(W/stride), including non-multiple-of-16 inputs.
        p8 = p8[:, :, :(height + 7) // 8, :(width + 7) // 8]
        p32 = F.avg_pool2d(p16, 2, 2, ceil_mode=True, count_include_pad=False)
        outputs = []
        for feature in (p8, p16, p32):
            mask = F.interpolate(
                pixel_mask[:, None].float(), size=feature.shape[-2:], mode="nearest",
            )[:, 0].bool()
            outputs.append((feature.masked_fill(~mask[:, None], 0), mask))
        return outputs


def attach_cuneimae(detector, pretrained_path=None):
    """Call on CPU before creating the optimizer; omit path on detector resume."""
    config = detector.config
    if config.d_model != 256 or config.num_feature_levels != 4:
        raise ValueError("This adapter requires d_model=256 and four detector levels")
    if getattr(config, "cuneimae_arch", ARCH) != ARCH:
        raise ValueError("Unsupported CuneiMAE encoder architecture")
    if getattr(config, "cuneimae_adapter", ADAPTER) != ADAPTER:
        raise ValueError("Unsupported CuneiMAE adapter configuration")
    backbone = CuneiMAEBackbone()
    if pretrained_path is not None:
        backbone.encoder.load_pretrained(pretrained_path)
    detector.model.backbone = backbone
    # First three features already have 256 channels. The existing HF forward
    # computes the fourth level from features[-1][0], now also 256 channels.
    extra = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GroupNorm(32, 256))
    nn.init.xavier_uniform_(extra[0].weight)
    nn.init.zeros_(extra[0].bias)
    detector.model.input_proj = nn.ModuleList([
        nn.Identity(), nn.Identity(), nn.Identity(), extra,
    ])
    config.backbone_type = "cuneimae"
    config.cuneimae_frozen = False
    config.cuneimae_arch = dict(ARCH)
    config.cuneimae_adapter = dict(ADAPTER)
    return detector
