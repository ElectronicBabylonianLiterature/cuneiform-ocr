"""Model and complete two-stage loss, compatible with transformers==5.17.0.

Model documentation: https://huggingface.co/docs/transformers/model_doc/deformable_detr
"""

import math
from pathlib import Path

import torch
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection
from transformers.loss.loss_deformable_detr import (
    DeformableDetrForObjectDetectionLoss,
    DeformableDetrHungarianMatcher,
    DeformableDetrImageLoss,
)


PRETRAINED_MODEL = "SenseTime/deformable-detr-with-box-refine-two-stage"
MODEL_CONFIG = {
    "num_queries": 300,
    "two_stage_num_proposals": 300,
    "two_stage": True,
    "with_box_refine": True,
    "auxiliary_loss": True,
    "class_cost": 2,
}
CLASS_LOSS_COEFFICIENT = 2.0
CLASS_PRIOR = 0.01
GRADIENT_CHECKPOINTING = True


class DetectionModel(DeformableDetrForObjectDetection):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "backbone_type", "resnet") == "cuneimae":
            from cuneimae import attach_cuneimae
            # Rebuild the architecture when restoring a detector checkpoint; do not reload MAE weights.
            attach_cuneimae(self)

    # Keep the HF detector and FP32 matching/loss with either ResNet or a trainable CuneiMAE backbone.
    # FP16 sigmoid can round to 1, producing inf in the Hungarian cost's log(1-p).
    def loss_function(self, logits, labels, device, pred_boxes, config,
                      outputs_class=None, outputs_coord=None, **kwargs):
        with torch.autocast(logits.device.type, enabled=False):
            return DeformableDetrForObjectDetectionLoss(
                logits.float(), labels, device, pred_boxes.float(), config,
                outputs_class.float(), outputs_coord.float(),
            )


def build_model(class_names, checkpoint=None, *, backbone="resnet", mae_checkpoint=None):
    if backbone not in ("resnet", "cuneimae"):
        raise ValueError(f"Unknown backbone: {backbone}")
    if checkpoint is not None:
        # Rebuild from the saved config without downloading pretrained weights; restore all trained heads.
        config = DeformableDetrConfig.from_dict(checkpoint["model_config"])
        saved_backbone = getattr(config, "backbone_type", "resnet")
        if saved_backbone != backbone:
            raise ValueError(f"Checkpoint backbone {saved_backbone} != requested {backbone}")
        model = DetectionModel(config)
        model.load_state_dict(checkpoint["model"])
        return model

    id2label = dict(enumerate(class_names))
    config = DeformableDetrConfig.from_pretrained(
        PRETRAINED_MODEL, **MODEL_CONFIG, id2label=id2label,
        label2id={name: label for label, name in id2label.items()},
    )
    config.class_loss_coefficient = CLASS_LOSS_COEFFICIENT
    model = DetectionModel.from_pretrained(
        PRETRAINED_MODEL, config=config, ignore_mismatched_sizes=True,
    )
    # Match the original experiment: identical initial classifier weights, a low foreground prior,
    # and independently trained parameters for each head.
    initial_weight = model.class_embed[0].weight.detach().clone()
    with torch.no_grad():
        for head in model.class_embed:
            head.weight.copy_(initial_weight)
            head.bias.fill_(math.log(CLASS_PRIOR / (1 - CLASS_PRIOR)))
    if backbone == "cuneimae":
        from cuneimae import attach_cuneimae
        path = (Path(mae_checkpoint) if mae_checkpoint is not None else
                Path(__file__).resolve().parent / "pretrained" / "checkpoint-10.pth")
        attach_cuneimae(model, path)
    else:
        model.config.backbone_type = "resnet"
    return model


class DetectionLoss(torch.nn.Module):
    """Preserve the original decoder classification weight and encoder proposal supervision."""

    def __init__(self, config):
        super().__init__()
        self.class_weight = config.class_loss_coefficient
        self.box_weight = config.bbox_loss_coefficient
        self.giou_weight = config.giou_loss_coefficient
        matcher = DeformableDetrHungarianMatcher(
            class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost,
        )
        self.encoder_loss = DeformableDetrImageLoss(
            matcher=matcher, num_classes=config.num_labels, focal_alpha=config.focal_alpha,
            losses=["labels", "boxes", "cardinality"],
        )

    def forward(self, outputs, targets):
        # HF 5.17 uses classification weight 1; the original experiment uses 2, including auxiliary layers.
        classification = sum(value for key, value in outputs.loss_dict.items()
                             if key == "loss_ce" or key.startswith("loss_ce_"))
        decoder = outputs.loss + (self.class_weight - 1) * classification
        # The two-stage encoder predicts objectness; the decoder learns the actual classes.
        proposals = [{"class_labels": torch.zeros_like(t["class_labels"]), "boxes": t["boxes"]}
                     for t in targets]
        with torch.autocast(outputs.logits.device.type, enabled=False):
            losses = self.encoder_loss(
                {"logits": outputs.enc_outputs_class.float(),
                 "pred_boxes": outputs.enc_outputs_coord_logits.float().sigmoid()}, proposals,
            )
        encoder = (self.class_weight * losses["loss_ce"] + self.box_weight * losses["loss_bbox"]
                   + self.giou_weight * losses["loss_giou"])
        return decoder + encoder, {"decoder": decoder.detach(), "encoder": encoder.detach()}
