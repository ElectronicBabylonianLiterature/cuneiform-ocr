"""Integration checks with the local MAE checkpoint; no GPU/training run needed.

HF_HUB_OFFLINE=1 python -m unittest discover -s detection_model/tests -v
"""

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import DeformableDetrConfig
from transformers.utils import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import model as model_module
import train as base_train
import test as base_test
import train_with_cuneimae as cunei_train
import test_with_cuneimae as cunei_test


def fingerprint(module):
    digest = hashlib.sha256()
    for value in module.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainableCuneiMAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = ROOT / "pretrained" / "checkpoint-10.pth"
        if not checkpoint.exists():
            raise unittest.SkipTest("Local CuneiMAE checkpoint is required")
        logging.set_verbosity_error()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = model_module.build_model(
            [f"class_{i}" for i in range(210)], backbone="cuneimae", mae_checkpoint=checkpoint,
        )
        cls.pixels = torch.randn(1, 3, 224, 224)
        cls.mask = torch.ones(1, 224, 224, dtype=torch.bool)

    def test_01_dynamic_shapes_and_padding(self):
        model = self.model.eval()
        with torch.inference_mode():
            out = model(pixel_values=self.pixels, pixel_mask=self.mask)
        self.assertEqual(list(out.encoder_last_hidden_state.shape), [1, 1045, 256])
        self.assertEqual(list(out.logits.shape), [1, 300, 210])
        self.assertTrue(torch.isfinite(out.pred_boxes).all())

        pixels = torch.zeros(2, 3, 229, 301)
        pixels[0, :, :224, :224] = self.pixels[0]
        pixels[1] = torch.randn(3, 229, 301)
        mask = torch.ones(2, 229, 301, dtype=torch.bool)
        mask[0] = False
        mask[0, :224, :224] = True
        with torch.inference_mode():
            alone = model.model.backbone.encoder(self.pixels, self.mask)
            batched = model.model.backbone.encoder(pixels, mask)
            torch.testing.assert_close(alone[0], batched[0, :, :14, :14], rtol=2e-5, atol=2e-5)
            out = model(pixel_values=pixels, pixel_mask=mask)
        self.assertEqual(list(out.encoder_last_hidden_state.shape), [2, 1487, 256])
        self.assertTrue(torch.isfinite(out.logits).all())

    def test_02_optimizer_updates_encoder_and_adapters(self):
        model = self.model.train()
        model.gradient_checkpointing_enable()
        encoder = model.model.backbone.encoder
        self.assertTrue(encoder.training)
        self.assertTrue(model.model.backbone.projection.training)
        self.assertTrue(all(p.requires_grad for name, p in encoder.named_parameters() if name != "pos_embed"))
        self.assertFalse(encoder.pos_embed.requires_grad)
        before_encoder = fingerprint(encoder)
        projection = model.model.backbone.projection[0].weight
        before_projection = projection.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in optimizer_ids for p in encoder.parameters() if p.requires_grad))
        self.assertNotIn(id(encoder.pos_embed), optimizer_ids)
        targets = [{"class_labels": torch.tensor([3]),
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.15]])}]
        outputs = model(pixel_values=self.pixels, pixel_mask=self.mask, labels=targets)
        loss, _ = model_module.DetectionLoss(model.config)(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in encoder.named_parameters():
            if name == "pos_embed":
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in (encoder.patch_embed.proj.weight, encoder.blocks[0].attn.qkv.weight,
                          encoder.blocks[-1].attn.qkv.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for heads in (model.class_embed, model.bbox_embed):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in heads.parameters()))
        for parameter in (projection, model.model.input_proj[3][0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer.step()
        self.assertNotEqual(before_encoder, fingerprint(encoder))
        self.assertFalse(torch.equal(before_projection, projection))
        model.zero_grad(set_to_none=True)

    def test_03_training_checkpoint_restores_trainable_model_and_settings(self):
        model = self.model.eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40])
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = torch.Generator().manual_seed(42)
        dataset = SimpleNamespace(class_names=[f"class_{i}" for i in range(210)],
                                  category_to_label={i: i for i in range(210)}, ids=[1])
        config = {**base_train.TRAIN_CONFIG, "epochs": 7}
        with torch.inference_mode():
            before = model(pixel_values=self.pixels, pixel_mask=self.mask)
        with tempfile.TemporaryDirectory() as temporary:
            path = base_train.checkpoint_path(temporary, "last.pt", "test-run")
            base_train.save_checkpoint(path, model, optimizer, scheduler, scaler, 2, 9,
                                       0.1, generator, dataset, dataset, train_config=config, run_tag="test-run")
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self.assertEqual(saved["train_config"]["epochs"], 7)
            self.assertEqual(path.name, "last_test-run.pt")
            self.assertEqual(saved["run_tag"], "test-run")
            self.assertFalse(saved["model_config"]["cuneimae_frozen"])
            restored = model_module.build_model([], saved, backbone="cuneimae",
                                                 mae_checkpoint="/tmp/absent-mae.pth").eval()
            with torch.inference_mode():
                after = restored(pixel_values=self.pixels, pixel_mask=self.mask)
            torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
            torch.testing.assert_close(before.pred_boxes, after.pred_boxes, rtol=0, atol=0)
            restored.train()
            self.assertTrue(restored.model.backbone.encoder.training)
            self.assertTrue(all(p.requires_grad for name, p in restored.model.backbone.encoder.named_parameters()
                                if name != "pos_embed"))
            restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=2e-4)
            restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer, [40])
            start, step, best = base_train.restore_training(
                saved, restored_optimizer, restored_scheduler, scaler, generator,
            )
            self.assertEqual((start, step, best), (3, 9, 0.1))
            with self.assertRaisesRegex(ValueError, "Checkpoint backbone"):
                model_module.build_model([], saved, backbone="resnet")



class SharedEntrypointsTest(unittest.TestCase):
    def test_wrappers_delegate_without_overwriting_baseline_settings(self):
        baseline_debug, baseline_output = base_train.DEBUG_SUBSET, base_train.OUTPUT_DIR
        with patch.object(base_train, "main") as run:
            cunei_train.main()
        settings = run.call_args.kwargs
        self.assertEqual(settings["debug_subset"], cunei_train.DEBUG_SUBSET)
        self.assertEqual(settings["output_dir"].name, "output_with_cuneimae")
        self.assertEqual(settings["model_builder"].keywords["backbone"], "cuneimae")
        self.assertTrue(settings["tag_checkpoints"])
        self.assertIsNone(settings["resume"])
        self.assertIsNot(cunei_train.TRAIN_CONFIG, base_train.TRAIN_CONFIG)
        self.assertEqual((base_train.DEBUG_SUBSET, base_train.OUTPUT_DIR), (baseline_debug, baseline_output))
        with patch.object(base_test, "main") as run:
            cunei_test.main()
        settings = run.call_args.kwargs
        self.assertEqual(settings["checkpoint_path"], cunei_train.OUTPUT_DIR / f"best_{cunei_train.RUN_TAG}.pt")
        self.assertEqual(settings["model_builder"].keywords["backbone"], "cuneimae")
        self.assertIsNot(cunei_test.TEST_CONFIG, base_test.TEST_CONFIG)

    def test_legacy_resnet_checkpoint_still_reloads(self):
        config = DeformableDetrConfig.from_pretrained(
            model_module.PRETRAINED_MODEL, **model_module.MODEL_CONFIG,
            id2label={0: "a", 1: "b"}, label2id={"a": 0, "b": 1},
        )
        config.class_loss_coefficient = 2.0
        baseline = model_module.DetectionModel(config)
        saved = {"model_config": config.to_dict(), "model": baseline.state_dict()}
        self.assertNotIn("backbone_type", saved["model_config"])
        restored = model_module.build_model([], saved)
        self.assertEqual(restored.model.backbone.intermediate_channel_sizes, [512, 1024, 2048])
        self.assertEqual(restored.model.input_proj[3][0].in_channels, 2048)
        self.assertFalse(hasattr(restored.model.backbone, "encoder"))


if __name__ == "__main__":
    unittest.main()
