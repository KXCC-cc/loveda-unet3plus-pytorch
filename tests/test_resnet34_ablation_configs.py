"""ResNet34 受控消融配置测试。"""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine.trainer import _make_optimizer
from utils.config import load_config
from utils.constants import IGNORE_INDEX, map_raw_mask
from utils.metrics import SegmentationMetrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
BASELINE_PATH = CONFIG_DIR / "loveda_4060_resnet34.yaml"
BASELINE_SHA256 = "245c8b352680314a86750d8842e29db3c614c9d7ff8a77c1ac2bab0e79e6d107"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


def _resolved(path: str) -> dict:
    config = copy.deepcopy(load_config(CONFIG_DIR / path))
    config.pop("config_path", None)
    return config


def _changed_paths(baseline: dict, variant: dict) -> set[str]:
    left = _flatten(baseline)
    right = _flatten(variant)
    return {
        key
        for key in set(left) | set(right)
        if key not in left or key not in right or left[key] != right[key]
    }


class _ToySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1))
        self.decoder = nn.Sequential(nn.Conv2d(4, 7, 1))


class ResNet34AblationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = _resolved("loveda_4060_resnet34.yaml")
        cls.multiscale = _resolved("resnet_pretrained_512_multiscale.yaml")
        cls.classaware = _resolved("resnet34_512_classaware_only.yaml")
        cls.diff_lr = _resolved("resnet34_512_diff_lr.yaml")
        cls.focal = _resolved("resnet34_512_focal.yaml")
        cls.combined = _resolved("resnet34_512_multiscale_classaware.yaml")

    def assert_only_changed(self, variant, allowed_prefixes):
        changed = _changed_paths(self.baseline, variant)
        unexpected = {
            path
            for path in changed
            if not any(
                path == prefix or path.startswith(prefix + ".")
                for prefix in allowed_prefixes
            )
        }
        self.assertEqual(unexpected, set(), f"发现非预期配置变化：{sorted(unexpected)}")

    def assert_common_resnet34_baseline(self, config):
        self.assertEqual(config["model"]["backbone"], "resnet34")
        self.assertTrue(config["model"]["pretrained"])
        self.assertEqual(config["model"]["weights"], "IMAGENET1K_V1")
        self.assertEqual(config["model"]["cat_channels"], 64)
        self.assertEqual(config["data"]["crop_size"], [512, 512])
        self.assertEqual(config["data"]["loader"]["batch_size"], 1)
        self.assertEqual(config["training"]["accumulation_steps"], 4)
        self.assertEqual(
            config["data"]["loader"]["batch_size"]
            * config["training"]["accumulation_steps"],
            4,
        )
        self.assertTrue(config["runtime"]["amp"])
        self.assertTrue(config["model"]["freeze_encoder_bn"])
        self.assertEqual(config["optimizer"]["name"], "sgd")
        self.assertEqual(config["optimizer"]["momentum"], 0.9)
        self.assertEqual(config["optimizer"]["weight_decay"], 1e-4)
        self.assertEqual(config["scheduler"]["name"], "poly")
        self.assertEqual(config["scheduler"]["power"], 0.9)
        self.assertEqual(config["training"]["max_iters"], 15000)

    def test_baseline_file_is_byte_identical(self):
        digest = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, BASELINE_SHA256)

    def test_multiscale_only_changes_run_identity_and_train_scales(self):
        self.assert_only_changed(
            self.multiscale,
            {"run.name", "run.output_dir", "data.train_scales"},
        )
        self.assertEqual(
            self.multiscale["data"]["train_scales"],
            [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        )

    def test_classaware_only_changes_run_identity_and_crop_sampler(self):
        self.assert_only_changed(
            self.classaware,
            {"run.name", "run.output_dir", "data.class_aware_crop"},
        )
        self.assertEqual(self.classaware["data"]["train_scales"], [1.0])
        self.assertEqual(
            self.classaware["data"]["class_aware_crop"],
            {
                "probability": 0.5,
                "classes": [1, 2, 3, 4],
                "min_target_pixels": 512,
                "max_attempts": 8,
            },
        )

    def test_differential_lr_only_changes_run_identity_and_lr_groups(self):
        self.assert_only_changed(
            self.diff_lr,
            {"run.name", "run.output_dir", "optimizer.differential_lr"},
        )
        self.assertEqual(
            self.diff_lr["optimizer"]["differential_lr"],
            {"encoder_lr": 0.0005, "decoder_lr": 0.0025},
        )

    def test_differential_lr_is_not_scaled_twice(self):
        model = _ToySegmentationModel()
        optimizer, resolved_lr = _make_optimizer(self.diff_lr, model, effective_batch=4)
        self.assertEqual(resolved_lr, 0.0025)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.0005, 0.0025],
        )

    def test_focal_only_changes_run_identity_and_loss_weights(self):
        self.assert_only_changed(
            self.focal,
            {
                "run.name",
                "run.output_dir",
                "loss.ce_weight",
                "loss.dice_weight",
                "loss.focal_weight",
                "loss.focal_gamma",
            },
        )
        self.assertEqual(self.focal["loss"]["ce_weight"], 0.5)
        self.assertEqual(self.focal["loss"]["dice_weight"], 1.0)
        self.assertEqual(self.focal["loss"]["focal_weight"], 0.5)
        self.assertEqual(self.focal["loss"]["focal_gamma"], 2.0)

    def test_every_experiment_keeps_common_resnet34_settings(self):
        for config in (
            self.baseline,
            self.multiscale,
            self.classaware,
            self.diff_lr,
            self.focal,
            self.combined,
        ):
            with self.subTest(run=config["run"]["name"]):
                self.assert_common_resnet34_baseline(config)

    def test_existing_classaware_profile_is_multiscale_plus_classaware(self):
        self.assertEqual(
            self.combined["data"]["train_scales"],
            [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        )
        self.assertEqual(
            self.combined["data"]["class_aware_crop"]["probability"],
            0.5,
        )

    def test_label_mapping_and_global_confusion_matrix_semantics(self):
        raw = np.array([[0, 1, 7]], dtype=np.uint8)
        self.assertEqual(map_raw_mask(raw).tolist(), [[IGNORE_INDEX, 0, 6]])

        metrics = SegmentationMetrics()
        predictions = torch.tensor([[[0, 0, 2, 6]]], dtype=torch.long)
        targets = torch.tensor([[[0, 1, 2, IGNORE_INDEX]]], dtype=torch.long)
        metrics.update(predictions, targets)
        scores = metrics.compute()
        expected_iou = torch.tensor([0.5, 0.0, 1.0], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(scores["per_class_iou"][:3], expected_iou)
        )
        self.assertAlmostEqual(scores["mean_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
