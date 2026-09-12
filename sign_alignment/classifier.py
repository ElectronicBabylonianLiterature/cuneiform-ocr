"""Minimal ResNet18 classifier used to relabel DETR sign boxes."""

from __future__ import annotations

import cv2
import torch

from .sign import CLASSES_NAME


# The checkpoint was trained with ``sorted(df["label"].unique())``.  This is
# the explicit checkpoint-output-index -> sign mapping after removing period.
# Repeated names are the period-specific outputs that are summed at inference.
RESNET18_INDEX_TO_SIGN = (
    'AB', 'AD', 'AD', 'AK', 'AN', 'AN', 'AN', 'AN',
    'AN', 'AN', 'AN', 'AN', 'AN', 'AN', 'AN', 'AN',
    'AN', 'ARAD', 'A', 'A', 'A', 'A', 'A', 'A',
    'A', 'A', 'A', 'A', 'A', 'A', 'AŠ@z', 'AŠ',
    'AŠ', 'AŠ', 'AŠ', 'AŠ', 'AŠ', 'AŠ', 'AŠ', 'AŠ₂',
    'A₂', 'A₂', 'A₂', 'A₂', 'BAD', 'BAD', 'BAD', 'BAD',
    'BAD', 'BA', 'BA', 'BA', 'BA', 'BA', 'BA', 'BI',
    'BI', 'BI', 'BI', 'BI', 'BI', 'BI', 'BI', 'BI',
    'BU', 'BU', 'BU', 'BU', 'BU', 'BU', 'BU', 'DA',
    'DA', 'DA', 'DA', 'DA', 'DA', 'DIB', 'DIB', 'DIM₂',
    'DIN', 'DIN', 'DI', 'DI', 'DI', 'DI', 'DIŠ', 'DIŠ',
    'DIŠ', 'DIŠ', 'DIŠ', 'DIŠ', 'DIŠ', 'DIŠ', 'DIŠ', 'DUN',
    'DUN₃@g', 'DUN₃@g', 'DUN₃@g', 'DU', 'DU', 'DU', 'DU', 'DU',
    'DU', 'DU', 'EN', 'EN', 'EN', 'EN', 'EN', 'EN',
    'ERIN₂', 'ERIN₂', 'E', 'E', 'E', 'E', 'E', 'E',
    'E', 'EŠ₂', 'EŠ₂', 'E₂', 'E₂', 'E₂', 'E₂', 'E₂',
    'E₂', 'GAL', 'GAL', 'GAN', 'GAR', 'GAR', 'GAR', 'GAR',
    'GAR', 'GAR', 'GAR', 'GAR', 'GAR', 'GA', 'GA', 'GA',
    'GA', 'GA', 'GA₂', 'GIR₂', 'GI', 'GIŠ', 'GIŠ', 'GIŠ',
    'GIŠ', 'GIŠ', 'GUD', 'GURUŠ', 'GUR', 'HA', 'HA', 'HI',
    'HI', 'HI', 'HI', 'HU', 'IB', 'IB', 'IGI', 'IGI',
    'IGI', 'IGI', 'IGI', 'IGI', 'IGI', 'IGI', 'IG', 'IG',
    'IG', 'IM', 'IM', 'IM', 'IN', 'IR', 'I', 'I',
    'I', 'I', 'I', 'I', 'I', 'I', 'IŠ', 'KAK',
    'KAK', 'KAL', 'KAL', 'KA', 'KA', 'KA', 'KA', 'KA',
    'KA', 'KA', 'KID', 'KID', 'KID', 'KI', 'KI', 'KI',
    'KI', 'KI', 'KI', 'KI', 'KI', 'KI', 'KUL', 'KUR',
    'KUR', 'KUR', 'KUR', 'KU', 'KU', 'KU', 'KU', 'KU',
    'KU₃', 'KU₃', 'KU₃', 'LAL', 'LAL', 'LAL', 'LA', 'LA',
    'LA', 'LA', 'LA', 'LA', 'LI', 'LI', 'LI', 'LI',
    'LUGAL', 'LUGAL', 'LUGAL', 'LUGAL', 'LU', 'LU', 'LU', 'LU',
    'LU₂', 'LU₂', 'LU₂', 'LU₂', 'LU₂', 'LU₂', 'MA', 'MA',
    'MA', 'MA', 'MA', 'MA', 'MA', 'MA', 'MAŠ', 'MAŠ₂',
    'ME', 'ME', 'ME', 'ME', 'ME', 'ME', 'ME', 'MEŠ',
    'MEŠ', 'MEŠ', 'MEŠ', 'MEŠ', 'MEŠ', 'MEŠ', 'MEŠ', 'MIN',
    'MIN', 'MI', 'MI', 'MI', 'MI', 'MI', 'MU', 'MU',
    'MU', 'MU', 'MU', 'MU', 'MU', 'MU', 'MU', 'NAM',
    'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA',
    'NA', 'NA', 'NA', 'NE', 'NE', 'NE', 'NE', 'NE',
    'NIM', 'NI', 'NI', 'NI', 'NI', 'NI', 'NI', 'NI',
    'NI', 'NU', 'NU', 'NU', 'NU', 'NU', 'NU', 'NU',
    'NU', 'PAP', 'PAP', 'PA', 'PA', 'PA', 'PI', 'RA',
    'RA', 'RA', 'RA', 'RA', 'RA', 'RI', 'RI', 'RI',
    'RI', 'RI', 'RU', 'RU', 'RU', 'RU', 'RU', 'RU',
    'SAG', 'SAG', 'SAG', 'SAR', 'SIG', 'SILA₃', 'SI', 'SI',
    'SI', 'SI', 'SU', 'TA', 'TA', 'TA', 'TA', 'TA',
    'TA', 'TA', 'TA', 'TE', 'TI', 'TI', 'TI', 'TI',
    'TI', 'TI', 'TUG₂', 'TUM', 'TUM', 'TUM', 'TUR', 'TUR',
    'TUR', 'TU', 'TU', 'TU', 'UB', 'UD', 'UD', 'UD',
    'UD', 'UD', 'UD', 'UD', 'UD', 'UD', 'UD', 'UM',
    'UM', 'UN', 'UR', 'UR', 'UR', 'U', 'U', 'U',
    'U', 'U', 'UŠ', 'UŠ', 'UŠ', 'U₂', 'U₂', 'U₂',
    'U₂', 'U₂', 'U₂', 'U₂', 'U₂', 'ZA', 'ZA', 'ZI',
    'ZI', 'ZI', 'ZI', 'ZU', 'ZU', '|3×AN|', '|A.AN|', '|EN.ZU|',
    '|GUD×KUR|', '|HI×AŠ₂|', '|HI×BAD|', '|HI×BAD|', '|HI×BAD|', '|I.A|', '|IGI.DIB|', '|SAL.TUG₂|',
    '|SAL.TUG₂|', '|SI.A|', '|U.GA|', '|U.GUD|', '|U.U.U|', '|U.U.U|', '|U.U|', '|UD×(U.U.U)|',
    '|UD×(U.U.U)|', '|UD×(U.U.U)|', '|URU×MIN|', '|URU×MIN|', '|ŠE.NUN&NUN|', '|ŠEŠ.KI|', '|ŠEŠ.KI|', 'ŠAR₂',
    'ŠA', 'ŠA', 'ŠA', 'ŠA', 'ŠA₃', 'ŠA₃', 'ŠA₃', 'ŠA₃',
    'ŠA₃', 'ŠE', 'ŠE', 'ŠE', 'ŠEŠ', 'ŠU', 'ŠU', 'ŠU',
    'ŠU', 'ŠU', 'ŠU', 'ŠU', 'ŠU', 'ŠU₂', 'ŠU₂', 'ŠU₂',
    'ŠU₂', 'ŠU₂',
)

COMMON_SIGN_NAMES = frozenset(RESNET18_INDEX_TO_SIGN).intersection(CLASSES_NAME)


class SignClassifier:
    """Load ResNet18 and classify crops from DETR boxes."""

    def __init__(
        self,
        checkpoint_file: str,
        device: str = "auto",
        batch_size: int = 32,
        is_load_now: bool = True,
    ):
        self.checkpoint_file = checkpoint_file
        self.device = self._select_device(device)
        self.batch_size = batch_size
        self.model = None
        self.transform = None
        if is_load_now:
            self.load_model()

    @staticmethod
    def _select_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def load_model(self) -> None:
        from torchvision import models, transforms

        try:
            checkpoint = torch.load(
                self.checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_file, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(RESNET18_INDEX_TO_SIGN))
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((232, 232)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def unload_model(self) -> None:
        self.model = None
        self.transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def classify_boxes(self, boxes) -> list[tuple[str, float]]:
        from PIL import Image

        if self.model is None:
            self.load_model()

        tensors = [
            self.transform(
                Image.fromarray(cv2.cvtColor(box.crop_image(), cv2.COLOR_BGR2RGB))
            )
            for box in boxes
        ]
        results = []
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device)
            probabilities = torch.softmax(self.model(batch), dim=1).cpu().tolist()
            for row in probabilities:
                merged = {}
                for sign_name, score in zip(RESNET18_INDEX_TO_SIGN, row):
                    if sign_name in COMMON_SIGN_NAMES:
                        merged[sign_name] = merged.get(sign_name, 0.0) + score
                results.append(max(merged.items(), key=lambda item: item[1]))
        return results
