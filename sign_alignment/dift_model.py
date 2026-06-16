"""Minimal access to the local SD-DIFT implementation."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import torch


@dataclass
class DiftConfig:
    repo_root: str
    img_size: int = 512
    weights_subdir: str = "weights/SD_with_prompt"

    @property
    def weights_dir(self) -> str:
        return str(Path(self.repo_root).expanduser() / self.weights_subdir)


def _add_repo_to_path(repo_root: str) -> str:
    root = str(Path(repo_root).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class DiftModel:
    def __init__(
        self,
        config: DiftConfig,
        model=None,
    ):
        self.config = config
        self.model = model

    def load(self):
        if self.model is not None:
            return self.model

        _add_repo_to_path(self.config.repo_root)
        from src.dift import SDFeaturizer

        self.model = SDFeaturizer(sd_id=self.config.weights_dir)
        return self.model

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def make_wrapper(
        self,
        prompt: str = "",
        img_size: Optional[int] = None,
    ):
        _add_repo_to_path(self.config.repo_root)
        from src.dift import DiftWrapper

        return DiftWrapper(
            Namespace(
                prompt=prompt,
                img_size=img_size or self.config.img_size,
            ),
            self.load(),
        )
