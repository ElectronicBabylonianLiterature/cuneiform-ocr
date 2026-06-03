"""Minimal access to the local SD-DIFT implementation."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys


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
    if not root:
        raise ValueError("DIFT repo_root is required")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_dift_model(config: DiftConfig):
    _add_repo_to_path(config.repo_root)
    from src.dift import SDFeaturizer

    return SDFeaturizer(sd_id=config.weights_dir)


def make_dift_wrapper(
    config: DiftConfig,
    dift,
    prompt: str = "",
    img_size: Optional[int] = None,
):
    if dift is None:
        raise ValueError("DIFT model is not loaded")
    _add_repo_to_path(config.repo_root)
    from src.dift import DiftWrapper

    return DiftWrapper(
        Namespace(prompt=prompt, img_size=img_size or config.img_size),
        dift,
    )
