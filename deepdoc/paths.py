"""Project-relative defaults for DeepDoc model assets."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_base_directory() -> str:
    return os.environ.get("DEEPDOC_PROJECT_ROOT") or str(PROJECT_ROOT)


def get_deepdoc_model_directory() -> str:
    return os.environ.get("DEEPDOC_MODEL_DIR") or str(PROJECT_ROOT / "models" / "deepdoc")
