"""Repository-relative paths shared by experiment entry points.

Run scripts from any working directory. Large inputs may live outside the repo:
set HC_ROCP_DATASETS_DIR, HC_ROCP_MODELS_DIR, or HC_ROCP_OUTPUTS_DIR explicitly.
Importing this module does not create directories or load datasets or weights.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def _directory(name: str, default: Path) -> Path:
    path = Path(os.path.expandvars(os.environ.get(name, str(default)))).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


DATASETS_DIR = _directory("HC_ROCP_DATASETS_DIR", DATA_ROOT)
MODELS_DIR = _directory("HC_ROCP_MODELS_DIR", REPO_ROOT / "checkpoints")
OUTPUTS_DIR = _directory("HC_ROCP_OUTPUTS_DIR", REPO_ROOT / "outputs")


def data_path(value: str | Path) -> Path:
    """Resolve a YAML data path, preserving explicit absolute paths.

Paths are independent of the caller's working directory. Raw inputs use data/,
weights use checkpoints/, and experiment outputs use outputs/.
"""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    roots = {"data": DATASETS_DIR, "checkpoints": MODELS_DIR,
             "outputs": OUTPUTS_DIR}
    if parts and parts[0] in roots:
        return roots[parts[0]].joinpath(*parts[1:]).resolve()
    return REPO_ROOT.joinpath(*parts).resolve()


def load_config(path: str | Path) -> dict:
    """Load YAML and resolve only filesystem fields (never dataset split names)."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = yaml.safe_load(path.read_text())

    def resolve(value):
        if isinstance(value, dict):
            return {key: str(data_path(item))
                    if key in {"manifest", "output_dir", "checkpoint_dir", "pretrained_checkpoint"}
                    and isinstance(item, str) else resolve(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(config)


def checkpoint_directory(config: dict) -> Path:
    """Use one location for downloaded and newly trained task checkpoints."""
    return data_path(config.get("checkpoint_dir", f"checkpoints/{config['experiment']}"))


def image_path(value: str | Path) -> Path:
    """Resolve released data/... paths or explicitly supplied custom paths."""
    return data_path(value)


def result_path(name: str) -> Path:
    """Resolve an experiment result under the shared output root."""
    path = OUTPUTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
