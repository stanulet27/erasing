"""Checkpoint paths for the eval pipeline (synced from the esd-checkpoints volume).

There are two kinds of checkpoints the eval pipeline needs to locate:

* **ESD** — produced by *this* project, with a deterministic, hyperparameter-aware
  filename built by ``utils.esd_trainer.esd_checkpoint_filename``. Each sweep cell
  produces a unique file like
  ``esd-models/sd/esd-<concept>-from-<concept>-<method>-neg{g:g}_iter{i}.safetensors``.

* **RL** — produced by the *DDPO* project (separate codebase), uploaded manually
  to the ``esd-checkpoints`` volume. We do **not** own its naming convention and
  must not assume hyperparameter-based filename structure. Treat it as an opaque
  pointer that can change between experiments. Override with the
  ``CARVE_RL_CHECKPOINT`` env var (path relative to ``erasing/``) without editing
  source when DDPO produces a different checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.esd_trainer import esd_checkpoint_filename

from eval.layout import CHECKPOINTS_VOLUME_MOUNT, ESD_MODELS_SD

# Static placeholder for the DDPO-produced RL checkpoint. Override via env if needed.
RL_CHECKPOINT = os.environ.get(
    "CARVE_RL_CHECKPOINT",
    "esd-models/sd/teddy_bear-i200-s5.safetensors",
)


def esd_checkpoint_relpath(
    *,
    erase_concept: str,
    negative_guidance: float,
    iterations: int,
    train_method: str = "esd-u",
    erase_from: str | None = None,
) -> str:
    """Relative path (under erasing/) for the ESD checkpoint of a given sweep cell."""
    filename = esd_checkpoint_filename(
        erase_concept=erase_concept,
        erase_from=erase_from,
        train_method=train_method,
        negative_guidance=negative_guidance,
        iterations=iterations,
    )
    return f"esd-models/sd/{filename}"


def checkpoint_paths(
    *,
    erase_concept: str,
    negative_guidance: float,
    iterations: int,
    train_method: str = "esd-u",
    erase_from: str | None = None,
) -> tuple[str, str]:
    """Return (esd_relpath, rl_relpath) for the cell defined by the hyperparameters."""
    return (
        esd_checkpoint_relpath(
            erase_concept=erase_concept,
            negative_guidance=negative_guidance,
            iterations=iterations,
            train_method=train_method,
            erase_from=erase_from,
        ),
        RL_CHECKPOINT,
    )


def sync_checkpoints_from_volume(mount: Path = CHECKPOINTS_VOLUME_MOUNT) -> None:
    """Copy ``*.safetensors`` from Modal volume into ``esd-models/sd/``."""
    if not mount.is_dir():
        return
    ESD_MODELS_SD.mkdir(parents=True, exist_ok=True)
    for path in mount.glob("*.safetensors"):
        dest = ESD_MODELS_SD / path.name
        if not dest.exists() or path.stat().st_mtime > dest.stat().st_mtime:
            dest.write_bytes(path.read_bytes())


def publish_checkpoints_to_volume(mount: Path = CHECKPOINTS_VOLUME_MOUNT) -> None:
    """Copy local ``esd-models/sd/*.safetensors`` to the checkpoints volume."""
    if not ESD_MODELS_SD.is_dir():
        return
    mount.mkdir(parents=True, exist_ok=True)
    for path in ESD_MODELS_SD.glob("*.safetensors"):
        dest = mount / path.name
        dest.write_bytes(path.read_bytes())
