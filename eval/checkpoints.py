"""Fixed checkpoint paths (synced from the Modal esd-checkpoints volume)."""

from __future__ import annotations

from pathlib import Path

from eval.layout import CHECKPOINTS_VOLUME_MOUNT, ESD_MODELS_SD

# Always use these two files on the checkpoints volume / esd-models/sd/
ESD_CHECKPOINT = "esd-models/sd/esd-teddy_bear-from-teddy_bear-esdu.safetensors"
RL_CHECKPOINT = "esd-models/sd/teddy_bear-i200-s5.safetensors"


def checkpoint_paths() -> tuple[str, str]:
    return ESD_CHECKPOINT, RL_CHECKPOINT


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
