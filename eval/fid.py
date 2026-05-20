"""FID vs a reference image folder (matched COCO eval set or flat directory)."""

from __future__ import annotations

from pathlib import Path


def _iter_reference_images(reference_dir: Path):
    """Yield image paths under ``reference_dir`` (flat or train/val subdirs)."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    for pattern in patterns:
        yield from reference_dir.glob(pattern)
        yield from reference_dir.glob(f"**/{pattern}")


def count_reference_images(reference_dir: Path) -> int:
    return sum(1 for _ in _iter_reference_images(reference_dir))


def compute_fid(generated_dir: Path, reference_dir: Path) -> float:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"COCO reference directory not found: {reference_dir}")
    if count_reference_images(reference_dir) == 0:
        raise FileNotFoundError(f"No images in reference directory: {reference_dir}")

    from cleanfid import fid

    return float(
        fid.compute_fid(
            str(generated_dir),
            str(reference_dir),
            mode="clean",
            num_workers=4,
        )
    )
