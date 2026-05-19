"""Resolve COCO reference image paths for eval prompts."""

from __future__ import annotations

from pathlib import Path


def coco_reference_path(
    coco_images_root: Path,
    image_id: int,
    split: str,
) -> Path:
    """Path to a downloaded COCO JPG (``{split}/{id:012d}.jpg``)."""
    return coco_images_root / split / f"{int(image_id):012d}.jpg"


def assert_references_exist(
    coco_images_root: Path,
    rows: list[dict[str, str]],
) -> None:
    missing = []
    for row in rows:
        path = coco_reference_path(
            coco_images_root,
            int(row["coco_image_id"]),
            row["coco_split"],
        )
        if not path.is_file():
            missing.append(path)
    if missing:
        sample = "\n  ".join(str(p) for p in missing[:5])
        extra = f"\n  … and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} reference images missing under {coco_images_root}. "
            f"Run: python scripts/download_coco_eval_images.py\n  {sample}{extra}"
        )
