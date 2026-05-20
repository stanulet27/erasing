"""Per-image detector confidences (positive scores, not DDPO reward sign)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Faster R-CNN COCO category id for common eval concepts (matches ddpo rewards.Ensemble).
COCO_FRCNN_CLASS_IDS: dict[str, int] = {
    "teddy bear": 88,
}


def _ensure_ddpo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ddpo_root = repo_root / "ddpo-pytorch"
    if str(ddpo_root) not in sys.path:
        sys.path.insert(0, str(ddpo_root))


def load_ensemble():
    _ensure_ddpo_on_path()
    from ddpo_pytorch.rewards import Ensemble

    return Ensemble()


def score_image(ensemble, image: Image.Image, target_class: str) -> dict[str, float]:
    frcnn_id = COCO_FRCNN_CLASS_IDS.get(target_class)
    if frcnn_id is None:
        raise KeyError(
            f"No Faster R-CNN class id for {target_class!r}. "
            f"Add it to COCO_FRCNN_CLASS_IDS in eval/ensemble_scores.py"
        )

    yolo_id = ensemble.yolo_name_to_id[target_class]
    yolo = ensemble.call_yolo(image, yolo_id)
    rtdetr = ensemble.call_detr(image, yolo_id)
    frcnn = ensemble.call_resnet(image, frcnn_id)
    combined = float(np.mean([yolo, rtdetr, frcnn]))

    return {
        "yolo": yolo,
        "rtdetr": rtdetr,
        "frcnn": frcnn,
        "ensemble": combined,
    }


def list_images(folder: Path) -> list[Path]:
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(ext))
    return sorted(paths)


def mean_scores_for_folder(
    ensemble,
    folder: Path,
    target_class: str,
) -> tuple[dict[str, float], int]:
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    sums = {key: 0.0 for key in ("yolo", "rtdetr", "frcnn", "ensemble")}
    for path in images:
        with Image.open(path) as img:
            per_image = score_image(ensemble, img.convert("RGB"), target_class)
        for key in sums:
            sums[key] += per_image[key]

    n = len(images)
    means = {key: sums[key] / n for key in sums}
    return means, n
