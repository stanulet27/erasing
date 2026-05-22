"""Create and update results.json inside each eval/results/<run>/ directory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.layout import OUTPUT_MODELS

ENSEMBLE_DETECTORS = ("yolo", "rtdetr", "frcnn", "ensemble")
FID_MODELS = ("sd14_base", "esd", "rl")


def _empty_detector_block() -> dict[str, dict[str, None]]:
    return {name: {"mean_conf": None} for name in ENSEMBLE_DETECTORS}


def _empty_ensemble_models() -> dict[str, dict]:
    return {model: _empty_detector_block() for model in OUTPUT_MODELS}


def make_results_template(
    *,
    negative_guidance: float,
    iterations: int,
    erase_concept: str,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    num_images: int = 500,
    prompts_path: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    esd_checkpoint: str | None = None,
    rl_checkpoint: str | None = None,
) -> dict[str, Any]:
    return {
        "hyperparameters": {
            "negative_guidance": negative_guidance,
            "iterations": iterations,
            "erase_concept": erase_concept,
            "base_model": base_model,
            "num_images": num_images,
            "prompts_path": prompts_path,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
        "checkpoints": {
            "esd": esd_checkpoint,
            "rl": rl_checkpoint,
        },
        "ensemble": {
            "target_class": erase_concept,
            **_empty_ensemble_models(),
        },
        "fid_vs_coco": {
            "sd14_base": None,
            "esd": None,
            "rl": None,
            "degeneration_pct_vs_sd14_base": {
                "esd": None,
                "rl": None,
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_ensemble_scores(
    results: dict[str, Any],
    model: str,
    scores: dict[str, float],
) -> None:
    block = results["ensemble"][model]
    for detector in ENSEMBLE_DETECTORS:
        block[detector]["mean_conf"] = scores[detector]


def update_fid_scores(
    results: dict[str, Any],
    *,
    sd14_base: float | None = None,
    esd: float | None = None,
    rl: float | None = None,
) -> None:
    """Write whichever FID scores were computed; recompute degeneration % from current state.

    All three models are optional so callers running a subset (e.g. ESD only,
    skipping RL) can pass just what they have without crashing.
    """
    fid = results["fid_vs_coco"]
    if sd14_base is not None:
        fid["sd14_base"] = sd14_base
    if esd is not None:
        fid["esd"] = esd
    if rl is not None:
        fid["rl"] = rl

    baseline = fid.get("sd14_base")
    if baseline is None or baseline == 0:
        return  # no baseline yet → can't compute degeneration

    deg = fid["degeneration_pct_vs_sd14_base"]
    if fid.get("esd") is not None:
        deg["esd"] = (fid["esd"] - baseline) / baseline * 100.0
    if fid.get("rl") is not None:
        deg["rl"] = (fid["rl"] - baseline) / baseline * 100.0
