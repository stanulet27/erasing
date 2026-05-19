"""Paths and naming for eval-runs experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ERASING_ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNS_ROOT = Path(os.environ.get("CARVE_EVAL_RUNS_ROOT", ERASING_ROOT / "eval-runs"))
ESD_MODELS_SD = Path(os.environ.get("CARVE_ESD_MODELS_SD", ERASING_ROOT / "esd-models" / "sd"))
CHECKPOINTS_VOLUME_MOUNT = Path(os.environ.get("CARVE_CHECKPOINTS_MOUNT", "/checkpoints"))

OUTPUT_MODELS = ("sd14_base", "esd", "rl")
RESULTS_FILENAME = "results.json"


def experiment_name(negative_guidance: float, iterations: int) -> str:
    """Directory name for one hyperparameter sweep point."""
    neg = f"{negative_guidance:g}"
    return f"neg{neg}_iter{iterations}"


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path
    results_path: Path
    outputs_dir: Path

    def output_dir(self, model: str) -> Path:
        if model not in OUTPUT_MODELS:
            raise ValueError(f"model must be one of {OUTPUT_MODELS}, got {model!r}")
        return self.outputs_dir / model


def experiment_paths(negative_guidance: float, iterations: int) -> ExperimentPaths:
    run_dir = EVAL_RUNS_ROOT / experiment_name(negative_guidance, iterations)
    return ExperimentPaths(
        run_dir=run_dir,
        results_path=run_dir / RESULTS_FILENAME,
        outputs_dir=run_dir / "outputs",
    )
