#!/usr/bin/env python3
"""Generate sd14_base, esd, and rl image folders for one eval run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import ERASING_ROOT

from eval.layout import OUTPUT_MODELS, experiment_paths
from eval.results_io import load_results


def _run_generate(
    *,
    base_model: str,
    prompts_path: Path,
    output_dir: Path,
    esd_path: Path | None,
    device: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_samples: int,
    from_case: int,
) -> None:
    cmd = [
        sys.executable,
        str(ERASING_ROOT / "eval" / "scripts" / "generate-images.py"),
        "--base_model",
        base_model,
        "--prompts_path",
        str(prompts_path),
        "--output_dir",
        str(output_dir),
        "--device",
        device,
        "--guidance_scale",
        str(guidance_scale),
        "--num_inference_steps",
        str(num_inference_steps),
        "--num_samples",
        str(num_samples),
        "--from_case",
        str(from_case),
    ]
    if esd_path is not None:
        cmd.extend(["--esd_path", str(esd_path)])
    subprocess.run(cmd, check=True, cwd=ERASING_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all three model output folders for an eval run.")
    parser.add_argument("--negative-guidance", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--only",
        choices=OUTPUT_MODELS,
        nargs="+",
        default=None,
        help="Generate only these models (default: all three)",
    )
    parser.add_argument("--from-case", type=int, default=0)
    args = parser.parse_args()

    paths = experiment_paths(args.negative_guidance, args.iterations)
    if not paths.results_path.is_file():
        raise FileNotFoundError(
            f"Missing {paths.results_path}. Run init_eval_run.py first."
        )

    results = load_results(paths.results_path)
    hp = results["hyperparameters"]
    ckpt = results["checkpoints"]

    prompts_path = ERASING_ROOT / hp["prompts_path"]
    if not prompts_path.is_file():
        raise FileNotFoundError(prompts_path)

    base_model = hp["base_model"]
    common = dict(
        base_model=base_model,
        prompts_path=prompts_path,
        device=args.device,
        guidance_scale=hp["guidance_scale"],
        num_inference_steps=hp["num_inference_steps"],
        num_samples=1,
        from_case=args.from_case,
    )

    models = args.only or list(OUTPUT_MODELS)

    if "sd14_base" in models:
        print("Generating sd14_base …")
        _run_generate(**common, output_dir=paths.output_dir("sd14_base"), esd_path=None)

    if "esd" in models:
        esd_rel = ckpt.get("esd")
        if not esd_rel:
            raise ValueError("checkpoints.esd is not set in results.json")
        esd_path = ERASING_ROOT / esd_rel
        if not esd_path.is_file():
            raise FileNotFoundError(esd_path)
        print("Generating esd …")
        _run_generate(**common, output_dir=paths.output_dir("esd"), esd_path=esd_path)

    if "rl" in models:
        rl_rel = ckpt.get("rl")
        if not rl_rel:
            raise ValueError("checkpoints.rl is not set in results.json")
        rl_path = ERASING_ROOT / rl_rel
        if not rl_path.is_file():
            raise FileNotFoundError(rl_path)
        print("Generating rl …")
        _run_generate(**common, output_dir=paths.output_dir("rl"), esd_path=rl_path)

    print(f"Done. Images under {paths.outputs_dir}")


if __name__ == "__main__":
    main()
