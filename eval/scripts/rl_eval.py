#!/usr/bin/env python3
"""Standalone RL eval: sd14_base + rl images, ensemble scores, FID, results.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import ERASING_ROOT

from eval.fid import compute_fid
from eval.results_io import (
    make_results_template,
    save_results,
    update_ensemble_scores,
    update_fid_scores,
)
from eval.scripts.score_ensemble import load_ensemble, mean_scores_for_folder


def _run_generate(
    *,
    base_model: str,
    prompts_path: Path,
    output_dir: Path,
    rl_path: Path | None,
    device: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_samples: int,
    from_case: int,
) -> None:
    cmd = [
        sys.executable,
        str(ERASING_ROOT / "eval" / "scripts" / "generate-images.py"),
        "--base_model", base_model,
        "--prompts_path", str(prompts_path),
        "--output_dir", str(output_dir),
        "--device", device,
        "--guidance_scale", str(guidance_scale),
        "--num_inference_steps", str(num_inference_steps),
        "--num_samples", str(num_samples),
        "--from_case", str(from_case),
    ]
    if rl_path is not None:
        cmd.extend(["--rl_path", str(rl_path)])
    subprocess.run(cmd, check=True, cwd=ERASING_ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="RL-only eval (sd14_base + rl).")
    p.add_argument("--rl-checkpoint", type=Path, required=True,
                   help="Absolute path to RL LoRA checkpoint (e.g. /ddpo-logs/runs/.../checkpoint_500)")
    p.add_argument("--run-id", required=True, help="Identifier for output dir: eval/results/rl_<run_id>/")
    p.add_argument("--prompts-path", type=Path, required=True)
    p.add_argument("--erase-concept", default="teddy bear")
    p.add_argument("--base-model", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--output-root", type=Path, default=ERASING_ROOT / "eval" / "results")
    p.add_argument("--coco-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--num-images", type=int, default=500, help="For results.json hyperparams block")
    p.add_argument("--from-case", type=int, default=0)
    p.add_argument("--skip-base", action="store_true", help="Reuse existing sd14_base images")
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    args = p.parse_args()

    if not args.rl_checkpoint.is_file() and not args.rl_checkpoint.is_dir():
        raise FileNotFoundError(args.rl_checkpoint)
    if not args.prompts_path.is_file():
        raise FileNotFoundError(args.prompts_path)

    run_dir = args.output_root / f"rl_{args.run_id}"
    sd_dir = run_dir / "sd14_base"
    rl_dir = run_dir / "rl"
    sd_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)

    gen_common = dict(
        base_model=args.base_model,
        prompts_path=args.prompts_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
    )

    if not args.skip_base:
        print(f"Generating sd14_base → {sd_dir}")
        _run_generate(**gen_common, output_dir=sd_dir, rl_path=None)

    print(f"Generating rl → {rl_dir}")
    _run_generate(**gen_common, output_dir=rl_dir, rl_path=args.rl_checkpoint)

    results = make_results_template(
        negative_guidance=0.0,
        iterations=0,
        erase_concept=args.erase_concept,
        base_model=args.base_model,
        num_images=args.num_images,
        prompts_path=str(args.prompts_path),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        esd_checkpoint=None,
        rl_checkpoint=str(args.rl_checkpoint),
    )
    results["hyperparameters"]["rl_run_id"] = args.run_id

    if not args.skip_ensemble:
        print("Loading detector ensemble …")
        ensemble = load_ensemble()

        print("Scoring sd14_base ensemble …")
        sd_means, _ = mean_scores_for_folder(ensemble, sd_dir, args.erase_concept)
        update_ensemble_scores(results, "sd14_base", sd_means)

        print("Scoring rl ensemble …")
        rl_means, _ = mean_scores_for_folder(ensemble, rl_dir, args.erase_concept)
        update_ensemble_scores(results, "rl", rl_means)

    if not args.skip_fid and args.coco_dir and args.coco_dir.is_dir():
        print(f"FID sd14_base vs {args.coco_dir}")
        sd_fid = compute_fid(sd_dir, args.coco_dir)
        print(f"  = {sd_fid:.4f}")
        print(f"FID rl vs {args.coco_dir}")
        rl_fid = compute_fid(rl_dir, args.coco_dir)
        print(f"  = {rl_fid:.4f}")
        update_fid_scores(results, sd14_base=sd_fid, rl=rl_fid)

    results_path = run_dir / "results.json"
    save_results(results_path, results)
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()