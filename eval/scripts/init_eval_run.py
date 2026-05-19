#!/usr/bin/env python3
"""Create eval/results/<negX_iterY>/ with outputs/ subdirs and results.json."""

from __future__ import annotations

import argparse

from _paths import ERASING_ROOT  # noqa: F401 — adds erasing/ to sys.path

from eval.checkpoints import checkpoint_paths
from eval.layout import ESD_MODELS_SD, OUTPUT_MODELS, experiment_paths
from eval.results_io import make_results_template, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize an eval experiment directory.")
    parser.add_argument("--negative-guidance", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--erase-concept", type=str, default="teddy bear")
    parser.add_argument("--prompts-path", type=str, required=True, help="CSV path relative to erasing/")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--base-model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    paths = experiment_paths(args.negative_guidance, args.iterations)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    for model in OUTPUT_MODELS:
        paths.output_dir(model).mkdir(parents=True, exist_ok=True)

    ESD_MODELS_SD.mkdir(parents=True, exist_ok=True)

    esd_rel, rl_rel = checkpoint_paths()
    results = make_results_template(
        negative_guidance=args.negative_guidance,
        iterations=args.iterations,
        erase_concept=args.erase_concept,
        base_model=args.base_model,
        num_images=args.num_images,
        prompts_path=args.prompts_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        esd_checkpoint=esd_rel,
        rl_checkpoint=rl_rel,
    )
    save_results(paths.results_path, results)

    print(f"Created {paths.run_dir}")
    print(f"  results: {paths.results_path}")
    print(f"  checkpoints: esd={esd_rel}  rl={rl_rel}")
    print(f"  outputs: {paths.outputs_dir}/{{{', '.join(OUTPUT_MODELS)}}}")


if __name__ == "__main__":
    main()
