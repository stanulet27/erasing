#!/usr/bin/env python3
"""Compute FID vs COCO for all three output folders and update results.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from _paths import ERASING_ROOT  # noqa: F401

from eval.fid import compute_fid
from eval.layout import OUTPUT_MODELS, experiment_paths
from eval.results_io import load_results, save_results, update_fid_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID vs COCO and update results.json.")
    parser.add_argument("--negative-guidance", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--coco-dir",
        type=str,
        required=True,
        help="Reference images (e.g. COCO val2017 folder)",
    )
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    paths = experiment_paths(args.negative_guidance, args.iterations)
    results = load_results(paths.results_path)

    scores = {}
    for model in OUTPUT_MODELS:
        folder = paths.output_dir(model)
        print(f"FID {model}: {folder}")
        scores[model] = compute_fid(folder, coco_dir)
        print(f"  -> {scores[model]:.4f}")

    update_fid_scores(
        results,
        sd14_base=scores["sd14_base"],
        esd=scores["esd"],
        rl=scores["rl"],
    )
    save_results(paths.results_path, results)

    deg = results["fid_vs_coco"]["degeneration_pct_vs_sd14_base"]
    print(f"Updated {paths.results_path}")
    print(f"  degeneration vs sd14_base: esd={deg['esd']:.2f}%  rl={deg['rl']:.2f}%")


if __name__ == "__main__":
    main()
