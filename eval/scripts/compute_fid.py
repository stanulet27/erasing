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
    parser.add_argument(
        "--only",
        choices=OUTPUT_MODELS,
        nargs="+",
        default=None,
        help="Compute FID only for these models (default: all three)",
    )
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    paths = experiment_paths(args.negative_guidance, args.iterations)
    results = load_results(paths.results_path)

    models = args.only or list(OUTPUT_MODELS)
    scores = {}
    for model in models:
        folder = paths.output_dir(model)
        print(f"FID {model}: {folder}")
        scores[model] = compute_fid(folder, coco_dir)
        print(f"  -> {scores[model]:.4f}")

    update_fid_scores(
        results,
        sd14_base=scores.get("sd14_base"),
        esd=scores.get("esd"),
        rl=scores.get("rl"),
    )
    save_results(paths.results_path, results)

    deg = results["fid_vs_coco"]["degeneration_pct_vs_sd14_base"]
    parts = [f"{m}={deg[m]:.2f}%" for m in ("esd", "rl") if deg.get(m) is not None]
    print(f"Updated {paths.results_path}")
    if parts:
        print(f"  degeneration vs sd14_base: {'  '.join(parts)}")


if __name__ == "__main__":
    main()
