#!/usr/bin/env python3
"""Score generated images with YOLO / RT-DETR / FRCNN and update results.json."""

from __future__ import annotations

import argparse

from _paths import ERASING_ROOT  # noqa: F401

from eval.layout import OUTPUT_MODELS, experiment_paths
from eval.ensemble_scores import load_ensemble, mean_scores_for_folder
from eval.results import load_results, save_results, update_ensemble_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ensemble detectors on eval output folders.")
    parser.add_argument("--negative-guidance", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--only",
        choices=OUTPUT_MODELS,
        nargs="+",
        default=None,
    )
    args = parser.parse_args()

    paths = experiment_paths(args.negative_guidance, args.iterations)
    results = load_results(paths.results_path)
    target_class = results["ensemble"]["target_class"]

    models = args.only or list(OUTPUT_MODELS)
    ensemble = load_ensemble()

    for model in models:
        folder = paths.output_dir(model)
        print(f"Scoring {model} ({folder}) …")
        means, n = mean_scores_for_folder(ensemble, folder, target_class)
        update_ensemble_scores(results, model, means)
        print(f"  n={n} ensemble mean_conf={means['ensemble']:.4f}")

    save_results(paths.results_path, results)
    print(f"Updated {paths.results_path}")


if __name__ == "__main__":
    main()
