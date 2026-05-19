#!/usr/bin/env python3
"""Record FID scores (computed externally) into results.json."""

from __future__ import annotations

import argparse

from _paths import ERASING_ROOT  # noqa: F401

from eval.layout import experiment_paths
from eval.results_io import load_results, save_results, update_fid_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write FID-vs-COCO scores and degeneration %% into results.json."
    )
    parser.add_argument("--negative-guidance", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--sd14-base", type=float, required=True, help="FID for baseline outputs")
    parser.add_argument("--esd", type=float, required=True)
    parser.add_argument("--rl", type=float, required=True)
    args = parser.parse_args()

    paths = experiment_paths(args.negative_guidance, args.iterations)
    results = load_results(paths.results_path)
    update_fid_scores(
        results,
        sd14_base=args.sd14_base,
        esd=args.esd,
        rl=args.rl,
    )
    save_results(paths.results_path, results)

    deg = results["fid_vs_coco"]["degeneration_pct_vs_sd14_base"]
    print(f"Updated {paths.results_path}")
    print(f"  degeneration vs sd14_base: esd={deg['esd']:.2f}%  rl={deg['rl']:.2f}%")


if __name__ == "__main__":
    main()
