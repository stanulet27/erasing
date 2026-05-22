#!/usr/bin/env python3
"""Build a side-by-side comparison grid of sd14_base vs esd outputs.

Pulls images from the local copy of the eval-runs volume (run ``modal volume get``
first to populate it). For a given list of prompt case numbers and a list of
sweep cells, lays out a PNG with rows=prompts and columns=(sd14_base, esd@cell1,
esd@cell2, ...).

Usage::

    # 1. Pull the generated images down (large — ~500 MB per cell)
    modal volume get eval-runs neg2_iter200/outputs erasing/eval/results/neg2_iter200/outputs
    modal volume get eval-runs neg10_iter1000/outputs erasing/eval/results/neg10_iter1000/outputs

    # 2. Build a 3-prompt comparison figure
    python erasing/eval/scripts/extract_eval_images.py \\
        --cases 0 1 2 \\
        --cells neg2_iter200 neg10_iter1000

    # Or pick random prompts:
    python erasing/eval/scripts/extract_eval_images.py --random 4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ERASING_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ERASING_ROOT / "eval" / "results"
PROMPTS_CSV = ERASING_ROOT / "eval" / "prompts" / "bears_eval_500.csv"


def _load_prompts() -> dict[int, str]:
    out: dict[int, str] = {}
    with PROMPTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["case_number"])] = row["prompt"]
    return out


def _image_path(cell: str, model: str, case_number: int, sample_idx: int = 0) -> Path:
    return RESULTS_ROOT / cell / "outputs" / model / f"{case_number}_{sample_idx}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["neg2_iter200"],
        help="ESD cells to include as columns (e.g. neg2_iter200 neg10_iter1000)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help="Prompt case numbers (rows). If omitted, use --random.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Pick N random prompts (seeded)",
    )
    parser.add_argument(
        "--baseline-cell",
        type=str,
        default=None,
        help="Which cell to pull sd14_base from (default: first --cells entry)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT / "comparison.png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random",
    )
    parser.add_argument(
        "--max-width-chars",
        type=int,
        default=60,
        help="Wrap prompt text after N chars",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install matplotlib pillow", file=sys.stderr)
        return 1

    matplotlib.use("Agg")

    if not PROMPTS_CSV.is_file():
        print(f"Prompts CSV not found: {PROMPTS_CSV}", file=sys.stderr)
        return 1
    prompts = _load_prompts()

    if args.cases is None:
        if args.random is None:
            print("Provide either --cases or --random", file=sys.stderr)
            return 1
        rng = random.Random(args.seed)
        args.cases = sorted(rng.sample(sorted(prompts.keys()), args.random))

    baseline_cell = args.baseline_cell or args.cells[0]
    columns = [("sd14_base", baseline_cell)] + [("esd", c) for c in args.cells]
    n_rows = len(args.cases)
    n_cols = len(columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows + 0.6 * n_rows),
        squeeze=False,
    )

    for r, case in enumerate(args.cases):
        prompt = prompts.get(case, f"<case {case} not in CSV>")
        prompt_wrapped = "\n".join(
            prompt[i : i + args.max_width_chars]
            for i in range(0, len(prompt), args.max_width_chars)
        )
        for c, (model, cell) in enumerate(columns):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            path = _image_path(cell, model, case)
            if not path.is_file():
                ax.text(0.5, 0.5, f"missing\n{path.name}\n({cell}/{model})",
                        ha="center", va="center", color="red", fontsize=8)
            else:
                with Image.open(path) as img:
                    ax.imshow(img)
            if r == 0:
                label = "sd14_base" if model == "sd14_base" else f"esd\n{cell}"
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"case {case}\n{prompt_wrapped}", fontsize=8, rotation=0, ha="right", va="center")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}  ({n_rows} prompts × {n_cols} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
