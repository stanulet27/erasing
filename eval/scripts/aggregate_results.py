#!/usr/bin/env python3
"""Aggregate sweep results.json files into one table.

Walks a directory of eval runs (default: ``erasing/eval/results/``), reads each
``neg*_iter*/results.json``, and prints a sorted summary table of ensemble
detection means and FID scores per model.

Usage::

    python erasing/eval/scripts/aggregate_results.py
    python erasing/eval/scripts/aggregate_results.py --csv > sweep.csv
    python erasing/eval/scripts/aggregate_results.py --models sd14_base esd
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_MODELS = ("sd14_base", "esd", "rl")
RUN_NAME_RE = re.compile(r"^neg(?P<neg>[\d.eE+\-]+)_iter(?P<it>\d+)$")


def _find_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("neg*_iter*/results.json") if p.is_file())


def _parse_cell(run_dir_name: str) -> tuple[float, int]:
    m = RUN_NAME_RE.match(run_dir_name)
    if not m:
        return (float("inf"), -1)
    return (float(m["neg"]), int(m["it"]))


def _ens(results: dict, model: str) -> float | None:
    return results.get("ensemble", {}).get(model, {}).get("ensemble", {}).get("mean_conf")


def _fid(results: dict, model: str) -> float | None:
    return results.get("fid_vs_coco", {}).get(model)


def _deg(results: dict, model: str) -> float | None:
    return results.get("fid_vs_coco", {}).get("degeneration_pct_vs_sd14_base", {}).get(model)


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate eval sweep results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval" / "results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Subset of models to include as columns (default: all three)",
    )
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of markdown")
    args = parser.parse_args()

    runs = _find_runs(args.results_root)
    if not runs:
        print(f"No results.json under {args.results_root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for path in runs:
        with path.open(encoding="utf-8") as f:
            results = json.load(f)
        neg, it = _parse_cell(path.parent.name)
        row = {"neg": neg, "iter": it}
        for m in args.models:
            row[f"ens_{m}"] = _ens(results, m)
            row[f"fid_{m}"] = _fid(results, m)
        if "esd" in args.models:
            row["deg_esd"] = _deg(results, "esd")
        if "rl" in args.models:
            row["deg_rl"] = _deg(results, "rl")
        rows.append(row)

    rows.sort(key=lambda r: (r["neg"], r["iter"]))

    headers = ["neg", "iter"]
    for m in args.models:
        headers.extend([f"ens_{m}", f"fid_{m}"])
    if "esd" in args.models:
        headers.append("deg_esd%")
    if "rl" in args.models:
        headers.append("deg_rl%")

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(
                [f"{r['neg']:g}", r["iter"]]
                + [v for m in args.models for v in (r[f"ens_{m}"], r[f"fid_{m}"])]
                + ([r["deg_esd"]] if "esd" in args.models else [])
                + ([r["deg_rl"]] if "rl" in args.models else [])
            )
        return 0

    width = max(len(h) for h in headers)
    print("| " + " | ".join(h.ljust(width) for h in headers) + " |")
    print("|" + "|".join("-" * (width + 2) for _ in headers) + "|")
    for r in rows:
        cells = [f"{r['neg']:g}".ljust(width), str(r["iter"]).ljust(width)]
        for m in args.models:
            cells.append(_fmt(r[f"ens_{m}"]).ljust(width))
            cells.append(_fmt(r[f"fid_{m}"], digits=2).ljust(width))
        if "esd" in args.models:
            cells.append(_fmt(r.get("deg_esd"), digits=2).ljust(width))
        if "rl" in args.models:
            cells.append(_fmt(r.get("deg_rl"), digits=2).ljust(width))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
