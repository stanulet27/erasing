#!/usr/bin/env python3
"""Render a 3-panel sweep figure: erasure vs degeneration tradeoff plus heatmaps.

Reads ``erasing/eval/results/neg*_iter*/results.json`` and writes one PNG with:

1. Pareto scatter: ESD ensemble confidence (lower=better erasure) vs FID
   degeneration % vs baseline (lower=less collateral damage). One marker per
   sweep cell, colored by ``iterations``, sized by ``negative_guidance``.
2. Heatmap of ESD ensemble confidence over (neg, iter).
3. Heatmap of degeneration % over (neg, iter).

Usage::

    pip install matplotlib  # one-off
    python erasing/eval/scripts/plot_sweep.py
    python erasing/eval/scripts/plot_sweep.py --out sweep.png --show
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RUN_NAME_RE = re.compile(r"^neg(?P<neg>[\d.eE+\-]+)_iter(?P<it>\d+)$")


def _load_cells(root: Path) -> list[dict]:
    cells: list[dict] = []
    for path in sorted(root.glob("neg*_iter*/results.json")):
        m = RUN_NAME_RE.match(path.parent.name)
        if not m:
            continue
        with path.open(encoding="utf-8") as f:
            r = json.load(f)
        cells.append(
            {
                "neg": float(m["neg"]),
                "iter": int(m["it"]),
                "ens_base": r["ensemble"]["sd14_base"]["ensemble"]["mean_conf"],
                "ens_esd": r["ensemble"]["esd"]["ensemble"]["mean_conf"],
                "fid_base": r["fid_vs_coco"]["sd14_base"],
                "fid_esd": r["fid_vs_coco"]["esd"],
                "deg_esd": r["fid_vs_coco"]["degeneration_pct_vs_sd14_base"]["esd"],
            }
        )
    return cells


def _grid(cells: list[dict], key: str) -> tuple[list[float], list[int], list[list[float]]]:
    negs = sorted({c["neg"] for c in cells})
    iters = sorted({c["iter"] for c in cells})
    lookup = {(c["neg"], c["iter"]): c[key] for c in cells}
    z = [[lookup.get((n, i), float("nan")) for n in negs] for i in iters]
    return negs, iters, z


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot eval sweep results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval" / "results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "eval" / "results" / "sweep.png",
    )
    parser.add_argument("--show", action="store_true", help="Also open the figure window")
    args = parser.parse_args()

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        return 1

    cells = _load_cells(args.results_root)
    if not cells:
        print(f"No results.json under {args.results_root}", file=sys.stderr)
        return 1

    if not args.show:
        matplotlib.use("Agg")

    fig, (ax_pareto, ax_h1, ax_h2) = plt.subplots(1, 3, figsize=(16, 5))

    iters_all = sorted({c["iter"] for c in cells})
    iter_cmap = plt.get_cmap("viridis")
    iter_to_color = {it: iter_cmap(i / max(1, len(iters_all) - 1)) for i, it in enumerate(iters_all)}

    negs_all = sorted({c["neg"] for c in cells})
    neg_min, neg_max = min(negs_all), max(negs_all)
    if neg_max == neg_min:
        size_scale = lambda _ng: 120
    else:
        size_scale = lambda ng: 60 + 240 * (ng - neg_min) / (neg_max - neg_min)

    for c in cells:
        ax_pareto.scatter(
            c["ens_esd"],
            c["deg_esd"],
            s=size_scale(c["neg"]),
            color=iter_to_color[c["iter"]],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax_pareto.annotate(
            f"n{c['neg']:g}/i{c['iter']}",
            (c["ens_esd"], c["deg_esd"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    if cells:
        base_ens = cells[0]["ens_base"]
        ax_pareto.axvline(base_ens, color="red", linestyle="--", linewidth=1, label=f"sd14_base ens={base_ens:.3f}")
        ax_pareto.legend(loc="upper left", fontsize=8)
    ax_pareto.set_xlabel("ESD ensemble mean_conf  (lower = better erasure)")
    ax_pareto.set_ylabel("FID degeneration % vs sd14_base  (lower = less damage)")
    ax_pareto.set_title("Erasure / Degeneration tradeoff\n(color = iterations, size = negative_guidance)")
    ax_pareto.grid(True, alpha=0.3)

    negs, iters, z_ens = _grid(cells, "ens_esd")
    im1 = ax_h1.imshow(z_ens, aspect="auto", origin="lower", cmap="viridis")
    ax_h1.set_xticks(range(len(negs)))
    ax_h1.set_xticklabels([f"{n:g}" for n in negs])
    ax_h1.set_yticks(range(len(iters)))
    ax_h1.set_yticklabels([str(i) for i in iters])
    ax_h1.set_xlabel("negative_guidance")
    ax_h1.set_ylabel("iterations")
    ax_h1.set_title("ESD ensemble mean_conf\n(darker = more erased)")
    plt.colorbar(im1, ax=ax_h1, shrink=0.8)
    for i, row in enumerate(z_ens):
        for j, v in enumerate(row):
            if v == v:  # not NaN
                ax_h1.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)

    _, _, z_deg = _grid(cells, "deg_esd")
    im2 = ax_h2.imshow(z_deg, aspect="auto", origin="lower", cmap="magma")
    ax_h2.set_xticks(range(len(negs)))
    ax_h2.set_xticklabels([f"{n:g}" for n in negs])
    ax_h2.set_yticks(range(len(iters)))
    ax_h2.set_yticklabels([str(i) for i in iters])
    ax_h2.set_xlabel("negative_guidance")
    ax_h2.set_ylabel("iterations")
    ax_h2.set_title("FID degeneration %\n(darker = less damage)")
    plt.colorbar(im2, ax=ax_h2, shrink=0.8)
    for i, row in enumerate(z_deg):
        for j, v in enumerate(row):
            if v == v:
                ax_h2.text(j, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
