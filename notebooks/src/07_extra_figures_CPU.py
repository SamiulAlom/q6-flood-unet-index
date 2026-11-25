"""Two descriptive figures added after results at the human's request (2026-11-24, plan §10 D5).

Reads only files already published by 02 and 04; computes nothing new except differences of stored
values for plotting.
- fig_event_diff: per-event paired IoU difference (each U-Net minus tuned index), with the stored
  mean and 95% event-bootstrap interval (outputs/stats/results.json all_water.primary, .secondary[0]).
- fig_thresholds: tuned MNDWI and NDWI thresholds per fold, the index chosen on the validation event
  marked (outputs/index/fold_thresholds.csv).

Run:  python notebooks/src/07_extra_figures_CPU.py
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p6_common import OUTPUTS

COLOURS = {"index_tuned": "#8c6d31", "unet_r34_scratch": "#2b6f97", "unet_r50_ssl4eo": "#b5523b"}
LABELS = {"unet_r34_scratch": "U-Net (ResNet-34, scratch)", "unet_r50_ssl4eo": "U-Net (ResNet-50, SSL4EO)"}
IDX = {"MNDWI": "#7a5195", "NDWI": "#3a923a"}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def contrast(results, a):
    aw = results["all_water"]
    cands = [aw["primary"]] + list(aw["secondary"])
    hit = [c for c in cands if c["a"] == a and c["b"] == "index_tuned"]
    assert len(hit) == 1, a
    return hit[0]


def fig_event_diff(plt, results, out):
    c34, c50 = contrast(results, "unet_r34_scratch"), contrast(results, "unet_r50_ssl4eo")
    events = sorted(c34["per_event"], key=lambda e: c34["per_event"][e])
    assert set(events) == set(c50["per_event"]) and len(events) == 11
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    y = np.arange(len(events))
    ax.axvline(0, color="#888888", lw=0.7)
    for m, c, dy, mk in (("unet_r34_scratch", c34, -0.15, "o"), ("unet_r50_ssl4eo", c50, 0.15, "s")):
        ax.scatter([c["per_event"][e] for e in events], y + dy, s=12, marker=mk, color=COLOURS[m], label=LABELS[m], zorder=3)
        yy = len(events) + (0.6 if m == "unet_r34_scratch" else 1.2)
        lo, hi = c["ci95"]
        ax.errorbar(c["mean"], yy, xerr=[[c["mean"] - lo], [hi - c["mean"]]], fmt=mk, ms=4,
                    color=COLOURS[m], capsize=2, lw=1)
    ax.set_yticks(list(y) + [len(events) + 0.9], [e.replace("-", " ") for e in events] + ["Mean (95% CI)"])
    ax.set_xlabel("IoU difference, U-Net minus tuned index")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_event_diff.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_thresholds(plt, rows, out):
    rows = sorted(rows, key=lambda r: r["test"])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = np.arange(len(rows))
    for k, dx in (("MNDWI", -0.12), ("NDWI", 0.12)):
        t = [float(r[f"t_{k}"]) for r in rows]
        chosen = [r["chosen"] == k for r in rows]
        ax.scatter(x[chosen] + dx, np.array(t)[chosen], s=22, color=IDX[k], label=f"{k} (chosen)", zorder=3)
        ax.scatter(x[~np.array(chosen)] + dx, np.array(t)[~np.array(chosen)], s=22, facecolors="none",
                   edgecolors=IDX[k], label=f"{k} (not chosen)", zorder=3)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xticks(x, [r["test"].replace("-", " ") for r in rows], rotation=45, ha="right")
    ax.set_xlabel("Held-out (test) event of the fold")
    ax.set_ylabel("Tuned threshold t")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_thresholds.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    results = json.loads((OUTPUTS / "stats" / "results.json").read_text(encoding="utf-8"))
    assert results["smoke"] is False
    with open(OUTPUTS / "index" / "fold_thresholds.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 11
    out = OUTPUTS / "figures"
    plt = pyplot()
    fig_event_diff(plt, results, out)
    fig_thresholds(plt, rows, out)
    print("wrote fig_event_diff and fig_thresholds to", out)


if __name__ == "__main__":
    main()
