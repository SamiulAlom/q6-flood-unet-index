"""Statistics and result figures (analysis_plan.md v1.1, §6-8).

Inputs : outputs/index/ and outputs/train/<model>/<event>/seed<k>/ (each with run_record.json)
Outputs: outputs/stats/results.json, event_metrics.csv, chip_metrics.csv,
         outputs/figures/fig_event_iou, fig_cloud_strata, fig_examples (.pdf, .png),
         outputs/MANIFEST.csv

Definitions
- Event IoU: water IoU pooled over the scored pixels of a held-out event, per seed, then
  averaged over seeds. Precision, recall and F1 are pooled the same way.
- Primary summary: unweighted mean of the event IoUs (one vote per event).
- Pooled IoU (§6 secondary): IoU over all scored pixels of all 441 chips, averaged over seeds.
- Chip-weighted summary (§8.2): mean of event IoUs weighted by the event's chip count.
- Undefined metrics (zero denominator) are written as null.

Run:  python notebooks/src/04_stats_figures_CPU.py [--smoke]
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p6_common import (EVENTS, INTERIM, LEARNED, OUTPUTS, SCOPES, STRATA, data_provenance, folds,
                       load_chip_table, prf, production_design, provenance, smoke_plan, water_indices,
                       write_json)

PRIMARY = ("unet_r34_scratch", "index_tuned")
SECONDARY = (("unet_r50_ssl4eo", "index_tuned"), ("unet_r50_ssl4eo", "unet_r34_scratch"))
INDEX_SENSITIVITY = ("index_mndwi_t0", "index_otsu_mndwi", "index_otsu_ndwi")
INDEX_RULES = ("index_tuned",) + INDEX_SENSITIVITY
N_BOOT = 10_000
BOOT_SEED = 20261124
LABELS = {"index_tuned": "Tuned MNDWI/NDWI", "unet_r34_scratch": "U-Net (ResNet-34, scratch)",
          "unet_r50_ssl4eo": "U-Net (ResNet-50, SSL4EO)"}


# ---------------------------------------------------------------- loading and validation
def load_counts(root, expected_tests, seeds, epochs, chips_all, expected_test_chips):
    """Read every chip_counts.csv; require each model, seed, scope and test chip exactly once, rows
    whose test_event matches the chip's event, and records built from the current data."""
    counts = {}
    event_of = {c["chip_id"]: c["event"] for c in chips_all}
    data_now = data_provenance(provenance())

    def add(path, event=None):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["model"], int(r["seed"]), r["scope"], r["chip_id"])
                assert key not in counts, f"duplicate row {key}"
                assert r["test_event"] == event_of[r["chip_id"]], f"event mismatch {key}"
                assert event is None or r["test_event"] == event, f"row from another fold {key}"
                counts[key] = (int(r["tp"]), int(r["fp"]), int(r["fn"]))

    record = json.loads((root / "index" / "run_record.json").read_text(encoding="utf-8"))
    assert record["complete"], "index run incomplete"
    assert data_provenance(record["provenance"]) == data_now, "index run built from other data"
    assert sorted(record["test_chips"]) == sorted(expected_test_chips), "index test chips differ from design"
    add(root / "index" / "chip_counts.csv")
    for model in LEARNED:
        for event in expected_tests:
            for seed in seeds:
                d = root / "train" / model / event / f"seed{seed}"
                r = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
                ident = r["identity"]
                assert r["complete"] and ident["model"] == model and ident["seed"] == seed and ident["test"] == event
                assert ident["epochs"] == epochs, f"{d}: epochs {ident['epochs']} != design {epochs}"
                assert data_provenance(ident["provenance"]) == data_now, f"{d}: built from other data"
                add(d / "chip_counts.csv", event)
    test_chips = sorted(expected_test_chips)
    for model in INDEX_RULES:
        for scope in SCOPES:
            got = sorted(k[3] for k in counts if k[0] == model and k[2] == scope)
            assert got == test_chips, f"{model} {scope}: chip coverage differs"
    for model in LEARNED:
        for seed in seeds:
            for scope in SCOPES:
                got = sorted(k[3] for k in counts if k[0] == model and k[1] == seed and k[2] == scope)
                assert got == test_chips, f"{model} seed {seed} {scope}: chip coverage differs"
    assert len(counts) == len(test_chips) * len(SCOPES) * (len(INDEX_RULES) + len(LEARNED) * len(seeds))
    return counts, test_chips


def seeds_of(model, seeds):
    return [0] if model in INDEX_RULES else list(seeds)


def pooled(counts, model, seed, scope, chip_ids):
    tp = fp = fn = 0
    for c in chip_ids:
        a, b, d = counts[(model, seed, scope, c)]
        tp, fp, fn = tp + a, fp + b, fn + d
    return tp, fp, fn


def iou_of(tp, fp, fn):
    return tp / (tp + fp + fn) if tp + fp + fn else np.nan


def mean_over_seeds(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


class Study:
    def __init__(self, counts, chips, scope, seeds):
        self.counts, self.chips, self.scope, self.seeds = counts, chips, scope, seeds
        self.events = [e for e in EVENTS if any(c["event"] == e for c in chips)]
        self.by_event = {e: [c["chip_id"] for c in chips if c["event"] == e] for e in self.events}
        self.models = INDEX_RULES + LEARNED

    def iou(self, model, chip_ids):
        return mean_over_seeds([iou_of(*pooled(self.counts, model, s, self.scope, chip_ids))
                                for s in seeds_of(model, self.seeds)])

    def seed_metrics(self, model, chip_ids):
        """Per-seed pooled metrics (one row per seed)."""
        out = []
        for s in seeds_of(model, self.seeds):
            tp, fp, fn = pooled(self.counts, model, s, self.scope, chip_ids)
            p, r, f = prf(tp, fp, fn)
            out.append({"seed": s, "tp": tp, "fp": fp, "fn": fn, "iou": iou_of(tp, fp, fn),
                        "precision": p, "recall": r, "f1": f})
        return out

    def metrics(self, model, chip_ids):
        per_seed = [pooled(self.counts, model, s, self.scope, chip_ids) for s in seeds_of(model, self.seeds)]
        p, r, f = zip(*[prf(*x) for x in per_seed])
        return {"iou": mean_over_seeds([iou_of(*x) for x in per_seed]), "precision": mean_over_seeds(p),
                "recall": mean_over_seeds(r), "f1": mean_over_seeds(f),
                "iou_seed_sd": float(np.std([iou_of(*x) for x in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}

    def event_table(self):
        return {m: {e: self.iou(m, self.by_event[e]) for e in self.events} for m in self.models}


# ---------------------------------------------------------------- statistics
def contrast(table, a, b, events, rng, decide):
    """Paired event contrast. Undefined event IoUs are kept as null; summaries use the defined
    pairs only and report their number; the primary decision needs every pair defined."""
    all_diffs = {e: table[a][e] - table[b][e] for e in events}
    defined = [e for e in events if not np.isnan(all_diffs[e])]
    diffs = np.array([all_diffs[e] for e in defined])
    out = {"a": a, "b": b, "per_event": all_diffs, "defined_pairs": len(defined), "events": len(events),
           "interval": "paired event bootstrap, percentile, approximate and conditional on the fitted folds"}
    if len(diffs):
        idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
        boots = diffs[idx].mean(1)
        out.update(mean=float(diffs.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   leave_one_event_out_means={e: float(np.delete(diffs, i).mean()) if len(diffs) > 1 else None
                                              for i, e in enumerate(defined)})
    else:
        out.update(mean=None, ci95=[None, None], leave_one_event_out_means={})
    if decide:
        out["better_on_unseen_events"] = (out["ci95"][0] > 0) if len(defined) == len(events) else None
    return out


def slope(d):
    ok = ~np.isnan(d)
    return float(np.polyfit(np.arange(len(d))[ok], d[ok], 1)[0]) if ok.sum() >= 2 else np.nan


def strata_analysis(study, a, b, rng):
    """Per-stratum pooled IoU differences, the slope across strata, and the within-event slope
    (exploratory). Bootstrap: resample events, then chips within each drawn event occurrence."""
    stratum = {c["chip_id"]: c["stratum"] for c in study.chips}

    def diffs(chip_ids):
        out = []
        for n in STRATA:
            ids = [c for c in chip_ids if stratum[c] == n]
            out.append(study.iou(a, ids) - study.iou(b, ids) if ids else np.nan)
        return np.array(out)

    all_ids = [c["chip_id"] for c in study.chips]
    spanning = [e for e in study.events if len({stratum[c] for c in study.by_event[e]}) >= 2]
    point_within = float(np.nanmean([slope(diffs(study.by_event[e])) for e in spanning])) if spanning else np.nan

    boot_d, boot_s, boot_w = [], [], []
    for _ in range(N_BOOT):
        draws = [list(rng.choice(study.by_event[e], size=len(study.by_event[e]), replace=True))
                 for e in rng.choice(study.events, size=len(study.events), replace=True)]
        d = diffs([c for occ in draws for c in occ])
        boot_d.append(d)
        boot_s.append(slope(d))
    for _ in range(N_BOOT if spanning else 0):
        occ_slopes = []
        for e in rng.choice(spanning, size=len(spanning), replace=True):
            occ = list(rng.choice(study.by_event[e], size=len(study.by_event[e]), replace=True))
            occ_slopes.append(slope(diffs(occ)))
        occ_slopes = [s for s in occ_slopes if not np.isnan(s)]
        boot_w.append(float(np.mean(occ_slopes)) if occ_slopes else np.nan)
    boot_d, boot_s, boot_w = np.array(boot_d), np.array(boot_s), np.array(boot_w)

    def ci(x):
        x = x[~np.isnan(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]

    water = {n: sum(int(c["water_pixels"]) for c in study.chips if stratum[c["chip_id"]] == n) for n in STRATA}
    point = diffs(all_ids)
    return {
        "a": a, "b": b, "label": "exploratory",
        "per_stratum": {n: {"diff": float(point[i]), "ci95": ci(boot_d[:, i]),
                            "usable_replicates": int((~np.isnan(boot_d[:, i])).sum()),
                            "iou_a": study.iou(a, [c for c in all_ids if stratum[c] == n]),
                            "iou_b": study.iou(b, [c for c in all_ids if stratum[c] == n]),
                            "chips": sum(stratum[c] == n for c in all_ids),
                            "events": len({c["event"] for c in study.chips if c["stratum"] == n}),
                            "water_pixels": water[n]} for i, n in enumerate(STRATA)},
        "slope_across_strata": {"estimate": slope(point), "ci95": ci(boot_s),
                                "usable_replicates": int((~np.isnan(boot_s)).sum())},
        "within_event_slope": {"estimate": point_within, "ci95": ci(boot_w), "events": spanning,
                               "usable_replicates": int((~np.isnan(boot_w)).sum()),
                               "undefined_slopes": "a slope needs at least 2 strata with a defined IoU difference; event occurrences without that are dropped"},
        "bootstrap_seed": BOOT_SEED, "replicates": N_BOOT,
    }


# ---------------------------------------------------------------- figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


COLOURS = {"index_tuned": "#8c6d31", "unet_r34_scratch": "#2b6f97", "unet_r50_ssl4eo": "#b5523b"}


def fig_event_iou(plt, table, events, out_dir):
    order = sorted(events, key=lambda e: table["index_tuned"][e])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(order))
    for m, dy in (("index_tuned", -0.18), ("unet_r34_scratch", 0.0), ("unet_r50_ssl4eo", 0.18)):
        ax.scatter([table[m][e] for e in order], y + dy, s=14, color=COLOURS[m], label=LABELS[m], zorder=3)
    ax.set_yticks(y, [e.replace("-", " ") for e in order])
    ax.set_xlabel("Water IoU on the held-out event")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_event_iou.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_strata(plt, strata, out_dir):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(STRATA))
    bars = (("index_tuned", strata["primary"], "iou_b", -0.2), ("unet_r34_scratch", strata["primary"], "iou_a", 0.0),
            ("unet_r50_ssl4eo", strata["unet_r50_ssl4eo"], "iou_a", 0.2))
    for m, source, key, dx in bars:
        ax.bar(x + dx, [source["per_stratum"][n][key] for n in STRATA], width=0.2, color=COLOURS[m], label=LABELS[m])
    ax.set_xticks(x, [n.replace('_', '\n') + f"\n(n={strata['primary']['per_stratum'][n]['chips']})" for n in STRATA])
    ax.set_ylabel("Pooled water IoU")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_cloud_strata.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def example_selection(study, table):
    """Fixed rule: best, median and worst event by the primary learned model (U-Net, scratch)
    event IoU; in each event the chip whose seed-0 U-Net IoU is the median (lower median for
    even counts). All three methods are shown for the same chips."""
    order = sorted(study.events, key=lambda e: (table["unet_r34_scratch"][e], e))
    picks = {"worst": order[0], "median": order[(len(order) - 1) // 2], "best": order[-1]}
    out = {}
    for tag, event in picks.items():
        scored = sorted((iou_of(*study.counts[("unet_r34_scratch", 0, "all_water", c)]), c)
                        for c in study.by_event[event]
                        if not np.isnan(iou_of(*study.counts[("unet_r34_scratch", 0, "all_water", c)])))
        out[tag] = {"event": event, "chip_id": scored[(len(scored) - 1) // 2][1]}
    return out


def fig_examples(plt, root, picks, chips, out_dir):
    ids = json.loads((INTERIM / "cache_record.json").read_text(encoding="utf-8"))["chip_ids"]
    s2 = np.load(INTERIM / "s2.npy", mmap_mode="r")
    label = np.load(INTERIM / "label.npy", mmap_mode="r")
    with open(root / "index" / "fold_thresholds.csv", encoding="utf-8") as fh:
        thresholds = {r["test"]: r for r in csv.DictReader(fh)}
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6))
    for row, tag in enumerate(("best", "median", "worst")):
        event, chip_id = picks[tag]["event"], picks[tag]["chip_id"]
        k = ids.index(chip_id)
        x = np.asarray(s2[k])
        lab = np.asarray(label[k])
        t = thresholds[event]
        index_pred = water_indices(x)[t["chosen"]] > float(t[f"t_{t['chosen']}"])
        learned = []
        for m in LEARNED:
            packed = np.load(root / "train" / m / event / "seed0" / "predictions_seed0.npz")[chip_id]
            learned.append(np.unpackbits(packed)[: 512 * 512].reshape(512, 512).astype(bool))
        rgb = np.clip(np.stack([x[3], x[2], x[1]], -1) / 3000.0, 0, 1)
        panels = (rgb, lab, index_pred, *learned)
        titles = ("True colour", "Hand label", "Tuned index", "U-Net scratch", "U-Net SSL4EO")
        for col, (img, title) in enumerate(zip(panels, titles)):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(img)
            else:
                shown = np.where(lab == -1, 0.5, img.astype(float) if col > 1 else (lab == 1).astype(float))
                ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
        axes[row, 0].set_ylabel(f"{tag}: {event.replace('-', ' ')}")
    from matplotlib.patches import Patch
    cmap = plt.get_cmap("Blues")
    fig.legend(handles=[Patch(color=cmap(1.0), label="water"), Patch(color=cmap(0.0), label="not water"),
                        Patch(color=cmap(0.5), label="unlabelled (not scored)")],
               loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_examples.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- outputs
def manifest(root):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.csv" and "smoke" not in p.relative_to(root).parts[:1]:
            rows.append((str(p.relative_to(root)).replace("\\", "/"), p.stat().st_size,
                         hashlib.sha256(p.read_bytes()).hexdigest()))
    with open(root / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("path,bytes,sha256\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = OUTPUTS / "smoke" if args.smoke else OUTPUTS
    chips_all = load_chip_table()
    if args.smoke:
        plan = smoke_plan(chips_all)
        tests, seeds, epochs = [f["test"] for f in plan["folds"]], plan["seeds"], plan["epochs"]
        expected = [c for f in plan["folds"] for c in f["chips"]["test"]]
    else:
        design = production_design()
        tests, seeds, epochs = [f["test"] for f in folds()], design["seeds"], design["epochs"]
        expected = [c["chip_id"] for c in chips_all]
    counts, test_chips = load_counts(root, tests, seeds, epochs, chips_all, expected)
    if not args.smoke:
        assert len(test_chips) == 441 and set(tests) == set(EVENTS)
    chips = [c for c in chips_all if c["chip_id"] in set(test_chips)]
    stats_dir, fig_dir = root / "stats", root / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    plt = style()
    result = {"bootstrap_seed": BOOT_SEED, "replicates": N_BOOT, "smoke": args.smoke, "seeds": seeds,
              "epochs": epochs, "provenance": provenance(),
              "chips": [{"chip_id": c["chip_id"], "event": c["event"], "stratum": c["stratum"],
                         "cloud_pixels_scored": int(c["cloud_pixels_scored"]),
                         "scored_pixels": int(c["scored_pixels"])} for c in chips_all]}
    event_rows, chip_rows = [], []
    for scope in SCOPES:
        study = Study(counts, chips, scope, seeds)
        table = study.event_table()
        weights = {e: len(study.by_event[e]) for e in study.events}
        res = {"events": study.events, "event_iou": table,
               "summary_mean_event_iou": {m: float(np.mean(list(table[m].values()))) for m in table},
               "chip_count_weighted_mean_event_iou": {
                   m: float(sum(table[m][e] * weights[e] for e in study.events) / sum(weights.values())) for m in table},
               "pooled_iou_all_chips": {m: study.iou(m, [c["chip_id"] for c in chips]) for m in table}}
        is_primary = scope == "all_water"
        res["primary"] = contrast(table, *PRIMARY, study.events, rng, decide=is_primary)
        res["primary"]["label"] = "primary" if is_primary else "sensitivity (flood water only)"
        res["secondary"] = [dict(contrast(table, a, b, study.events, rng, decide=False), label="exploratory")
                            for a, b in SECONDARY]
        res["index_sensitivity"] = [dict(contrast(table, "unet_r34_scratch", b, study.events, rng, decide=False),
                                         label="sensitivity") for b in INDEX_SENSITIVITY]
        for m in study.models:
            for e in study.events:
                event_rows.append({"scope": scope, "model": m, "event": e, "seed": "mean", "chips": weights[e],
                                   **study.metrics(m, study.by_event[e])})
                for row in study.seed_metrics(m, study.by_event[e]):
                    event_rows.append({"scope": scope, "model": m, "event": e, "chips": weights[e], **row,
                                       "iou_seed_sd": ""})
            for c in chips:
                for s in seeds_of(m, seeds):
                    tp, fp, fn = counts[(m, s, scope, c["chip_id"])]
                    p, r, f = prf(tp, fp, fn)
                    chip_rows.append({"scope": scope, "model": m, "seed": s, "event": c["event"],
                                      "chip_id": c["chip_id"], "stratum": c["stratum"], "tp": tp, "fp": fp, "fn": fn,
                                      "iou": iou_of(tp, fp, fn), "precision": p, "recall": r, "f1": f})
        if is_primary:
            strata = {"primary": strata_analysis(study, *PRIMARY, rng),
                      "unet_r50_ssl4eo": strata_analysis(study, "unet_r50_ssl4eo", "index_tuned", rng)}
            res["cloud_strata"] = strata
            res["examples"] = example_selection(study, table)
            fig_event_iou(plt, table, study.events, fig_dir)
            fig_strata(plt, strata, fig_dir)
            fig_examples(plt, root, res["examples"], chips, fig_dir)
        result[scope] = res
    write_json(stats_dir / "results.json", result)
    for name, rows in (("event_metrics.csv", event_rows), ("chip_metrics.csv", chip_rows)):
        with open(stats_dir / name, "w", newline="", encoding="utf-8") as fh:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(fh, fieldnames=fields, restval="")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    manifest(root)
    print(f"stats written to {stats_dir}")


if __name__ == "__main__":
    main()
