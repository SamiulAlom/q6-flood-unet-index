"""Step 3 inventory of the Sen1Floods11 hand-labelled Sentinel-2 chips (no model, no result).

Writes to data/records/:
  chip_inventory.csv     one row per chip: event, bounds, valid and water pixel counts,
                         cloud fraction over the whole chip and over scored pixels, stratum
  band_check.txt         band count, descriptions and value ranges of the S2 files
  overlap_audit.csv      chip pairs whose footprints intersect across or within events,
                         and chips with identical image hashes
  strata_counts.csv      chips, events and valid water pixels per cloud stratum
  audit_png/             5 chips: true colour, cloud mask, label (for a visual check)

Cloud rule (analysis_plan.md §6, D2): s2cloudless default mask (probability > 0.4, default
post-processing),
all 13 L1C bands, reflectance = digital number / 10000. Strata from the fraction over
scored pixels (LabelHand != -1), from integer counts: clear < 5%, partly cloudy 5-30%
(both ends included), cloudy > 30%.

Run:  python notebooks/src/01_inventory.py
"""
import collections
import csv
import hashlib
import itertools
import pathlib

import numpy as np
import rasterio
from s2cloudless import S2PixelCloudDetector

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p6_common import stratum_from_counts

PROJECT = pathlib.Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw" / "sen1floods11_v1.1" / "data" / "flood_events" / "HandLabeled"
OUT = PROJECT / "data" / "records"
AUDIT_CHIPS = 5
STRATA = (("clear", 0.0, 0.05), ("partly_cloudy", 0.05, 0.30), ("cloudy", 0.30, 1.01))


def stratum(fraction):
    for name, low, high in STRATA:
        if low <= fraction < high:
            return name
    raise ValueError(fraction)


def read_chip(chip_id):
    with rasterio.open(RAW / "S2Hand" / f"{chip_id}_S2Hand.tif") as src:
        s2 = src.read().astype(np.float32)
        bounds, descriptions, dtype = src.bounds, src.descriptions, src.dtypes[0]
    with rasterio.open(RAW / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
        label = src.read(1)
    return s2, label, bounds, descriptions, dtype


def save_audit_png(chip_id, s2, cloud, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.clip(np.stack([s2[3], s2[2], s2[1]], -1) / 3000.0, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(rgb)
    axes[0].set_title("true colour (B4 B3 B2)")
    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(cloud == 0, cloud), cmap="autumn", alpha=0.5)
    axes[1].set_title("s2cloudless mask")
    axes[2].imshow(label, cmap="viridis", vmin=-1, vmax=1)
    axes[2].set_title("label (-1, 0, 1)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(chip_id)
    fig.tight_layout()
    (OUT / "audit_png").mkdir(exist_ok=True)
    fig.savefig(OUT / "audit_png" / f"{chip_id}.png", dpi=80)
    plt.close(fig)


def main():
    chip_ids = sorted(p.name.removesuffix("_S2Hand.tif") for p in (RAW / "S2Hand").glob("*_S2Hand.tif"))
    detector = S2PixelCloudDetector(threshold=0.4, all_bands=True)
    rows, image_hash = [], collections.defaultdict(list)
    band_info, ranges = None, []
    rng = np.random.default_rng(20261124)
    audit = set(rng.choice(chip_ids, size=AUDIT_CHIPS, replace=False))
    for chip_id in chip_ids:
        s2, label, bounds, descriptions, dtype = read_chip(chip_id)
        if band_info is None:
            band_info = (s2.shape, descriptions, dtype)
        ranges.append(np.nanpercentile(s2.reshape(s2.shape[0], -1), [1, 99], axis=1))
        image_hash[hashlib.sha256(s2.tobytes()).hexdigest()].append(chip_id)
        reflectance = np.nan_to_num(np.moveaxis(s2, 0, -1) / 10000.0)[None]
        cloud = detector.get_cloud_masks(reflectance)[0]
        scored = label != -1
        n_scored = int(scored.sum())
        cloud_all = float(cloud.mean())
        cloud_scored = float(cloud[scored].mean()) if n_scored else float("nan")
        rows.append({
            "chip_id": chip_id, "event": chip_id.split("_")[0],
            "left": bounds.left, "bottom": bounds.bottom, "right": bounds.right, "top": bounds.top,
            "pixels": label.size, "scored_pixels": n_scored, "water_pixels": int((label == 1).sum()),
            "nodata_fraction": round(1 - n_scored / label.size, 5),
            "cloud_pixels_chip": int(cloud.sum()),
            "cloud_pixels_scored": int(cloud[scored].sum()),
            "cloud_fraction_chip": round(cloud_all, 5),
            "cloud_fraction_scored": round(cloud_scored, 5) if n_scored else "",
            "stratum": stratum_from_counts(int(cloud[scored].sum()), n_scored) if n_scored else "no_scored_pixels",
            "s2_nan_pixels": int(np.isnan(s2).any(0).sum()),
        })
        if chip_id in audit:
            save_audit_png(chip_id, s2, cloud, label)

    with open(OUT / "chip_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranges = np.array(ranges)
    with open(OUT / "band_check.txt", "w", encoding="utf-8") as fh:
        fh.write(f"shape {band_info[0]} dtype {band_info[2]}\ndescriptions {band_info[1]}\n")
        fh.write("band | median 1st percentile | median 99th percentile\n")
        for b in range(ranges.shape[2]):
            fh.write(f"{b + 1} | {np.median(ranges[:, 0, b]):.0f} | {np.median(ranges[:, 1, b]):.0f}\n")

    overlaps = []
    for a, b in itertools.combinations(rows, 2):
        if a["left"] < b["right"] and b["left"] < a["right"] and a["bottom"] < b["top"] and b["bottom"] < a["top"]:
            w = min(a["right"], b["right"]) - max(a["left"], b["left"])
            h = min(a["top"], b["top"]) - max(a["bottom"], b["bottom"])
            area = (a["right"] - a["left"]) * (a["top"] - a["bottom"])
            overlaps.append({"chip_a": a["chip_id"], "chip_b": b["chip_id"], "kind": "footprint",
                             "same_event": a["event"] == b["event"], "share_of_a": round(w * h / area, 4)})
    for ids in image_hash.values():
        for a, b in itertools.combinations(ids, 2):
            overlaps.append({"chip_a": a, "chip_b": b, "kind": "identical_image",
                             "same_event": a.split("_")[0] == b.split("_")[0], "share_of_a": 1.0})
    with open(OUT / "overlap_audit.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chip_a", "chip_b", "kind", "same_event", "share_of_a"])
        writer.writeheader()
        writer.writerows(overlaps)

    strata = collections.defaultdict(lambda: {"chips": 0, "events": set(), "water_pixels": 0, "scored_pixels": 0})
    for r in rows:
        s = strata[r["stratum"]]
        s["chips"] += 1
        s["events"].add(r["event"])
        s["water_pixels"] += r["water_pixels"]
        s["scored_pixels"] += r["scored_pixels"]
    with open(OUT / "strata_counts.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("stratum,chips,events,event_names,scored_pixels,water_pixels\n")
        for name in [s[0] for s in STRATA] + ["no_scored_pixels"]:
            if name in strata:
                s = strata[name]
                fh.write(f"{name},{s['chips']},{len(s['events'])},{' '.join(sorted(s['events']))},"
                         f"{s['scored_pixels']},{s['water_pixels']}\n")

    print(f"chips {len(rows)}; overlaps across events "
          f"{sum(1 for o in overlaps if not o['same_event'])}; within events "
          f"{sum(1 for o in overlaps if o['same_event'])}; audit chips {sorted(audit)}")


if __name__ == "__main__":
    main()
