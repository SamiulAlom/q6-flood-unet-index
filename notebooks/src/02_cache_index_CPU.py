"""Build the chip cache, the fold table and all water-index results (CPU).

1. Cache: S2 (int16), labels and JRC permanent water for the 441 kept chips, as
   memory-mapped arrays in data/interim/, rebuilt from the read-only raw files. A
   cache_record.json binds the cache to the raw-file hashes, the chip list and the grid.
2. Folds: fold_table.csv and one chip manifest per fold (production or the frozen smoke
   design from p6_common.smoke_plan).
3. Index baselines (analysis_plan.md §4.1 and §8.3):
   - tuned: per fold, threshold fitted on the fitting chips for MNDWI and NDWI (pooled
     water IoU compared exactly; ties to the threshold nearest 0, then the lower one);
     index chosen on the validation chips (ties to MNDWI);
   - MNDWI at t = 0, no tuning;
   - Otsu per chip on the chip's own scored, finite index values (MNDWI and NDWI).
   Pixels where an index is undefined (band sum 0) are predicted land by every rule.
   Per-chip confusion counts on the test chips, with and without JRC permanent water.

Outputs are written to a temporary folder and published in one step with run_record.json.

Run:  python notebooks/src/02_cache_index_CPU.py [--smoke]
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
import rasterio
import skimage
from skimage.filters import threshold_otsu

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p6_common import (BANDS, INTERIM, OUTPUTS, RAW, THRESHOLDS, best_by_exact_iou, cache_is_valid, confusion,
                       fold_chips,
                       folds, load_chip_table, provenance, publish, sha256_file, smoke_plan, water_indices,
                       write_json)

INDEX_RULES = ("index_tuned", "index_mndwi_t0", "index_otsu_mndwi", "index_otsu_ndwi")


def build_cache(chips, prov):
    """Write S2, label and JRC arrays from the raw files, checking grid and label domains."""
    ids = [c["chip_id"] for c in chips]
    if cache_is_valid(ids, prov):
        return
    INTERIM.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    s2 = np.lib.format.open_memmap(INTERIM / "s2.npy", "w+", np.int16, (n, 13, 512, 512))
    label = np.lib.format.open_memmap(INTERIM / "label.npy", "w+", np.int8, (n, 512, 512))
    jrc = np.lib.format.open_memmap(INTERIM / "jrc.npy", "w+", np.uint8, (n, 512, 512))
    for i, chip_id in enumerate(ids):
        grids = []
        with rasterio.open(RAW / "S2Hand" / f"{chip_id}_S2Hand.tif") as src:
            assert tuple(src.descriptions) == BANDS, src.descriptions
            s2[i] = src.read()
            grids.append((src.transform, src.crs, src.shape))
        with rasterio.open(RAW / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
            lab = src.read(1)
            grids.append((src.transform, src.crs, src.shape))
        with rasterio.open(RAW / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif") as src:
            water = src.read(1)
            grids.append((src.transform, src.crs, src.shape))
        assert all(g[1] == grids[0][1] and g[2] == grids[0][2] and g[0].almost_equals(grids[0][0], precision=1e-9)
                   for g in grids), f"grid mismatch in {chip_id}"
        assert set(np.unique(lab)) <= {-1, 0, 1}, f"label values in {chip_id}"
        assert set(np.unique(water)) <= {0, 1}, f"JRC values in {chip_id}"
        label[i], jrc[i] = lab, water
    s2.flush(), label.flush(), jrc.flush()
    del s2, label, jrc
    write_json(INTERIM / "cache_record.json", {
        "chip_ids": ids, "file_hashes_csv": prov["file_hashes_csv"],
        "array_sha256": {n: sha256_file(INTERIM / n) for n in ("s2.npy", "label.npy", "jrc.npy")}})


def threshold_counts(index, label):
    """(tp, fp, fn) at every grid threshold with the rule index > t, on scored pixels."""
    scored = label != -1
    water = np.sort(index[scored & (label == 1)])
    land = np.sort(index[scored & (label == 0)])
    tp = len(water) - np.searchsorted(water, THRESHOLDS, side="right")
    fp = len(land) - np.searchsorted(land, THRESHOLDS, side="right")
    return np.stack([tp, fp, len(water) - tp], 1).astype(np.int64)


def otsu(values):
    """Otsu threshold of the chip's scored finite index values; a constant chip uses its value."""
    if values.size == 0:
        return np.inf  # nothing scorable: predict land everywhere
    if np.ptp(values) == 0:
        return float(values[0])
    return float(threshold_otsu(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="the frozen smoke design; writes outputs/smoke/")
    args = parser.parse_args()
    base = OUTPUTS / "smoke" if args.smoke else OUTPUTS
    chips = load_chip_table()
    prov = provenance()
    build_cache(chips, prov)
    s2 = np.load(INTERIM / "s2.npy", mmap_mode="r")
    label = np.load(INTERIM / "label.npy", mmap_mode="r")
    jrc = np.load(INTERIM / "jrc.npy", mmap_mode="r")

    plan = smoke_plan(chips) if args.smoke else None
    fold_list = plan["folds"] if args.smoke else folds()
    roles = [fold_chips(chips, f, f if args.smoke else None) for f in fold_list]
    needed = sorted({k for r in roles for ks in r.values() for k in ks})

    per_chip = {}
    for k in needed:
        lab = np.asarray(label[k])
        idx = water_indices(np.asarray(s2[k]))
        scored = lab != -1
        per_chip[k] = {"counts": {n: threshold_counts(v, lab) for n, v in idx.items()},
                       "otsu": {n: otsu(v[scored & np.isfinite(v)]) for n, v in idx.items()}}

    tmp = base / "index.tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    chosen, rows = [], []
    for f, r in zip(fold_list, roles):
        with open(tmp / f"manifest_{f['test']}.csv", "w", newline="", encoding="utf-8") as fh:
            fh.write("chip_id,role\n")
            for role, ks in r.items():
                for k in ks:
                    fh.write(f"{chips[k]['chip_id']},{role}\n")
        fitted, val_iou = {}, {}
        for name in ("MNDWI", "NDWI"):
            fit_counts = sum(per_chip[k]["counts"][name] for k in r["fitting"])
            j = best_by_exact_iou([tuple(x) for x in fit_counts])
            fitted[name] = float(THRESHOLDS[j])
            tp, fp, fn = sum(per_chip[k]["counts"][name] for k in r["validation"])[j]
            val_iou[name] = (int(tp), int(tp + fp + fn))
        from fractions import Fraction
        v = {n: Fraction(a, b) if b else Fraction(0) for n, (a, b) in val_iou.items()}
        pick = "MNDWI" if v["MNDWI"] >= v["NDWI"] else "NDWI"
        chosen.append({"test": f["test"], "validation": f["validation"], "t_MNDWI": fitted["MNDWI"],
                       "t_NDWI": fitted["NDWI"], "val_iou_MNDWI": float(v["MNDWI"]),
                       "val_iou_NDWI": float(v["NDWI"]), "chosen": pick})
        for k in r["test"]:
            lab = np.asarray(label[k])
            permanent = np.asarray(jrc[k]) == 1
            idx = water_indices(np.asarray(s2[k]))
            preds = {"index_tuned": idx[pick] > fitted[pick],
                     "index_mndwi_t0": idx["MNDWI"] > 0.0,
                     "index_otsu_mndwi": idx["MNDWI"] > per_chip[k]["otsu"]["MNDWI"],
                     "index_otsu_ndwi": idx["NDWI"] > per_chip[k]["otsu"]["NDWI"]}
            for model, pred in preds.items():
                for scope, keep in (("all_water", None), ("flood_only", ~permanent)):
                    tp, fp, fn = confusion(pred, lab, keep)
                    rows.append({"model": model, "seed": 0, "test_event": f["test"],
                                 "chip_id": chips[k]["chip_id"], "scope": scope, "tp": tp, "fp": fp, "fn": fn})

    keys = [(r["model"], r["chip_id"], r["scope"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate index rows"
    for name, data in (("fold_thresholds.csv", chosen), ("chip_counts.csv", rows)):
        with open(tmp / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_json(tmp / "run_record.json", {
        "complete": True, "smoke": args.smoke, "folds": [f["test"] for f in fold_list],
        "test_chips": sorted(chips[k]["chip_id"] for r in roles for k in r["test"]),
        "rules": list(INDEX_RULES), "provenance": prov,
        "versions": {"numpy": np.__version__, "rasterio": rasterio.__version__, "scikit-image": skimage.__version__}})
    publish(tmp, base / "index")
    print(f"index results: {len(fold_list)} folds, {len(rows)} rows -> {base / 'index'}")


if __name__ == "__main__":
    main()
