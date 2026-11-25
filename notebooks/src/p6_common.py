"""Shared paths, fold rules and metrics for the flood-segmentation study.

Everything here follows analysis_plan.md v1.1. No function in this file looks at a
held-out event when it builds a fold.
"""
import csv
import hashlib
import json
import os
import pathlib
import shutil
from fractions import Fraction

import numpy as np

# On Colab the runner sets PAPER_ROOT to its job folder; locally the project folder is used.
PROJECT = pathlib.Path(os.environ.get("PAPER_ROOT", pathlib.Path(__file__).resolve().parents[2]))
RAW = PROJECT / "data" / "raw" / "sen1floods11_v1.1" / "data" / "flood_events" / "HandLabeled"
RECORDS = PROJECT / "data" / "records"
INTERIM = PROJECT / "data" / "interim"
OUTPUTS = PROJECT / "outputs"

BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
BAND = {name: i for i, name in enumerate(BANDS)}

# Inventory chip counts per event (gaps.md and data_sources.md), used for the fold rule.
INVENTORY_COUNTS = {"Bolivia": 15, "Ghana": 53, "India": 68, "Mekong": 30, "Nigeria": 18, "Pakistan": 28,
                    "Paraguay": 67, "Somalia": 26, "Spain": 30, "Sri-Lanka": 42, "USA": 69}
EVENTS = tuple(sorted(INVENTORY_COUNTS))
STRATA = ("clear", "partly_cloudy", "cloudy")
MODELS = ("index_tuned", "unet_r34_scratch", "unet_r50_ssl4eo")
LEARNED = ("unet_r34_scratch", "unet_r50_ssl4eo")
SEEDS = (0, 1, 2)
SCOPES = ("all_water", "flood_only")
SCRIPTS = ("p6_common.py", "02_cache_index_CPU.py", "03_train_GPU.py", "04_stats_figures_CPU.py")


def stratum_from_counts(cloud_pixels, scored_pixels):
    """Plan §6 with exact integer boundaries: clear < 5%, partly cloudy 5-30% (both ends
    included), cloudy > 30%, all as shares of scored pixels."""
    if 100 * cloud_pixels < 5 * scored_pixels:
        return "clear"
    if 100 * cloud_pixels <= 30 * scored_pixels:
        return "partly_cloudy"
    return "cloudy"


THRESHOLDS = np.round(np.arange(-50, 51) / 100.0, 2)  # -0.50 ... 0.50, step 0.01


def validation_event(test_event):
    """Plan §5: sort the 10 remaining events by inventory count (ties alphabetical), take the 5th."""
    remaining = sorted((e for e in EVENTS if e != test_event), key=lambda e: (INVENTORY_COUNTS[e], e))
    return remaining[4]


def folds():
    """One fold per held-out event: test, validation and the 9 fitting events."""
    out = []
    for test in EVENTS:
        val = validation_event(test)
        fit = tuple(e for e in EVENTS if e not in (test, val))
        out.append({"test": test, "validation": val, "fitting": fit})
    return out


def load_chip_table():
    """Chips kept by the subset rule (at least one valid label pixel), in a fixed order."""
    with open(RECORDS / "chip_inventory.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["scored_pixels"]) > 0]
    for r in rows:
        r["stratum"] = stratum_from_counts(int(r["cloud_pixels_scored"]), int(r["scored_pixels"]))
    rows = sorted(rows, key=lambda r: r["chip_id"])
    assert len(rows) == 441, f"expected 441 chips, found {len(rows)}"
    return rows


def smoke_plan(chips):
    """The one frozen smoke design shared by 02, 03 and 04: the first two folds, the first two
    chips of each role (sorted chip IDs), seeds 0 and 1, 1 epoch of 2 steps."""
    plan = []
    for f in folds()[:2]:
        roles = {}
        for role, events in (("test", (f["test"],)), ("validation", (f["validation"],)), ("fitting", f["fitting"])):
            roles[role] = [c["chip_id"] for c in chips if c["event"] in events][: (2 if role != "fitting" else 16)]
        plan.append({**f, "chips": roles})
    return {"folds": plan, "seeds": [0, 1], "epochs": 1, "steps": 2}


def fold_chips(chips, fold, smoke_fold=None):
    """Row indices of the fitting, validation and test chips of a fold (smoke: the frozen subset)."""
    index = {c["chip_id"]: k for k, c in enumerate(chips)}
    if smoke_fold is not None:
        return {role: [index[c] for c in ids] for role, ids in smoke_fold["chips"].items()}
    return {"fitting": [k for k, c in enumerate(chips) if c["event"] in fold["fitting"]],
            "validation": [k for k, c in enumerate(chips) if c["event"] == fold["validation"]],
            "test": [k for k, c in enumerate(chips) if c["event"] == fold["test"]]}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    """Hashes that tie a result to its data, inventory and code."""
    here = pathlib.Path(__file__).parent
    return {"file_hashes_csv": sha256_file(RECORDS / "file_hashes.csv"),
            "chip_inventory_csv": sha256_file(RECORDS / "chip_inventory.csv"),
            "code": {name: sha256_file(here / name) for name in SCRIPTS}}


def publish(tmp_dir, final_dir):
    """Replace final_dir with a fully written tmp_dir; the old copy is kept until the new one is in place."""
    old = final_dir.with_name(final_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final_dir.exists():
        os.replace(final_dir, old)
    os.replace(tmp_dir, final_dir)
    if old.exists():
        shutil.rmtree(old)


DESIGN_FILE = OUTPUTS / "production_design.json"


def production_design():
    """The frozen production design: epochs and seeds, fixed from GPU timing alone (compute_plan.md
    cost rule) before the full run. Training and statistics both refuse to run without it."""
    d = json.loads(DESIGN_FILE.read_text(encoding="utf-8"))
    assert d["epochs"] in (60, 40) and d["seeds"] in ([0, 1, 2], [0, 1]), d
    return d


def cache_is_valid(ids, prov):
    """True when the cache in data/interim matches the chip list, the raw-file hashes and its array hashes."""
    record = INTERIM / "cache_record.json"
    if not record.exists():
        return False
    r = json.loads(record.read_text(encoding="utf-8"))
    if r.get("chip_ids") != ids or r.get("file_hashes_csv") != prov["file_hashes_csv"]:
        return False
    shapes = {"s2.npy": [len(ids), 13, 512, 512], "label.npy": [len(ids), 512, 512], "jrc.npy": [len(ids), 512, 512]}
    for name, shape in shapes.items():
        if not (INTERIM / name).exists():
            return False
        arr = np.load(INTERIM / name, mmap_mode="r")
        if list(arr.shape) != shape or sha256_file(INTERIM / name) != r["array_sha256"][name]:
            return False
    return True


def data_provenance(prov):
    """The data part of a provenance record (code hashes may differ between stages)."""
    return {k: prov[k] for k in ("file_hashes_csv", "chip_inventory_csv")}


def best_by_exact_iou(counts):
    """Index of the maximum IoU compared as exact fractions; ties go to the threshold nearest 0,
    then to the lower threshold. counts: list of (tp, fp, fn) per grid threshold."""
    values = [Fraction(tp, tp + fp + fn) if tp + fp + fn else Fraction(0) for tp, fp, fn in counts]
    best = max(values)
    candidates = [k for k, v in enumerate(values) if v == best]
    return min(candidates, key=lambda k: (abs(THRESHOLDS[k]), THRESHOLDS[k]))


def confusion(pred, label, keep=None):
    """True positives, false positives and false negatives of the water class on scored pixels.

    pred: boolean water prediction; label: -1/0/1; keep: optional extra mask of pixels to score.
    """
    scored = label != -1
    if keep is not None:
        scored &= keep
    water = label == 1
    tp = int(np.count_nonzero(pred & water & scored))
    fp = int(np.count_nonzero(pred & ~water & scored))
    fn = int(np.count_nonzero(~pred & water & scored))
    return tp, fp, fn


def iou(tp, fp, fn):
    """Water IoU; None when the denominator is 0 (no water predicted or present)."""
    denom = tp + fp + fn
    return tp / denom if denom else None


def prf(tp, fp, fn):
    """Precision, recall and F1 of the water class; None when undefined."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return precision, recall, f1


def normalised_difference(a, b):
    """(a - b) / (a + b) in float64; pixels where a + b == 0 get -inf (never water)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    total = a + b
    out = np.full(a.shape, -np.inf)
    np.divide(a - b, total, out=out, where=total != 0)
    return out


def water_indices(s2):
    """MNDWI (B3, B11) and NDWI (B3, B8) from a (13, H, W) chip."""
    return {"MNDWI": normalised_difference(s2[BAND["B3"]], s2[BAND["B11"]]),
            "NDWI": normalised_difference(s2[BAND["B3"]], s2[BAND["B8"]])}


def write_json(path, obj):
    """JSON with NaN written as null."""
    def clean(x):
        if isinstance(x, float) and x != x:
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False), encoding="utf-8")
