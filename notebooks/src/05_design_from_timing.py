"""Apply the compute plan's cost rule to the timing runs alone and write outputs/production_design.json.

Reads only outputs/timing/train/<model>/Bolivia/seed0/run_record.json (durations and workload; timing
runs score nothing). Rule (compute_plan.md): estimate the full run as
  sum over the 11 folds, 2 models and 3 seeds of (seconds per fitting chip-epoch x fitting chips x epochs),
scaled from the timed fold. Over 16 h at 60 epochs -> 40 epochs; still over 16 h -> 2 seeds.

Run:  python notebooks/src/05_design_from_timing.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p6_common import DESIGN_FILE, INVENTORY_COUNTS, OUTPUTS, folds, write_json

LIMIT_HOURS = 16.0
MODELS = ("unet_r34_scratch", "unet_r50_ssl4eo")


def main():
    per_chip_epoch = {}
    gpu = set()
    for m in MODELS:
        r = json.loads((OUTPUTS / "timing" / "train" / m / "Bolivia" / "seed0" / "run_record.json").read_text())
        assert r["complete"] and r["timing_only"] and r["identity"]["epochs"] == 60, r
        per_chip_epoch[m] = r["train_seconds_including_validation"] / (r["fitting_chips"] * 60)
        gpu.add(r["gpu"])
    fitting = []
    for f in folds():
        fitting.append(sum(INVENTORY_COUNTS[e] for e in f["fitting"]) - (5 if "Ghana" in f["fitting"] else 0))

    def hours(epochs, seeds):
        return sum(per_chip_epoch[m] * n * epochs * seeds for m in MODELS for n in fitting) / 3600

    design = {"epochs": 60, "seeds": [0, 1, 2]}
    if hours(60, 3) > LIMIT_HOURS:
        design["epochs"] = 40
        if hours(40, 3) > LIMIT_HOURS:
            design["seeds"] = [0, 1]
    design.update({"estimated_hours": round(hours(design["epochs"], len(design["seeds"])), 2),
                   "estimate_60_epochs_3_seeds_hours": round(hours(60, 3), 2),
                   "seconds_per_fitting_chip_epoch": per_chip_epoch, "gpu": sorted(gpu),
                   "rule": "compute_plan.md cost rule, timing records only"})
    write_json(DESIGN_FILE, design)
    print(json.dumps(design, indent=1))


if __name__ == "__main__":
    main()
