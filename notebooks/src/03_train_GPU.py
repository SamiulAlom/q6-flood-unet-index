"""Train and evaluate the two learned models for every fold and seed (analysis_plan.md §4-5).

Models
  unet_r34_scratch : U-Net, ResNet-34 encoder, random initialisation; inputs standardised
                     with band means and SDs of the fold's 9 fitting events.
  unet_r50_ssl4eo  : U-Net, ResNet-50 encoder initialised from SSL4EO-S12 MoCo weights
                     (13-band Sentinel-2 L1C); inputs = digital number / 10000, the scaling
                     those weights were trained with.

Training: BCE + Dice on scored pixels only, AdamW (lr 1e-3, weight decay 1e-4), cosine decay,
60 epochs, batch 8, random 256x256 crops with flips and 90-degree rotations.
Fixed training budget (60 epochs, or --epochs 40 under the compute plan's cost rule); the
checkpoint kept is the epoch with the best pooled water IoU on the validation event
(strictly better, so ties keep the earlier epoch). The held-out event is scored once, with
that checkpoint, on full 512x512 chips. Per-chip confusion counts are saved with and without
JRC permanent water; seed-0 predictions are saved as packed bits for the qualitative figure.

Each (model, fold, seed) run is written to a temporary folder and published in one step
with run_record.json, which binds it to the data and inventory hashes, the code hashes,
the fold roles, the weights and the run settings. A finished run is reused only when its
record matches the current ones exactly; otherwise it is rerun.

Run:  python notebooks/src/03_train_GPU.py
      [--smoke] [--models a,b] [--folds USA,Spain] [--seeds 0,1,2] [--device cuda|cpu]
"""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / ".pylib"))  # segmentation_models_pytorch, kept out of the shared venv

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from p6_common import (INTERIM, OUTPUTS, cache_is_valid, confusion, fold_chips, folds, load_chip_table,
                       production_design, provenance, publish, smoke_plan, write_json)

SSL4EO_URL = ("https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
              "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth")
SSL4EO_FILE = HERE.parents[1] / "data" / "weights" / "resnet50_sentinel2_all_moco-df8b932e.pth"
MODELS = ("unet_r34_scratch", "unet_r50_ssl4eo")
SSL4EO_SHA256 = "df8b932e2a23a0773febedf3f650aa7d342b805f7876ca5ed6b139d7245d7c09"
BATCH, CROP, LR, WD = 8, 256, 1e-3, 1e-4


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def ssl4eo_state_dict():
    """Download the pinned SSL4EO-S12 weights once; return encoder weights for smp's ResNet-50."""
    if not SSL4EO_FILE.exists():
        SSL4EO_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(SSL4EO_URL, str(SSL4EO_FILE), progress=False)
    digest = hashlib.sha256(SSL4EO_FILE.read_bytes()).hexdigest()
    assert digest == SSL4EO_SHA256, f"unexpected SSL4EO weights hash {digest}"
    state = torch.load(SSL4EO_FILE, map_location="cpu", weights_only=True)
    return {k: v for k, v in state.items() if not k.startswith("fc.")}, digest


def build_model(name):
    if name == "unet_r34_scratch":
        return smp.Unet("resnet34", encoder_weights=None, in_channels=13, classes=1), {}
    state, digest = ssl4eo_state_dict()
    model = smp.Unet("resnet50", encoder_weights=None, in_channels=13, classes=1)
    target = model.encoder.state_dict()
    assert set(state) == set(target), (set(state) ^ set(target))
    assert all(state[k].shape == target[k].shape for k in state)
    model.encoder.load_state_dict(state, strict=True)  # smp's override returns nothing
    loaded = model.encoder.state_dict()
    assert all(torch.equal(loaded[k].cpu(), state[k]) for k in state), "encoder weights not loaded"
    return model, {"ssl4eo_sha256": digest, "ssl4eo_tensors_loaded": len(state)}


class Scaler:
    """Input scaling: fold band statistics (scratch model) or /10000 (SSL4EO model)."""

    def __init__(self, model_name, s2, fit_rows):
        if model_name == "unet_r50_ssl4eo":
            self.mean, self.std = np.zeros(13, np.float32), np.full(13, 10000.0, np.float32)
        else:
            total, sq, n = np.zeros(13), np.zeros(13), 0
            for i in fit_rows:
                x = np.asarray(s2[i], dtype=np.float64).reshape(13, -1)
                total += x.sum(1)
                sq += (x ** 2).sum(1)
                n += x.shape[1]
            self.mean = (total / n).astype(np.float32)
            self.std = np.sqrt(sq / n - (total / n) ** 2).astype(np.float32)
        assert np.all(np.isfinite(self.std)) and np.all(self.std > 0), self.std

    def __call__(self, x):
        return (x.astype(np.float32) - self.mean[:, None, None]) / self.std[:, None, None]


def random_batch(rng, rows, s2, label, scaler):
    """Random 256x256 crops with flips and 90-degree rotations."""
    xs, ys = [], []
    for i in rng.choice(rows, size=BATCH, replace=True):
        r, c = rng.integers(0, 512 - CROP + 1, size=2)
        x = scaler(np.asarray(s2[i, :, r:r + CROP, c:c + CROP]))
        y = np.asarray(label[i, r:r + CROP, c:c + CROP])
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.int64))


def loss_fn(logits, y):
    """BCE + soft Dice over scored pixels (label != -1)."""
    logits = logits[:, 0]
    scored = (y != -1).float()
    target = (y == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * scored).sum() / scored.sum().clamp(min=1)
    p = torch.sigmoid(logits) * scored
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + (target * scored).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict(model, rows, s2, scaler, device):
    model.eval()
    for i in rows:
        x = torch.from_numpy(scaler(np.asarray(s2[i]))[None]).to(device)
        logits = model(x)[0, 0]
        assert torch.isfinite(logits).all(), f"non-finite logits on chip row {i}"
        yield i, (logits > 0).cpu().numpy()


def run_identity(model_name, fold, seed, roles, chips, epochs, steps, device, prov):
    """Everything a finished run must match to be reused."""
    return {"model": model_name, "seed": seed, "test": fold["test"], "validation": fold["validation"],
            "fitting": list(fold["fitting"]),
            "chips": {r: [chips[k]["chip_id"] for k in ks] for r, ks in roles.items()},
            "epochs": epochs, "steps_per_epoch": steps, "batch": BATCH, "crop": CROP, "lr": LR, "wd": WD,
            "device_type": device.type, "provenance": prov,
            "ssl4eo_sha256": SSL4EO_SHA256 if model_name == "unet_r50_ssl4eo" else None,
            "versions": {"torch": torch.__version__, "smp": smp.__version__, "numpy": np.__version__}}


def finished(out_dir, identity):
    """True only for a complete run whose record matches this identity exactly."""
    record = out_dir / "run_record.json"
    if not record.exists():
        return False
    r = json.loads(record.read_text(encoding="utf-8"))
    if r.get("identity") != identity or not r.get("complete"):
        return False
    with open(out_dir / "chip_counts.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if r.get("timing_only"):
        return True
    expected = {(c, s) for c in identity["chips"]["test"] for s in ("all_water", "flood_only")}
    return {(x["chip_id"], x["scope"]) for x in rows} == expected and len(rows) == len(expected)


def run(model_name, fold, seed, roles, chips, s2, label, jrc, device, epochs, steps, identity, out_dir,
        timing=False):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    fit, val, test = roles["fitting"], roles["validation"], roles["test"]
    model, extra = build_model(model_name)
    model.to(device)
    scaler = Scaler(model_name, s2, fit)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * steps)
    best, best_epoch, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(steps):
            x, y = random_batch(rng, fit, s2, label, scaler)
            loss = loss_fn(model(x.to(device)), y.to(device))
            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            losses.append(float(loss))
        tp = fp = fn = 0
        for i, pred in predict(model, val, s2, scaler, device):
            a, b, c = confusion(pred, np.asarray(label[i]))
            tp, fp, fn = tp + a, fp + b, fn + c
        val_iou = tp / max(tp + fp + fn, 1)
        history.append({"epoch": epoch, "seconds": round(time.time() - t0, 1)} if timing else
                       {"epoch": epoch, "loss": float(np.mean(losses)), "val_iou": val_iou,
                        "seconds": round(time.time() - t0, 1)})
        if val_iou > best:
            best, best_epoch = val_iou, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0
    if timing:
        # Timing runs record durations and workload only: no held-out scoring, no predictions,
        # no validation scores (compute_plan.md cost rule uses timing alone).
        tmp = out_dir.with_name(out_dir.name + ".tmp")
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        write_json(tmp / "history.json", history)
        write_json(tmp / "run_record.json", {
            "complete": True, "timing_only": True, "identity": identity,
            "train_seconds_including_validation": round(train_seconds, 1),
            "fitting_chips": len(fit), "validation_chips": len(val), "test_chips": len(test),
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
        publish(tmp, out_dir)
        return
    model.load_state_dict(best_state)
    rows, packed = [], {}
    t1 = time.time()
    for i, pred in predict(model, test, s2, scaler, device):
        if seed == 0:
            packed[chips[i]["chip_id"]] = np.packbits(pred)
        lab = np.asarray(label[i])
        permanent = np.asarray(jrc[i]) == 1
        for scope, keep in (("all_water", None), ("flood_only", ~permanent)):
            tp, fp, fn = confusion(pred, lab, keep)
            rows.append({"model": model_name, "seed": seed, "test_event": fold["test"],
                         "chip_id": chips[i]["chip_id"], "scope": scope, "tp": tp, "fp": fp, "fn": fn})
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / "chip_counts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if packed:
        np.savez_compressed(tmp / "predictions_seed0.npz", **packed)
    write_json(tmp / "history.json", history)
    write_json(tmp / "run_record.json", {
        "complete": True, "identity": identity, "best_epoch": best_epoch, "best_val_iou": best,
        "train_seconds": round(train_seconds, 1), "predict_seconds": round(time.time() - t1, 1),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, **extra})
    publish(tmp, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--folds", default="")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing", action="store_true",
                        help="GPU timing run: 60 epochs, seed 0, writes outputs/timing/ (never used as a result)")
    args = parser.parse_args()
    chips = load_chip_table()
    prov = provenance()
    assert cache_is_valid([c["chip_id"] for c in chips], prov), "cache does not match data or chip table; rerun 02"
    s2 = np.load(INTERIM / "s2.npy", mmap_mode="r")
    label = np.load(INTERIM / "label.npy", mmap_mode="r")
    jrc = np.load(INTERIM / "jrc.npy", mmap_mode="r")
    plan = smoke_plan(chips) if args.smoke else None
    fold_list = plan["folds"] if args.smoke else folds()
    if args.smoke:
        seeds, n_epochs = plan["seeds"], plan["epochs"]
    elif args.timing:
        seeds, n_epochs = [0], 60
    else:
        design = production_design()
        seeds, n_epochs = design["seeds"], design["epochs"]
    if args.folds:
        wanted = args.folds.split(",")
        fold_list = [f for f in fold_list if f["test"] in wanted]
    # On Colab P6_TRAIN_ROOT points into checkpoints/, which the runner copies home every few minutes.
    base = pathlib.Path(os.environ["P6_TRAIN_ROOT"]) if os.environ.get("P6_TRAIN_ROOT") else OUTPUTS
    root = base / ("smoke" if args.smoke else "timing" if args.timing else "") / "train"
    device = torch.device(args.device)
    if device.type == "cuda":
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG not set"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    for seed in seeds:
        for fold in fold_list:
            roles = fold_chips(chips, fold, fold if args.smoke else None)
            epochs = n_epochs
            steps = plan["steps"] if args.smoke else max(1, len(roles["fitting"]) // BATCH)
            for model_name in args.models.split(","):
                out_dir = root / model_name / fold["test"] / f"seed{seed}"
                identity = run_identity(model_name, fold, seed, roles, chips, epochs, steps, device, prov)
                if finished(out_dir, identity):
                    continue
                start = time.time()
                run(model_name, fold, seed, roles, chips, s2, label, jrc, device, epochs, steps, identity, out_dir,
                    timing=args.timing)
                print(f"{model_name} {fold['test']} seed{seed} done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
