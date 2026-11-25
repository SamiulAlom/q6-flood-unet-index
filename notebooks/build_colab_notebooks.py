"""Write the Colab notebooks from the reviewed scripts in notebooks/src.

A: 05_colab_index_timing_T4.ipynb: download and verify data, index baselines (all folds),
   GPU smoke test, timing of one production fold (Bolivia held out, the largest fitting set)
   for both learned models.
B: 06_colab_train_T4.ipynb: full training under outputs/production_design.json.

Run:  python notebooks/build_colab_notebooks.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent

SETUP = r'''# RUNTIME: Colab T4 GPU (plan D3); the 1% smoke test ran in notebook A (03_train_GPU.py --smoke)
SMOKE_TEST = False
import os, pathlib, subprocess, sys, shutil, hashlib, csv, json, time
ROOT = pathlib.Path(os.environ.get("PAPER_ROOT", "/content/job"))
os.environ["PAPER_ROOT"] = str(ROOT)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["P6_TRAIN_ROOT"] = str(ROOT / "checkpoints")   # copied home every few minutes by the runner
SRC = ROOT / "notebooks" / "src"
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0",
                "rasterio==1.5.2", "scikit-image==0.26.0"], check=True)

def run(*args):
    """Run one reviewed script; stop the notebook if it fails."""
    print(">>", " ".join(map(str, args)), flush=True)
    proc = subprocess.run([sys.executable, *map(str, args)], env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"failed: {args}")

env = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
import torch
print("GPU:", env, "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU on this runtime"
assert "T4" in env, f"expected a T4 (plan D3), got {env}"
'''

DOWNLOAD = r'''# Download Sen1Floods11 from the public bucket and check every file against the hashes recorded at step 3.
records = ROOT / "data" / "records"
reference = {r["path"]: r["sha256"] for r in csv.DictReader(open(records / "file_hashes.csv", encoding="utf-8"))}
shutil.copy(records / "file_hashes.csv", records / "file_hashes_step3.csv")
run(SRC / "00_download.py")
fresh = {r["path"]: r["sha256"] for r in csv.DictReader(open(records / "file_hashes.csv", encoding="utf-8"))}
assert fresh == reference, f"downloaded files differ from step 3: {len(set(fresh.items()) ^ set(reference.items()))} entries"
shutil.copy(records / "file_hashes_step3.csv", records / "file_hashes.csv")  # keep the step-3 record byte for byte
print("all", len(fresh), "files match the step-3 hashes")
'''

FINISH = r'''# Copy results from checkpoints/ into outputs/ and write the manifest the runner requires.
def collect(src, dst):
    if (ROOT / "checkpoints" / src).exists():
        shutil.copytree(ROOT / "checkpoints" / src, ROOT / "outputs" / dst, dirs_exist_ok=True)
for src, dst in COLLECT:
    collect(src, dst)
(ROOT / "outputs" / f"colab_env_{NB_TAG}.txt").write_text(
    f"GPU: {env}\ntorch {torch.__version__}\npython {sys.version}\n"
    + subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
rows = ["| file | notebook | cell | description |", "|---|---|---|---|"]
for p in sorted((ROOT / "outputs").rglob("*")):
    if p.is_file() and p.name != "MANIFEST.md":
        rel = p.relative_to(ROOT / "outputs").as_posix()
        rows.append(f"| {rel} | {NB_NAME} | last | {DESCRIBE(rel)} |")
(ROOT / "outputs" / "MANIFEST.md").write_text("\n".join(rows) + "\n")
print(len(rows) - 2, "output files listed")
'''


def notebook(cells):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


A = notebook([
    SETUP,
    DOWNLOAD,
    '# Index baselines on all 11 folds (analysis_plan.md §4.1, §8.3).\nrun(SRC / "02_cache_index_CPU.py")\n',
    '# GPU smoke test: the frozen smoke design on CUDA (deterministic settings exercised).\nrun(SRC / "03_train_GPU.py", "--smoke", "--device", "cuda")\n',
    '# Timing: one production fold (Bolivia held out = largest fitting set), 60 epochs, seed 0, both models.\n'
    '# Records durations only (no held-out scoring); the design is computed from these records by\n'
    '# notebooks/src/05_design_from_timing.py before any result is opened.\n'
    'run(SRC / "03_train_GPU.py", "--timing", "--folds", "Bolivia", "--device", "cuda")\n',
    'NB_TAG, NB_NAME = "A", "05_colab_index_timing_T4.ipynb"\n'
    'COLLECT = [("smoke", "smoke_gpu"), ("timing", "timing")]\n'
    'DESCRIBE = lambda rel: ("index baselines (02)" if rel.startswith("index/") else "GPU smoke test, not a result"\n'
    '                        if rel.startswith("smoke_gpu/") else "timing run, not a result" if rel.startswith("timing/")\n'
    '                        else "run environment")\n' + FINISH,
])

B = notebook([
    SETUP,
    DOWNLOAD,
    '# Cache only (index results already published by notebook A are not rebuilt here).\n'
    'run(SRC / "02_cache_index_CPU.py", "--smoke")\nshutil.rmtree(ROOT / "outputs" / "smoke", ignore_errors=True)\n',
    '# Full training under the frozen production design (outputs/production_design.json, from timing only).\n'
    'shutil.copy(ROOT / "prior_outputs" / "production_design.json", ROOT / "outputs" / "production_design.json")\n'
    'run(SRC / "03_train_GPU.py", "--device", "cuda")\n',
    'NB_TAG, NB_NAME = "B", "06_colab_train_T4.ipynb"\n'
    'COLLECT = [("train", "train")]\n'
    'DESCRIBE = lambda rel: ("learned-model runs (03)" if rel.startswith("train/") else "production design (copy)"\n'
    '                        if rel == "production_design.json" else "run environment")\n' + FINISH,
])

for name, nb in (("05_colab_index_timing_T4.ipynb", A), ("06_colab_train_T4.ipynb", B)):
    (HERE / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", name)
