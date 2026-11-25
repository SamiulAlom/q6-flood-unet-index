# Nested leave-one-event-out comparison of U-Nets and a tuned water index on Sen1Floods11 (Sentinel-2)

Analysis code for the paper "U-Net versus a tuned water index on unseen flood events: a nested Sentinel-2 re-estimate with cloud-cover strata".

**Data are not included.** Sen1Floods11 v1.1 is downloaded from its public Google Cloud Storage bucket (`gs://sen1floods11`) by `notebooks/src/00_download.py`, which checks every file against the bucket MD5 and against `data/records/file_hashes.csv` (SHA-256 of the files used). Cite Bonafilia et al., CVPRW 2020, doi:10.1109/CVPRW50498.2020.00113.

## Pipeline
| Step | Script | Runtime |
|---|---|---|
| Download and hash | `notebooks/src/00_download.py` | CPU |
| Chip inventory, s2cloudless cloud fraction | `notebooks/src/01_inventory.py` | CPU |
| Cache and tuned MNDWI/NDWI baselines (11 folds) | `notebooks/src/02_cache_index_CPU.py` | CPU |
| U-Net training (ResNet-34 scratch; ResNet-50 SSL4EO-S12 MoCo), 60 epochs, 3 seeds | `notebooks/src/03_train_GPU.py` | GPU (Colab T4 used) |
| Training budget from timing runs only | `notebooks/src/05_design_from_timing.py` | CPU |
| Statistics (paired event bootstrap, 10,000 resamples) and figures | `notebooks/src/04_stats_figures_CPU.py` | CPU |
| Two descriptive figures | `notebooks/src/07_extra_figures_CPU.py` | CPU |

`notebooks/build_colab_notebooks.py` writes the two Colab notebooks used for the GPU steps. Every script accepts `--smoke` for a small end-to-end test.

## Environment
Python 3.13; torch 2.6 (CUDA 12.4), segmentation-models-pytorch 0.5.0, rasterio 1.5.2, scikit-image 0.26.0, s2cloudless 1.7.3, numpy, matplotlib. SSL4EO-S12 weights: `torchgeo/resnet50_sentinel2_all_moco` (CC-BY-4.0), SHA-256 `df8b932e2a23a0773febedf3f650aa7d342b805f7876ca5ed6b139d7245d7c09`.

## Licence
Code: MIT (see LICENSE). The Sen1Floods11 data and the SSL4EO-S12 weights keep their own licences.

## Outputs
`outputs/` holds the run records, per-chip and per-event counts, statistics and figures from the runs reported in the paper (prediction arrays and smoke-test files are not included).
