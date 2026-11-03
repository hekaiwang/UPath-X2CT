# UPath-X2CT

> **Pretrained weights: [Hugging Face · wanghekai/UPath-X2CT](https://huggingface.co/wanghekai/UPath-X2CT).**
> The bundle includes the frozen CT autoencoder, geometry-aware coarse model,
> validation-selected residual diffusion model, and training normalization parameters.

Official implementation of **Geometry-Aware Biplanar Chest CT Reconstruction
with Residual Latent Diffusion** by Hekai Wang, Zhenrong He, Xingyang Wu,
Caijin Zhou, Yihang Zhou, Dong Liang, Zhanqi Hu, and Haifeng Wang.

Calibrated AP/LAT projections → shared ConvNeXt-FPN → geometric feature lifting
and adaptive fusion → coarse 3D latent → residual latent diffusion → decoded CT.
The released model uses **50 DDIM steps, η = 0, and 8 decoded samples**. The
diffusion checkpoint was selected on validation at **22,000 updates**.

## Datasets and fixed splits

| Dataset / source | Patients | CT volumes | Train / val / test patients |
|---|---:|---:|---|
| [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | 1,010 | 1,018 | 817 / 91 / 102 |
| [MSD Lung / Task06](http://medicaldecathlon.com/) | 63 | 63 | 0 / 0 / 63 |
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | 425 | 425 | 0 / 0 / 425 |

LIDC train / validation / test **volume** counts are 824 / 91 / 103. Repeated
series from a patient stay in one split. The external sets are the exact study
subsets, not the full source datasets. Download CT images from the providers;
this repository contains identifiers and split files, not patient images.
See [data preparation](docs/DATA.md) and [fixed split manifests](splits/).

## Main results

Patient-level means on CT-derived projections. PSNR is in dB, SSIM is displayed
as percent, and MAE is in HU. These are the historical main-method and five
baseline results; full precision and per-case evidence are in [results/](results/).

| Dataset | Method | PSNR ↑ | SSIM (%) ↑ | MAE whole ↓ | MAE body ↓ | MAE lung ↓ |
|---|---|---:|---:|---:|---:|---:|
| LIDC-IDRI | X2CT-GAN | 24.05 | 72.67 | 69.49 | 131.67 | 163.80 |
| LIDC-IDRI | PerX2CT | 18.73 | 58.15 | 140.86 | 239.05 | 234.84 |
| LIDC-IDRI | BX2S HU adapter | 21.75 | 66.18 | 94.47 | 158.35 | 187.65 |
| LIDC-IDRI | XraySyn HU adapter (AP only) | 19.39 | 59.20 | 138.96 | 228.13 | 215.20 |
| LIDC-IDRI | DVG-Diffusion v1.4 | 24.55 | 74.18 | 65.33 | 127.36 | 146.47 |
| LIDC-IDRI | **UPath-X2CT** | 26.35 | 81.45 | 50.00 | 96.14 | 116.31 |
| MSD Lung | X2CT-GAN | 18.33 | 53.47 | 166.62 | 334.62 | 174.38 |
| MSD Lung | PerX2CT | 18.04 | 52.53 | 160.55 | 265.37 | 230.13 |
| MSD Lung | BX2S HU adapter | 21.54 | 64.10 | 101.87 | 164.20 | 162.09 |
| MSD Lung | XraySyn HU adapter (AP only) | 18.47 | 51.51 | 171.58 | 259.78 | 228.24 |
| MSD Lung | DVG-Diffusion v1.4 | 23.47 | 68.91 | 79.19 | 133.64 | 150.13 |
| MSD Lung | **UPath-X2CT** | 24.98 | 76.43 | 63.14 | 105.24 | 125.06 |
| CT-RATE | X2CT-GAN | 17.48 | 47.19 | 196.79 | 426.36 | 195.12 |
| CT-RATE | PerX2CT | 16.95 | 44.40 | 195.67 | 316.42 | 275.47 |
| CT-RATE | BX2S HU adapter | 21.15 | 59.65 | 109.96 | 178.01 | 194.58 |
| CT-RATE | XraySyn HU adapter (AP only) | 18.57 | 48.27 | 163.49 | 244.44 | 243.89 |
| CT-RATE | DVG-Diffusion v1.4 | 22.91 | 64.47 | 85.64 | 145.60 | 171.05 |
| CT-RATE | **UPath-X2CT** | 24.14 | 71.84 | 70.99 | 117.82 | 139.61 |

BX2S and XraySyn use project HU adapters; XraySyn uses AP only. DVG-Diffusion is
the project v1.4 decoder-unfrozen retraining. See [baseline provenance](docs/BASELINES.md).

## Install

Use a fresh Python 3.10 environment. Run commands from the repository root.
The reference runtime is PyTorch 2.5.1 with CUDA 12.4.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[download,dev]'
python -m pytest -q
```

## Download and predict

```bash
python scripts/download_weights.py --output weights
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py \
  --weights weights/upath_x2ct_final.pt --input /path/to/projections.npz \
  --case-id example --output runs/example
```

The download is pinned to an immutable Hugging Face revision and verifies file
SHA256. The only deployed checkpoint is `upath_x2ct_final.pt`, containing all
three internal model components and normalization parameters. No separate coarse
checkpoint or standalone coarse evaluation results are included in this release.
Bundling does not fix historical numerical replay differences; see
[reproducibility](docs/REPRODUCIBILITY.md#numerical-replay-limitations).

Input NPZ/HDF5 needs only `drr_frontal_ap` and `drr_left_lateral`
(512×512 raw DRRs), and `projection_matrices` (2×3×4, centered LPS geometry).
No target CT or target mask is read during prediction. Output files are
`ct_hu.nii.gz` and `std_hu.nii.gz`, on a centered 128³ grid with 3-mm spacing.
See [input geometry and inference](docs/INFERENCE.md). CPU inference accepts
`--device cpu --precision fp32`, but is substantially slower.

## Prepare and train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_data.py \
  --dataset lidc --source /path/to/lidc_idri --output data/lidc
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
  --dataset-root data/lidc/pack --output runs/training --gpus 2
python scripts/predict.py --weights runs/training/weights \
  --input /path/to/projections.npz --case-id example --output runs/trained_example
```

The training pipeline runs autoencoder training, latent normalization/caching,
coarse training, conditioning caching, residual diffusion training, and exports
the selected weights for the same inference interface. See [training and
reproduction](docs/REPRODUCIBILITY.md) for budgets, checkpoint resume, and validation.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --weights weights \
  --dataset-root data/lidc/pack --split test --output runs/lidc_test
```

Evaluation saves per-case metrics and patient-averaged means. Use the same
command for prepared MSD and CT-RATE packs. Normalization always comes from
LIDC training data.

## Scope

This is a controlled **CT-derived DRR-to-CT** study. Performance on real clinical
radiographs or diagnostic replacement is not established. This focused release
contains the main model, its three-dataset workflow, weights, and five baseline
result sets. Internal component/sampling experiments and private clinical data
are outside the release.

Source code retains the project's Apache-2.0 license. Data, pretrained
initialization and dependencies retain their own terms; see [third-party
notices](THIRD_PARTY_NOTICES.md).
