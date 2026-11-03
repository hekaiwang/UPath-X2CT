# Training and reproducibility

## Main training sequence

`scripts/train.py` runs only the main UPath-X2CT model. Inputs are the fixed
LIDC pack. MSD and CT-RATE never participate in optimization or selection.

| Stage | Budget / selection | Output |
|---|---|---|
| Autoencoder | 60,000 updates; validation ceiling checkpoint | 8×32³ latent, 128³ decoder |
| Latent cache | Training-only channel mean/std | Frozen normalized autoencoder |
| Coarse | Up to 80,000 updates, minimum 20,000, early-stop patience 12 validations | Validation-selected `best.pt` |
| Condition cache | Frozen coarse outputs; train-only residual std | Coarse latents and 96×32³ conditions |
| Diffusion | 50,000 updates; minimum validation loss | Selected `best.pt`, exported as deployable weights |

Historical coarse selection was step 20,000 and diffusion selection was step
22,000. Repeating training does not force selection of those steps. The historical
diffusion LR schedule was configured for 200,000 updates and stopped at 50,000;
the public wrapper preserves that schedule horizon separately from its run budget.

Paper settings: seed 20260724, two GPUs, batch 2 per GPU (effective batch 4),
BF16, AdamW, weight decay 1e-4, EMA. Autoencoder LR is 2e-4 and warm-up is 2,000.
Coarse/diffusion LR is 1e-4 and warm-up is 3,000. The coarse ConvNeXt encoder is
initialized from torchvision ConvNeXt-Tiny ImageNet1K V1, with LR multiplier 0.1.
Training downloads that initializer from its original provider on first use.

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
  --dataset-root data/lidc/pack --output runs/main --gpus 2 --batch-size 2
```

Each stage is also a standalone `sota_14`–`sota_20` script. Use `--help` for
arguments. The trainer saves optimizer/scheduler/EMA state in `latest.pt`.
Resume an interrupted stage with its standalone script's `--resume latest.pt`
and the original run arguments; then continue the wrapper using
`--start-stage latents|coarse|conditions|diffusion|export`. A checkpoint resume
restores optimization state; exact data-loader position and RNG replay are
not guaranteed. Cache scripts have their own `--resume` switch.

The public Hugging Face bundle is for inference or weight initialization and
does not contain optimizer state for continuing the historical run. A fresh
training run produces its own resumable checkpoints and finishes by exporting
`runs/main/weights`, directly usable by `predict.py`.

## Evaluation

Use `scripts/evaluate.py`. PSNR uses the full 2524-HU range; SSIM is the average
of slice-wise 2D SSIM in three orthogonal planes. MAE is reported over the full
volume, body and lung. Repeated series are averaged within patient before cohort
means. Raw CSV SSIM is a fraction, while the README shows percent.

## Release validation

The included tests check exact split counts, patient disjointness, independent
recomputation of all 90 main-table metric means, and inference without target CT.
Release checks include strict checkpoint loading and tensor equality after EMA
packaging. GPU smoke checks and replay measurements are recorded under
`reproducibility/`. Full training and all-cohort inference are expensive and are
not claimed to have been rerun for this packaging release.

The release GPU check also runs one optimizer/validation/save step for each of
the three trainers, generates and finalizes both caches, exports the resulting
checkpoints, and runs the public predictor on that exported bundle. These are
small-sample functional checks, not evidence of convergence from retraining.

Raw-source preprocessing was replayed for one case from each of LIDC-IDRI, MSD
Lung and CT-RATE. Prepared CT, both DRRs and camera matrices matched the historical
arrays exactly in all three checks; see `../reproducibility/preprocessing_validation.json`.
