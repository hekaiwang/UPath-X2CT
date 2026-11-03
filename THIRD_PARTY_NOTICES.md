# Third-party notices

Original UPath-X2CT source retains its existing Apache-2.0 license. This does not
relicense datasets, pretrained initialization or dependencies.

- The encoder is initialized using torchvision's ConvNeXt-Tiny ImageNet1K V1
  weights. Reproduction downloads them from the torchvision provider.
- Autoencoder/diffusion networks use MONAI Generative. DiffDRR provides projection
  rendering. These packages retain their upstream licenses and are dependencies,
  not vendored copies.
- LIDC-IDRI (TCIA), LUNA16 annotation geometry, MSD Task06 Lung, and CT-RATE
  retain their original dataset/annotation terms. Only split identifiers,
  preprocessing annotation fields and numerical evaluation evidence are supplied.
  Raw CT, derived CT/DRR volumes and private clinical data are not redistributed.
- The model was trained on LIDC-IDRI; MSD Lung and CT-RATE were external tests.
  See the model card for intended use and limitations.
- Baseline upstream URLs/revisions and known license labels are preserved in
  `results/baseline_sources.json`. Their source and checkpoints are not bundled.
