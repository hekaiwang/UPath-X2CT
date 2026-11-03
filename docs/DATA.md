# Data preparation

The exact case IDs, patient IDs and series IDs are in `splits/{lidc,msd,ctrate}/`.
`splits/counts.json` records patient and volume counts separately. Never randomly
resplit the data. CT-RATE and MSD are external test cohorts only.

## LIDC-IDRI

Obtain the 1,018 CT series identified by `splits/lidc_source_manifest.csv` from
[TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) or its IDC mirror.
The expected directory structure below the `--source` root is:

```text
LIDC-IDRI-XXXX/<study_uid>/CT_<series_uid>/*.dcm
```

`prepare_data.py --dataset lidc` relocates this manifest to your directory,
builds the full canonical cohort, and packs HDF5 cases under `<output>/pack`.
The supplied LUNA-derived nodule geometry supports the historical preprocessing
and audit fields; nodule masks are not inference inputs. Body and lung masks are
generated using the original threshold/morphology code.

```bash
python scripts/prepare_data.py --dataset lidc \
  --source /path/to/lidc_idri --output data/lidc --device cuda
```

Canonical processing first resamples to a thorax-centered 256³ grid at 1.5 mm,
then block-averages to 128³ at 3 mm. HU is clipped to [-1024,1500] and scaled to
[-1,1]. DiffDRR generates 512² calibrated AP and LAT projections. The public
configuration fixes detector geometry and density calibration.

DRRs are normalized as `clip((log1p(max(drr,0))-mean)/std,-6,6)`, using
**LIDC train-only** per-view means and standard deviations. External cohorts
reuse the internal statistics. A prepared HDF5 case stores `ct`, `body_mask`,
`lung_mask`, `lesion_proxy`, `drr_frontal_ap`, `drr_left_lateral`, and
`projection_matrices`; attributes identify case, patient, series, and split.

## MSD Lung

Download and extract the official [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
Task06_Lung archive. The 63 labelled training-source cases (`imagesTr`, `labelsTr`)
are used exclusively as external tests here.

```bash
python scripts/prepare_data.py --dataset msd --source /path/to/Task06_Lung \
  --internal-pack data/lidc/pack --output data/msd
```

## CT-RATE

Obtain access from the [CT-RATE provider](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).
Use corrected `valid_fixed` NIfTI volumes matching `volume_name` in the frozen
425-patient manifest. Expected paths are
`valid_fixed/valid_<id>/valid_<id>_a/valid_<id>_a_1.nii.gz`.

```bash
python scripts/prepare_data.py --dataset ctrate --source /path/to/valid_fixed \
  --internal-pack data/lidc/pack --output data/ctrate
```

Use a new output directory for preprocessing. The wrapper verifies all resulting
split memberships against the published manifests. Original volumes and derived
CT/DRR images must be obtained/generated locally under source dataset terms.
