"""Reconstruct a centered 128-cubed CT from calibrated AP/LAT projections."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from upathx2ct.inference import Predictor

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True, help='upath_x2ct_final.pt or its containing directory')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--case-id', required=True, help='Used for deterministic per-case sampling')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260724)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(4)
    model = Predictor(a.weights, a.device, a.precision)
    mean, std = model.predict(a.input, a.case_id, a.samples, a.steps, a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    # The network grid is centered LPS, ZYX; NIfTI array and affine are XYZ/RAS.
    affine = np.diag([-3., -3., 3., 1.])
    affine[:3, 3] = [190.5, 190.5, -190.5]
    for name, volume in [('ct_hu', mean), ('std_hu', std)]:
        image = nib.Nifti1Image(volume.transpose(2, 1, 0), affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, a.output / (name + '.nii.gz'))
    (a.output / 'prediction.json').write_text(json.dumps({'case_id': a.case_id, 'samples': a.samples, 'steps': a.steps, 'eta': 0, 'seed': a.seed, 'precision': a.precision, 'coordinates': 'centered LPS grid exported as RAS NIfTI; no source-patient translation', 'std': 'population standard deviation of decoded samples'}, indent=2)+'\n')

if __name__ == '__main__':
    main()
