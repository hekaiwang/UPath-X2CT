import csv
import json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import pytest
from upathx2ct.inference import read_projections, stable_seed

ROOT=Path(__file__).resolve().parents[1]

def test_patient_splits_and_counts():
    expected=json.loads((ROOT/'splits/counts.json').read_text())
    for cohort,counts in expected.items():
        rows=list(csv.DictReader((ROOT/f'splits/{cohort}/manifest.csv').open()))
        patient_roles=defaultdict(set)
        for row in rows:patient_roles[row['patient_id']].add(row['split'])
        assert all(len(v)==1 for v in patient_roles.values())
        for split,count in counts.items():
            selected=[r for r in rows if r['split']==split]
            ids=(ROOT/f'splits/{cohort}/{split}.txt').read_text().split()
            assert len(ids)==len(set(ids))==count['volumes']
            assert set(ids)=={r['case_id'] for r in selected}
            assert len({r['patient_id'] for r in selected})==count['patients']

def test_published_aggregates_match_per_case_evidence():
    cohorts={'LIDC-IDRI':'lidc','MSD Lung':'msd','CT-RATE':'ctrate'}
    means=list(csv.DictReader((ROOT/'results/main_results.csv').open()))
    assert len(means)==18
    for row in means:
        raw=list(csv.DictReader((ROOT/f"results/{cohorts[row['dataset']]}_per_case.csv").open()))
        raw=[r for r in raw if r['method']==row['method_id']]
        assert len(raw)==int(row['series'])
        for metric in ['psnr_3d','ssim_2d_orthogonal_mean','mae_hu','mae_hu_body','mae_hu_lung']:
            patients=defaultdict(list)
            for r in raw:patients[r['patient_id']].append(float(r[metric]))
            assert len(patients)==int(row['patients'])
            assert np.mean([np.mean(v) for v in patients.values()])==pytest.approx(float(row[metric]),abs=1e-9)

def test_inference_input_has_no_target_ct_dependency(tmp_path):
    matrices=np.stack([np.eye(3,4),np.eye(3,4)]).astype('float32')
    values={'drr_frontal_ap':np.ones((512,512)), 'drr_left_lateral':np.ones((512,512)), 'projection_matrices':matrices}
    for ext in ['h5','npz']:
        path=tmp_path/f'projection.{ext}'
        if ext=='npz':np.savez(path,**values)
        else:
            with h5py.File(path,'w') as f:
                for k,v in values.items():f[k]=v
        ap,lat,p=read_projections(path)
        assert ap.shape==lat.shape==(512,512)
        np.testing.assert_array_equal(p,matrices)

def test_projection_validation_rejects_invalid_geometry(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,drr_frontal_ap=np.ones((512,512)),drr_left_lateral=np.ones((512,512)),projection_matrices=np.zeros((2,3,4)))
    with pytest.raises(ValueError,match='rank'):read_projections(path)

def test_case_sampling_is_order_independent():
    assert stable_seed(20260724,'case_a',0)==stable_seed(20260724,'case_a',0)
    assert len({stable_seed(20260724,c,k) for c in ['case_a','case_b'] for k in range(8)})==16
