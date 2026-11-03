"""Evaluate the frozen model, aggregating repeated series within patient."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from upathx2ct.inference import Predictor
from upathx2ct.sota.evaluation import reconstruction_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--max-cases',type=int,default=0)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    ids=(a.dataset_root/(a.split+'.txt')).read_text().split()
    if a.max_cases: ids=ids[:a.max_cases]
    if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty or duplicate split')
    torch.set_num_threads(4)
    model=Predictor(a.weights,a.device,a.precision)
    rows=[]
    for case in ids:
        path=a.dataset_root/'cases'/(case+'.h5')
        hu,_=model.predict(path,case)
        with h5py.File(path,'r') as f:
            if str(f.attrs['split'])!=a.split: raise ValueError('Split mismatch')
            metrics=reconstruction_metrics((hu-238.)/1262.,np.asarray(f['ct'],dtype=np.float32),np.asarray(f['body_mask']),np.asarray(f['lung_mask']))
            rows.append(dict(case_id=case,patient_id=str(f.attrs['patient_id']),**metrics))
        print(case,metrics['psnr_3d'],flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'per_case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    patients=sorted({r['patient_id'] for r in rows})
    means={k:float(np.mean([np.mean([r[k] for r in rows if r['patient_id']==pid]) for pid in patients])) for k in rows[0] if k not in ['case_id','patient_id']}
    (a.output/'summary.json').write_text(json.dumps({'patients':len(patients),'volumes':len(rows),'aggregation':'patient means','samples':8,'steps':50,'means':means},indent=2)+'\n')

if __name__=='__main__':
    main()
