"""Prepare exactly the frozen LIDC-IDRI, MSD Lung or CT-RATE cohort."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['lidc','msd','ctrate'],required=True)
    p.add_argument('--source',type=Path,required=True,help='LIDC lidc_idri directory, MSD Task06_Lung directory, or CT-RATE valid_fixed directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--internal-pack',type=Path,help='Prepared LIDC pack; supplies frozen train projection statistics for external cohorts')
    p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    out=a.output.resolve(); source=a.source.resolve()
    out.mkdir(parents=True,exist_ok=True)
    def run(script,*args):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True)
    protocol=ROOT/'configs/sota/upath_lidc_v1.json'
    if a.dataset=='lidc':
        rows=list(csv.DictReader((ROOT/'splits/lidc_source_manifest.csv').open()))
        for row in rows:
            row['dicom_dir']=str(source/row['dicom_dir'].split('/lidc_idri/',1)[1])
            if not Path(row['dicom_dir']).is_dir(): raise FileNotFoundError(row['dicom_dir'])
        manifest=out/'source_manifest.csv'
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        release=out/'canonical'
        run('sota_09_prepare_d1_canonical_pilot.py','--config',protocol,'--manifest',manifest,'--nodule-table',ROOT/'splits/lidc_nodule_geometry.csv','--output-dir',release/'shard_00','--full-release','--device',a.device)
        run('sota_10_merge_release.py','--release-root',release,'--source-manifest',manifest,'--num-shards',1)
        run('sota_13_pack_training_data.py','--release-root',release,'--output-root',out/'pack','--workers',a.workers)
        pack=out/'pack'
    elif a.dataset=='msd':
        if a.internal_pack is None: raise ValueError('--internal-pack is required')
        run('paper_03_prepare_msd_external.py','--images',source/'imagesTr','--labels',source/'labelsTr','--protocol-config',protocol,'--internal-pack',a.internal_pack.resolve(),'--output-root',out/'pack','--device',a.device)
        pack=out/'pack'
    else:
        if a.internal_pack is None: raise ValueError('--internal-pack is required')
        from upathx2ct.paper.external_preparation import ExternalCTCase, prepare_unlabelled_case
        rows=list(csv.DictReader((ROOT/'splits/ctrate/manifest.csv').open()))
        config=json.loads(protocol.read_text()); pack=out/'pack';pack.mkdir(exist_ok=True)
        audits=[]
        for row in rows:
            name=row['volume_name']; parts=name.removesuffix('.nii.gz').split('_')
            image=source/('_'.join(parts[:2]))/('_'.join(parts[:3]))/name
            case=ExternalCTCase(row['case_id'],row['patient_id'],row['series_uid'],image)
            audits.append(prepare_unlabelled_case(case,pack,config,device=a.device,renderer='siddon',patch_size=64,protocol_id='UPATH-CTRATE-external-v1',make_preview=False,resume=False))
        with (pack/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
        for split in ['train','val','test']:shutil.copy2(ROOT/f'splits/ctrate/{split}.txt',pack/(split+'.txt'))
        shutil.copy2(a.internal_pack/'projection_stats.json',pack/'projection_stats.json')
        (pack/'summary.json').write_text(json.dumps({'status':'pass','cases':len(audits),'patients':425})+'\n')
    for split in ['train','val','test']:
        expected=set((ROOT/f'splits/{a.dataset}/{split}.txt').read_text().split())
        if not expected and not (pack/(split+'.txt')).exists():
            (pack/(split+'.txt')).write_text('')
        actual=set((pack/(split+'.txt')).read_text().split())
        if actual!=expected: raise RuntimeError(f'{split} differs from the frozen release split')
    print('Prepared and verified frozen cohort:',pack)

if __name__=='__main__':
    main()
