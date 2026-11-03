"""Export a trained model to the same layout consumed by predict.py."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from upathx2ct.b0.ema import resolve_checkpoint_model_state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['autoencoder','coarse','diffusion','projection-stats','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    for name,key in [('autoencoder','autoencoder_config'),('coarse','b0_config'),('diffusion','diffusion_config')]:
        c=torch.load(getattr(a,name),map_location='cpu',weights_only=False)
        torch.save({'model':resolve_checkpoint_model_state(c),key:c[key],'step':int(c.get('step',0))},a.output/(name+'.pt'))
    stats=json.loads(a.projection_stats.read_text())
    if stats['definition_split']!='train':
        raise ValueError('Expected training-only normalization')
    shutil.copy2(a.projection_stats,a.output/'projection_stats.json')

if __name__=='__main__':
    main()
