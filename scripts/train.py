"""Train the paper autoencoder, coarse model and residual diffusion in order."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=2, help='Per GPU; paper effective batch is 4')
    p.add_argument('--start-stage', choices=['autoencoder','latents','coarse','conditions','diffusion','export'], default='autoencoder')
    a = p.parse_args()
    if a.gpus < 1 or a.batch_size < 1:
        raise ValueError('GPU count and per-GPU batch size must be positive')
    root = Path(__file__).resolve().parents[1]
    out = a.output.resolve(); data = a.dataset_root.resolve()
    if a.start_stage == 'autoencoder' and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory or choose --start-stage')
    out.mkdir(parents=True, exist_ok=True)
    ae, lat, coarse, cond, diff = [out / x for x in ['autoencoder','latents','coarse','conditions','diffusion']]
    calibration = root / 'configs/projector_calibration.json'
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    def run(script, *args, distributed=False):
        prefix = [sys.executable]
        if distributed:
            prefix += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.gpus}']
        cmd = prefix + [str(root/'scripts'/script)] + list(map(str,args))
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, env=env, check=True)
    stages=['autoencoder','latents','coarse','conditions','diffusion','export']
    for stage in stages[stages.index(a.start_stage):]:
        if stage=='autoencoder':
            run('sota_14_train_autoencoder.py','--dataset-root',data,'--output-dir',ae,'--batch-size',a.batch_size,'--min-steps',60000,'--max-steps',60000,'--early-stop-validations',1000,distributed=True)
        elif stage=='latents':
            run('sota_15_cache_autoencoder_latents.py','--dataset-root',data,'--autoencoder-checkpoint',ae/'best_ceiling.pt','--output-dir',lat/'shard_00','--num-shards',1,'--shard-index',0)
            run('sota_16_finalize_autoencoder_latents.py','--dataset-root',data,'--autoencoder-checkpoint',ae/'best_ceiling.pt','--latent-root',lat,'--num-shards',1)
        elif stage=='coarse':
            run('sota_17_train_b0_cone_ddp.py','--dataset-root',data,'--latent-root',lat,'--autoencoder-checkpoint',lat/'autoencoder_frozen.pt','--projector-calibration',calibration,'--output-dir',coarse,'--batch-size',a.batch_size,distributed=True)
        elif stage=='conditions':
            run('sota_18_cache_b0_conditions.py','--dataset-root',data,'--latent-root',lat,'--coarse-checkpoint',coarse/'best.pt','--output-dir',cond/'shard_00','--num-shards',1,'--shard-index',0)
            run('sota_19_finalize_b0_conditions.py','--dataset-root',data,'--cache-root',cond,'--num-shards',1)
        elif stage=='diffusion':
            run('sota_20_train_b0_diffusion_ddp.py','--dataset-root',data,'--latent-root',lat,'--coarse-cache-root',cond,'--autoencoder-checkpoint',lat/'autoencoder_frozen.pt','--projector-calibration',calibration,'--output-dir',diff,'--batch-size',a.batch_size,'--max-steps',50000,'--min-steps',50000,distributed=True)
        else:
            run('export_weights.py','--autoencoder',lat/'autoencoder_frozen.pt','--coarse',coarse/'best.pt','--diffusion',diff/'best.pt','--projection-stats',data/'projection_stats.json','--output',out/'weights')

if __name__ == '__main__':
    main()
