"""Paper inference from calibrated projections; no target CT is required."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from upathx2ct.b0.autoencoder import load_autoencoder_checkpoint
from upathx2ct.b0.diffusion import DiffusionConfig, ResidualLatentDiffusion
from upathx2ct.b0.ema import resolve_checkpoint_model_state
from upathx2ct.b0.training import autocast_context
from upathx2ct.sota.data import normalize_drr
from upathx2ct.sota.model import load_b0_coarse_checkpoint


def stable_seed(base_seed, case_id, sample_index):
    digest = hashlib.sha256(f'{base_seed}:{case_id}:{sample_index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2**63 - 1)


def load_diffusion(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    allowed = {f.name for f in fields(DiffusionConfig)}
    cfg = {k: v for k, v in checkpoint['diffusion_config'].items() if k in allowed}
    for key in ['model_channels', 'num_res_blocks', 'attention_levels']:
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    model = ResidualLatentDiffusion(DiffusionConfig(**cfg))
    model.load_state_dict(resolve_checkpoint_model_state(checkpoint), strict=True)
    return model.to(device).eval().requires_grad_(False)


def read_projections(path):
    """NPZ/HDF5: negative-log DRRs and centered-LPS camera matrices."""
    path = Path(path)
    opener = h5py.File if path.suffix in ['.h5', '.hdf5'] else np.load
    with opener(path, 'r') if opener is h5py.File else opener(path, allow_pickle=False) as f:
        ap = np.asarray(f['drr_frontal_ap'], dtype=np.float32)
        lat = np.asarray(f['drr_left_lateral'], dtype=np.float32)
        matrices = np.asarray(f['projection_matrices'], dtype=np.float32)
    if ap.shape != (512, 512) or lat.shape != (512, 512) or matrices.shape != (2, 3, 4):
        raise ValueError('Expected AP/LAT (512,512) and projection_matrices (2,3,4)')
    if not all(np.isfinite(x).all() for x in [ap, lat, matrices]):
        raise ValueError('Input contains non-finite values')
    if any(np.linalg.matrix_rank(m) != 3 for m in matrices):
        raise ValueError('Projection matrices must have rank 3')
    return ap, lat, matrices


class Predictor:
    def __init__(self, weights, device='cuda', precision='bf16'):
        weights = Path(weights)
        self.device = torch.device(device)
        self.precision = precision
        if self.device.type == 'cpu' and precision != 'fp32':
            raise ValueError('Use --precision fp32 for CPU inference')
        self.stats = json.loads((weights / 'projection_stats.json').read_text())
        if self.stats['definition_split'] != 'train':
            raise ValueError('Normalization must be fitted on train only')
        self.autoencoder = load_autoencoder_checkpoint(str(weights / 'autoencoder.pt'), 'cpu').to(self.device).eval().requires_grad_(False)
        self.coarse = load_b0_coarse_checkpoint(weights / 'coarse.pt', self.device).eval().requires_grad_(False)
        self.diffusion = load_diffusion(weights / 'diffusion.pt', self.device)

    @torch.inference_mode()
    def predict(self, path, case_id, samples=8, steps=50, seed=20260724):
        if samples < 1 or not 2 <= steps <= 1000:
            raise ValueError('Require samples >= 1 and 2 <= steps <= 1000')
        ap, lat, matrices = read_projections(path)
        views = []
        for raw, key in [(ap, 'frontal_ap'), (lat, 'left_lateral')]:
            s = self.stats['views'][key]
            image = normalize_drr(raw, s['log1p_mean'], s['log1p_std'])
            views.append(torch.from_numpy(image)[None, None].to(self.device))
        matrices = torch.from_numpy(matrices)[None].to(self.device)
        with autocast_context(self.device, self.precision):
            coarse = self.coarse(*views, matrices)
            decoded = []
            for k in range(samples):
                generator = torch.Generator(device=self.device).manual_seed(stable_seed(seed, case_id, k))
                latent = self.diffusion.sample(coarse['coarse_latent'], coarse['condition'], steps=steps, eta=0., generator=generator)
                decoded.append(self.autoencoder.decode(latent, normalized=True, clamp=True).float())
            stack = torch.stack(decoded)
            mean = stack.mean(0)[0, 0].cpu().numpy() * 1262. + 238.
            std = stack.std(0, unbiased=False)[0, 0].cpu().numpy() * 1262.
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError('Non-finite reconstruction')
        return mean, std
