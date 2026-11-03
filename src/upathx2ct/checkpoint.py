"""Single-file deployment checkpoint for the complete UPath-X2CT model."""
import math
from pathlib import Path

import torch

FILENAME = 'upath_x2ct_final.pt'
FORMAT = 'upath-x2ct-final'
COMPONENT_CONFIGS = {
    'autoencoder': 'autoencoder_config',
    'coarse': 'b0_config',
    'diffusion': 'diffusion_config',
}


def validate_final_checkpoint(checkpoint):
    if checkpoint.get('format') != FORMAT or checkpoint.get('format_version') != 1:
        raise ValueError('Expected a UPath-X2CT final checkpoint (format version 1)')
    components = checkpoint.get('components', {})
    if set(components) != set(COMPONENT_CONFIGS):
        raise ValueError('Final checkpoint must contain all three model components')
    for name, config in COMPONENT_CONFIGS.items():
        if config not in components[name] or not components[name].get('model'):
            raise ValueError(f'Missing model state or configuration: {name}')
    norm = checkpoint.get('normalization', {})
    if norm.get('projection_stats', {}).get('definition_split') != 'train':
        raise ValueError('Normalization must be fitted on train only')
    for name in ('ct_hu_scale', 'ct_hu_offset'):
        if not isinstance(norm.get(name), (float, int)) or not math.isfinite(norm[name]):
            raise ValueError(f'Invalid normalization: {name}')
    if norm['ct_hu_scale'] <= 0:
        raise ValueError('CT HU scale must be positive')


def load_final_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / FILENAME
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    validate_final_checkpoint(checkpoint)
    return checkpoint
