"""Export all trained components and normalization into upath_x2ct_final.pt."""
import argparse
import json
from pathlib import Path
import torch
from upathx2ct.b0.ema import resolve_checkpoint_model_state
from upathx2ct.checkpoint import COMPONENT_CONFIGS, FILENAME, FORMAT, validate_final_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['autoencoder', 'coarse', 'diffusion', 'projection-stats', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--projector-calibration', type=Path)
    a = p.parse_args()
    target = a.output if a.output.suffix == '.pt' else a.output / FILENAME
    if target.exists():
        raise FileExistsError(f'Use a new output path: {target}')
    components = {}
    for name, config_key in COMPONENT_CONFIGS.items():
        # Training checkpoints contain optimizer metadata: use trusted local files.
        source = torch.load(getattr(a, name), map_location='cpu', weights_only=False)
        components[name] = {
            'model': resolve_checkpoint_model_state(source),
            config_key: source[config_key],
            'step': source.get('step'),
        }
    checkpoint = {
        'format': FORMAT,
        'format_version': 1,
        'components': components,
        'normalization': {
            'projection_stats': json.loads(a.projection_stats.read_text()),
            'ct_hu_scale': 1262.,
            'ct_hu_offset': 238.,
        },
        'projector_calibration': json.loads(a.projector_calibration.read_text()) if a.projector_calibration else None,
    }
    validate_final_checkpoint(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.pt.tmp')
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Final model ready: {target}')


if __name__ == '__main__':
    main()
