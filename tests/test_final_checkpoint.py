"""Validate the public single-file export contract with small synthetic states."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from upathx2ct.checkpoint import COMPONENT_CONFIGS, FILENAME, load_final_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_export_round_trip_preserves_tensors_and_normalization(tmp_path):
    args = []
    for name, key in COMPONENT_CONFIGS.items():
        path = tmp_path / (name + '.pt')
        torch.save({'model': {'weight': torch.arange(8, dtype=torch.float32)}, key: {'test': name},
                    'step': 22000, 'optimizer': {'private_training_state': True}}, path)
        args.extend(['--' + name, str(path)])
    stats = {'definition_split': 'train', 'views': {'frontal_ap': {'log1p_mean': 2.3}}}
    calibration = {'definition_split': 'train', 'views': {'frontal_ap': {'gain': 0.3}}}
    for name, value in [('stats', stats), ('calibration', calibration)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    target = tmp_path / 'export' / FILENAME
    args += ['--projection-stats', str(tmp_path / 'stats.json'), '--projector-calibration',
             str(tmp_path / 'calibration.json'), '--output', str(target)]
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_weights.py'), *args],
                   env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')), check=True)
    loaded = load_final_checkpoint(target)
    assert loaded['normalization']['projection_stats'] == stats
    assert loaded['normalization']['ct_hu_scale'] == 1262.
    assert loaded['normalization']['ct_hu_offset'] == 238.
    assert loaded['projector_calibration'] == calibration
    assert set(target.parent.iterdir()) == {target}
    assert load_final_checkpoint(target.parent)['format'] == loaded['format']
    for name, key in COMPONENT_CONFIGS.items():
        assert set(loaded['components'][name]) == {'model', key, 'step'}
        assert torch.equal(loaded['components'][name]['model']['weight'], torch.arange(8, dtype=torch.float32))


@pytest.mark.parametrize('change,message', [
    ('version', 'version'), ('component', 'three'), ('split', 'train'), ('scale', 'positive'),
])
def test_invalid_final_checkpoint_is_rejected(tmp_path, change, message):
    checkpoint = {'format': 'upath-x2ct-final', 'format_version': 1,
                  'components': {name: {'model': {'weight': torch.ones(1)}, key: {}}
                                 for name, key in COMPONENT_CONFIGS.items()},
                  'normalization': {'projection_stats': {'definition_split': 'train'},
                                    'ct_hu_scale': 1262., 'ct_hu_offset': 238.}}
    if change == 'version': checkpoint['format_version'] = 999
    elif change == 'component': del checkpoint['components']['coarse']
    elif change == 'split': checkpoint['normalization']['projection_stats']['definition_split'] = 'test'
    else: checkpoint['normalization']['ct_hu_scale'] = -1
    path = tmp_path / FILENAME
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message): load_final_checkpoint(path)
