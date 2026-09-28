"""Check the GenAD data and weight paths and build the model without using a GPU."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401 - registers project modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='optional Stage-II GenAD checkpoint to load')
    args = parser.parse_args()

    root = ROOT
    os.chdir(root)
    cfg = Config.fromfile(str(root / 'projects/configs/GenAD/GenAD_config.py'))
    eeg_dir = root / 'projects/mmdet3d_plugin/eeg_vedio'
    with (eeg_dir / 'cfgs/video_encoder_inference.yaml').open() as stream:
        video_cfg = yaml.safe_load(stream)['video_encoder']

    required = {
        'train annotations': root / cfg.data.train.ann_file,
        'validation annotations': root / cfg.data.val.ann_file,
        'map annotations': root / cfg.data.val.map_ann_file,
        'initialization checkpoint': root / cfg.load_from,
        'video encoder checkpoint': eeg_dir / video_cfg['ckpt_load_path'],
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f'{label}: {path}')
        print(f'FOUND {label}: {path}')

    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    print(f'DATASET_OK samples={len(dataset)}')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print(f'MODEL_OK parameters={sum(p.numel() for p in model.parameters())}')

    if args.checkpoint:
        path = root / args.checkpoint
        if not path.is_file():
            raise FileNotFoundError(f'Stage-II checkpoint: {path}')
        load_checkpoint(model, str(path), map_location='cpu', strict=True)
        print(f'CHECKPOINT_OK {path}')


if __name__ == '__main__':
    main()
