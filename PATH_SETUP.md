# GenAD relative path layout

Run training and evaluation commands from the repository root. The GenAD config resolves its relative dataset and checkpoint paths from the current working directory.

Place the required files at these paths (or create local links to them):

```text
data/nuscenes/vad_nuscenes_infos_temporal_train.pkl
data/nuscenes/vad_nuscenes_infos_temporal_val.pkl
data/nuscenes/nuscenes_map_anns_val.json
data/nuscenes/samples/...
ckpts/checkpoints.pth
projects/mmdet3d_plugin/eeg_vedio/ckpts/video_encoder_stage1.safetensors
```

`ckpts/checkpoints.pth` is the training initialization checkpoint configured by `load_from`. The video encoder file must be the Stage-I checkpoint used for this GenAD experiment; its SHA-256 is `ec395dd765ec0ecfa2d212525258334c9ea468b3ed0b54f0c2231809ebc6cf33`. Other similarly sized video encoder files in the EEG project are different weights.

Stage-II GenAD checkpoints used for evaluation belong under `path/` and are passed explicitly to `tools/test.py`. The EEG pretraining utilities expect optional real-car data under `projects/mmdet3d_plugin/eeg_vedio/data/real_car/`.

Datasets, checkpoints, generated results, and logs are excluded by `.gitignore`.

The server's `genad` environment also needs the local `mmdetection3d` source on `PYTHONPATH`. `test_8gpu.sh` sets this from `MMDET3D_ROOT`; its default is a relative sibling path for the current server layout. Set `MMDET3D_ROOT` when using a different checkout layout. The setup check can be run from the repository root with:

```bash
MMDET3D_ROOT=../../../wanghan_snapshot/mmdetection3d
PYTHONPATH="${MMDET3D_ROOT}:." conda run --no-capture-output -n genad python tools/verify_genad_setup.py --checkpoint path/retrain_20260906_bev100_20ep/epoch_20.pth
```
