from inspect import signature
import os
import sys
import yaml

from projects.mmdet3d_plugin.models.cognitive.front_video_clip_loader import FrontVideoClipLoader

import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class V1SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        
        use_fake_video_cog_feature=False,
        fake_cog_dim=200,
        use_online_video_cog_feature=True,
        online_cog_num_frames=4,
        e3ad_root="third_party/e3ad",
        front_video_info_path="data/infos/nuscenes_infos_val.pkl",
        front_video_data_root=".",
    ):
        super(V1SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        
        self.use_fake_video_cog_feature = use_fake_video_cog_feature
        self.fake_cog_dim = fake_cog_dim
        self.use_online_video_cog_feature = use_online_video_cog_feature
        self.online_cog_num_frames = online_cog_num_frames
        self.e3ad_root = e3ad_root
        self.cognitive_video_encoder = None
        self.front_video_info_path = front_video_info_path
        self.front_video_data_root = front_video_data_root
        self.front_video_clip_loader = None

        if self.use_online_video_cog_feature:
            e3ad_root_abs = os.path.abspath(self.e3ad_root)
            eeg_src = os.path.join(e3ad_root_abs, "eeg_vedio_src")

            if e3ad_root_abs not in sys.path:
                sys.path.insert(0, e3ad_root_abs)
            if eeg_src not in sys.path:
                sys.path.insert(0, eeg_src)

            from models.video_encoder.video_encoder import VideoEncoder

            cfg_path = os.path.join(e3ad_root_abs, "cfgs", "video_encoder_inference.yaml")
            with open(cfg_path, "r") as f:
                raw_cfg = yaml.safe_load(f)

            video_cfg = raw_cfg["video_encoder"]
            video_cfg["ckpt_load_path"] = "ckpts/Driving_thinking_model.safetensors"

            self.cognitive_video_encoder = VideoEncoder(video_cfg, e3ad_root_abs)
            self.cognitive_video_encoder.load_ckpts()
            self.cognitive_video_encoder.eval()

            for p in self.cognitive_video_encoder.parameters():
                p.requires_grad_(False)

            self.front_video_clip_loader = FrontVideoClipLoader(
                info_path=self.front_video_info_path,
                num_frames=self.online_cog_num_frames,
                image_size=224,
                data_root=self.front_video_data_root,
            )

        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps
    def _make_repeated_front_clip_from_current_img(self, img):
        """
        Smoke test only.

        当前 DiffusionDrive img:
            [B, 6, 3, 256, 704]

        临时构造:
            front_video_clip: [B, T, 3, 224, 224]

        注意:
            这里是把当前 CAM_FRONT 重复 T 次，不是最终严格历史帧方案。
        """
        if img.dim() != 5:
            return None

        # 默认第 0 个 camera 是 CAM_FRONT；这和前面 E-VAD 取 img[:,:,0,...] 的思想一致。
        front = img[:, 0]  # [B, 3, H, W]

        # E3AD VideoEncoder 期望 224x224。当前图已经经过 Normalize，这里只做尺寸适配。
        front = F.interpolate(
            front.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )  # [B, 3, 224, 224]

        clip = front[:, None].repeat(
            1,
            self.online_cog_num_frames,
            1,
            1,
            1,
        )  # [B, T, 3, 224, 224]

        return clip

    def _extract_tokens_from_data(self, data):
        img_metas = data.get("img_metas", None)

        if img_metas is None:
            return None

        if isinstance(img_metas, list):
            tokens = []
            for meta in img_metas:
                if "token" not in meta:
                    return None
                tokens.append(meta["token"])
            return tokens

        if isinstance(img_metas, dict):
            if "token" not in img_metas:
                return None
            return [img_metas["token"]]

        return None
    def _extract_online_video_cog_feature(self, img, data=None):
        """
        frozen E3AD VideoEncoder:
            front_video_clip [B,T,3,224,224] -> video_cog_feature [B,200]
        """
        if not self.use_online_video_cog_feature:
            return None

        if self.cognitive_video_encoder is None:
            return None

        front_video_clip = None

        if data is not None and self.front_video_clip_loader is not None:
            tokens = self._extract_tokens_from_data(data)
            if tokens is not None:
                front_video_clip = self.front_video_clip_loader(tokens)

        # fallback: 如果 img_metas 里没拿到 token，就退回当前帧重复版
        if front_video_clip is None:
            front_video_clip = self._make_repeated_front_clip_from_current_img(img)

        if front_video_clip is None:
            return None

        device = next(self.cognitive_video_encoder.parameters()).device
        front_video_clip = front_video_clip.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            video_cog_feature = self.cognitive_video_encoder(front_video_clip)

        if not hasattr(self, "_debug_online_cog_printed"):
            print(
                "[DEBUG][V1SparseDrive] online front_video_clip:",
                tuple(front_video_clip.shape),
                "video_cog_feature:",
                tuple(video_cog_feature.shape),
                flush=True,
            )
            self._debug_online_cog_printed = True

        return video_cog_feature
    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)
        
    def forward_train(self, img, **data):
        if self.use_fake_video_cog_feature and "video_cog_feature" not in data:
            bs = img.shape[0]
            data["video_cog_feature"] = img.new_zeros((bs, self.fake_cog_dim))
            if not hasattr(self, "_debug_fake_cog_printed_train"):
                print(
                    "[DEBUG][V1SparseDrive] fake train video_cog_feature:",
                    tuple(data["video_cog_feature"].shape),
                    flush=True,
                )
                self._debug_fake_cog_printed_train = True

        if (
            self.use_online_video_cog_feature
            and "video_cog_feature" not in data
        ):
            data["video_cog_feature"] = self._extract_online_video_cog_feature(img, data)

        feature_maps, depths = self.extract_feat(img, True, data)

        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)
    def simple_test(self, img, **data):
        if self.use_fake_video_cog_feature and "video_cog_feature" not in data:
            bs = img.shape[0]
            data["video_cog_feature"] = img.new_zeros((bs, self.fake_cog_dim))
            if not hasattr(self, "_debug_fake_cog_printed_test"):
                print(
                    "[DEBUG][V1SparseDrive] fake test video_cog_feature:",
                    tuple(data["video_cog_feature"].shape),
                    flush=True,
                )
                self._debug_fake_cog_printed_test = True

        if (
            self.use_online_video_cog_feature
            and "video_cog_feature" not in data
        ):
            data["video_cog_feature"] = self._extract_online_video_cog_feature(img, data)

        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
