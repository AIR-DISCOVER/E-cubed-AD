import os
import pickle
from PIL import Image

import torch
from torchvision import transforms


class FrontVideoClipLoader:
    """
    根据当前 sample token，从 nuscenes_infos_xxx.pkl 中找到当前样本，
    再取同一 scene 内的历史 CAM_FRONT 图像。

    输入:
        tokens: list[str], batch 内每个样本的 token

    输出:
        front_video_clip: [B, T, 3, 224, 224]
    """

    def __init__(
        self,
        info_path,
        num_frames=4,
        image_size=224,
        data_root=".",
    ):
        self.info_path = info_path
        self.num_frames = num_frames
        self.image_size = image_size
        self.data_root = data_root

        with open(info_path, "rb") as f:
            data = pickle.load(f)

        self.infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

        # token -> info，后面只需要传入当前 sample token
        self.token_to_info = {}
        for info in self.infos:
            self.token_to_info[info["token"]] = info

        # scene_token -> 按 timestamp 排序的 infos
        self.scene_to_infos = {}
        for info in self.infos:
            scene_token = info["scene_token"]
            self.scene_to_infos.setdefault(scene_token, []).append(info)

        for scene_token in self.scene_to_infos:
            self.scene_to_infos[scene_token] = sorted(
                self.scene_to_infos[scene_token],
                key=lambda x: x["timestamp"],
            )

        # (scene_token, timestamp) -> index
        self.timestamp_to_index = {}
        for scene_token, scene_infos in self.scene_to_infos.items():
            for idx, info in enumerate(scene_infos):
                self.timestamp_to_index[(scene_token, info["timestamp"])] = idx

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    #新增
    def _select_infos_for_one_sample(self, scene_token, timestamp):
        if scene_token not in self.scene_to_infos:
            raise KeyError(f"scene_token not found: {scene_token}")

        if (scene_token, timestamp) not in self.timestamp_to_index:
            raise KeyError(
                f"(scene_token, timestamp) not found: "
                f"scene_token={scene_token}, timestamp={timestamp}"
            )

        scene_infos = self.scene_to_infos[scene_token]
        cur_idx = self.timestamp_to_index[(scene_token, timestamp)]

        start_idx = max(0, cur_idx - self.num_frames + 1)
        selected = scene_infos[start_idx:cur_idx + 1]

        while len(selected) < self.num_frames:
            selected = [selected[0]] + selected

        if len(selected) != self.num_frames:
            raise RuntimeError(
                f"selected frame number mismatch: "
                f"len(selected)={len(selected)}, expected={self.num_frames}"
            )

        if selected[-1]["timestamp"] != timestamp:
            raise RuntimeError(
                f"Last selected frame is not current frame: "
                f"last={selected[-1]['timestamp']}, current={timestamp}"
            )

        if any(info["timestamp"] > timestamp for info in selected):
            raise RuntimeError(
                f"Future frame detected: "
                f"selected={[info['timestamp'] for info in selected]}, "
                f"current={timestamp}"
            )

        return selected
    def _load_image(self, path):
        if path.startswith("./"):
            path = path[2:]
        path = os.path.join(self.data_root, path)
        img = Image.open(path).convert("RGB")
        return self.preprocess(img)

    # def _get_clip_for_one_sample(self, scene_token, timestamp):
    #     scene_infos = self.scene_to_infos[scene_token]
    #     cur_idx = self.timestamp_to_index[(scene_token, timestamp)]

    #     start_idx = max(0, cur_idx - self.num_frames + 1)
    #     selected = scene_infos[start_idx:cur_idx + 1]

    #     # 如果在 scene 开头，不足 T 帧，用最早帧补齐
    #     while len(selected) < self.num_frames:
    #         selected = [selected[0]] + selected

    #     frames = []
    #     for info in selected:
    #         cam_front_path = info["cams"]["CAM_FRONT"]["data_path"]
    #         frames.append(self._load_image(cam_front_path))

    #     return torch.stack(frames, dim=0)  # [T, 3, 224, 224]
    def _get_clip_for_one_sample(self, scene_token, timestamp):
        selected = self._select_infos_for_one_sample(scene_token, timestamp)

        frames = []
        for info in selected:
            cam_front_path = info["cams"]["CAM_FRONT"]["data_path"]
            frames.append(self._load_image(cam_front_path))

        return torch.stack(frames, dim=0)  # [T, 3, 224, 224]

    # def _get_clip_by_token(self, token):
    #     info = self.token_to_info[token]
    #     scene_token = info["scene_token"]
    #     timestamp = info["timestamp"]
    #     return self._get_clip_for_one_sample(scene_token, timestamp)
    def _get_clip_by_token(self, token):
        if token not in self.token_to_info:
            raise KeyError(f"token not found in info file: {token}")

        info = self.token_to_info[token]
        scene_token = info["scene_token"]
        timestamp = info["timestamp"]

        return self._get_clip_for_one_sample(scene_token, timestamp)

    # def __call__(self, tokens):
    #     clips = []
    #     for token in tokens:
    #         clips.append(self._get_clip_by_token(token))
    #     return torch.stack(clips, dim=0)  # [B, T, 3, 224, 224]
    def __call__(self, tokens):
        clips = []

        for idx, token in enumerate(tokens):
            clips.append(self._get_clip_by_token(token))

            if idx == 0 and not hasattr(self, "_debug_alignment_printed"):
                info = self.token_to_info[token]
                scene_token = info["scene_token"]
                timestamp = info["timestamp"]
                selected = self._select_infos_for_one_sample(scene_token, timestamp)

                print(
                    "[DEBUG][FrontVideoClipLoader]",
                    "current_token:", token,
                    "current_timestamp:", timestamp,
                    "selected_tokens:", [x["token"] for x in selected],
                    "selected_timestamps:", [x["timestamp"] for x in selected],
                    "last_token_match:", selected[-1]["token"] == token,
                    "non_future:", all(x["timestamp"] <= timestamp for x in selected),
                    flush=True,
                )

                self._debug_alignment_printed = True

        return torch.stack(clips, dim=0)  # [B, T, 3, 224, 224]
