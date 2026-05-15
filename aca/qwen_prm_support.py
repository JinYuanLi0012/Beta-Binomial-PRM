from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import transformers


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels

    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages: List[Dict[str, Any]] = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = str(turn["value"])
        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)
            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("Number of <image> placeholders exceeds provided images.")
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("Number of <video> placeholders exceeds provided videos.")
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused.")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused.")
    return messages


class QwenVLPRMBetaBinomV1Model(nn.Module):
    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        prm_token: str = "<prm>",
        reward_tokens: Optional[List[str]] = None,
        beta_binom_eps: float = 1e-6,
        beta_binom_kappa_min: float = 1e-4,
        beta_binom_kappa_init: float = 4.0,
        beta_binom_evi_reg: float = 0.01,
        beta_debug_interval: int = 50,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        if reward_tokens is None:
            reward_tokens = ["Yes", "No"]

        self.prm_token_id = tokenizer.convert_tokens_to_ids(prm_token)
        self.reward_token_ids = tokenizer.convert_tokens_to_ids(reward_tokens)
        self.beta_binom_eps = float(beta_binom_eps)
        self.beta_binom_kappa_min = float(beta_binom_kappa_min)
        self.beta_binom_kappa_init = float(beta_binom_kappa_init)
        self.beta_binom_evi_reg = float(beta_binom_evi_reg)
        self.beta_debug_interval = int(beta_debug_interval)

        hidden_size = int(getattr(base_model.config, "hidden_size"))
        self.kappa_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.reset_kappa_head(self.beta_binom_kappa_init)

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        if x <= 20.0:
            return math.log(math.expm1(x))
        return x

    def reset_kappa_head(self, kappa_init: Optional[float] = None) -> None:
        target = float(self.beta_binom_kappa_init if kappa_init is None else kappa_init)
        target = max(target - self.beta_binom_kappa_min, 1e-6)
        bias_value = self._inverse_softplus(target)
        linear = self.kappa_head[-1]
        with torch.no_grad():
            nn.init.normal_(linear.weight, mean=0.0, std=1e-3)
            linear.bias.fill_(bias_value)

    def _forward_base(self, model_inputs: Dict[str, Any], need_last_hidden_state: bool):
        base_inputs = {
            k: v for k, v in model_inputs.items() if k not in {"labels", "prm_counts_k", "prm_counts_n"}
        }
        outputs = self.base_model(**base_inputs, return_dict=True)

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if need_last_hidden_state and last_hidden_state is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                last_hidden_state = hidden_states[-1]

        if need_last_hidden_state and last_hidden_state is None:
            outputs = self.base_model(
                **base_inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError("QwenVL Beta PRM V1 needs the last hidden state but none was returned.")
            last_hidden_state = hidden_states[-1]

        return outputs, last_hidden_state
