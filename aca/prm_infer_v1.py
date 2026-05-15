from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from qwen_prm_support import QwenVLPRMBetaBinomV1Model, _build_messages, update_processor_pixels


@dataclass(frozen=True)
class PRMConfig:
    checkpoint: str
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    auto_device_map: bool = False
    backend: str = "internvl"
    base_model_path: str = ""
    dataset_name: str = ""
    processor_use_fast: bool = False
    max_seq_length: int = 32768
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    video_min_pixels: int = 128 * 28 * 28
    video_max_pixels: int = 768 * 28 * 28
    video_min_frames: int = 4
    video_max_frames: int = 128
    video_fps: float = 2.0
    grid_max_cols: int = 3
    load_weights_to_gpu: bool = False
    bf16: bool = True


@dataclass
class LoadedPRM:
    backend: str
    model: Any
    tokenizer: Any
    processor: Any = None
    prm_token_id: Optional[int] = None
    reward_token_ids: Optional[List[int]] = None
    dataset_name: str = ""
    max_seq_length: int = 32768
    grid_max_cols: int = 3

    @property
    def config(self):
        if self.backend == "internvl":
            return getattr(self.model, "config", None)
        if hasattr(self.model, "base_model"):
            return getattr(self.model.base_model, "config", None)
        return getattr(self.model, "config", None)


def _force_module_floating_dtype(module: torch.nn.Module, dtype: Optional[torch.dtype]) -> None:
    if dtype is None:
        return
    module.to(dtype=dtype)
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
    for name, buf in module.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != dtype:
            module._buffers[name] = buf.to(dtype=dtype)


def _collect_unexpected_floating_dtypes(module: torch.nn.Module, dtype: Optional[torch.dtype]) -> List[str]:
    if dtype is None:
        return []
    bad: List[str] = []
    for name, param in module.named_parameters():
        if param.is_floating_point() and param.dtype != dtype:
            bad.append(f"param:{name}={param.dtype}")
    for name, buf in module.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != dtype:
            bad.append(f"buffer:{name}={buf.dtype}")
    return bad


def _find_checkpoint_file(checkpoint: str) -> str:
    if os.path.isfile(checkpoint):
        return checkpoint
    cand_names = [
        "model.safetensors",
        "pytorch_model.safetensors",
        "pytorch_model.bin",
    ]
    for name in cand_names:
        p = os.path.join(checkpoint, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find a model checkpoint file under {checkpoint}. Tried: {cand_names}")


def _load_state_dict(ckpt_file: str) -> dict:
    if ckpt_file.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(ckpt_file)
    return torch.load(ckpt_file, map_location="cpu")


def _load_internvl_prm_model(cfg: PRMConfig) -> LoadedPRM:
    from internvl.model import split_model
    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat_beta_binom import (
        InternVLChatModel as InternVLChatBetaBinomModel,
    )

    use_auto_map = bool(cfg.auto_device_map and torch.cuda.device_count() > 1)
    if use_auto_map:
        config = InternVLChatConfig.from_pretrained(cfg.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
        if "kappa_head" not in device_map:
            device_map["kappa_head"] = 0
        kwargs = {"device_map": device_map}
    else:
        kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatBetaBinomModel.from_pretrained(
        cfg.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        **kwargs,
    ).eval()
    if not cfg.load_in_8bit and not cfg.load_in_4bit and not use_auto_map:
        model = model.cuda()
    return LoadedPRM(backend="internvl", model=model, tokenizer=tokenizer)


def _load_qwenvl_prm_model(cfg: PRMConfig) -> LoadedPRM:
    if not cfg.base_model_path:
        raise ValueError("--prm-base-model is required for backend=qwenvl_beta_binom_v1")

    class _ProcArgs:
        def __init__(self, src: PRMConfig):
            self.min_pixels = src.min_pixels
            self.max_pixels = src.max_pixels
            self.video_min_pixels = src.video_min_pixels
            self.video_max_pixels = src.video_max_pixels
            self.video_min_frames = src.video_min_frames
            self.video_max_frames = src.video_max_frames
            self.video_fps = src.video_fps

    processor = AutoProcessor.from_pretrained(cfg.base_model_path, use_fast=cfg.processor_use_fast)
    processor = update_processor_pixels(processor, _ProcArgs(cfg))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.checkpoint,
            model_max_length=cfg.max_seq_length,
            padding_side="right",
            use_fast=False,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_path,
            model_max_length=cfg.max_seq_length,
            padding_side="right",
            use_fast=False,
        )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    prm_token = "<prm>"
    tokenizer.add_tokens([prm_token], special_tokens=True)
    prm_token_id = tokenizer.convert_tokens_to_ids(prm_token)
    prm_pieces = tokenizer.tokenize(prm_token)
    if len(prm_pieces) != 1 or prm_pieces[0] != prm_token:
        raise ValueError(f"`{prm_token}` is not a single token for this tokenizer: pieces={prm_pieces}")
    reward_tokens = ["Yes", "No"]
    reward_token_ids = tokenizer.convert_tokens_to_ids(reward_tokens)

    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )

    base_name = cfg.base_model_path
    torch_dtype = torch.bfloat16 if cfg.bf16 else None
    config = AutoConfig.from_pretrained(base_name)
    config.vocab_size = len(tokenizer)
    base_name_lower = base_name.lower()
    base_tail = Path(base_name.rstrip("/")).name.lower()
    if "qwen3" in base_name_lower and "a" in base_tail:
        base_model = Qwen3VLMoeForConditionalGeneration(config)
    elif "qwen3" in base_name_lower:
        base_model = Qwen3VLForConditionalGeneration(config)
    elif "qwen2.5" in base_name_lower:
        base_model = Qwen2_5_VLForConditionalGeneration(config)
    else:
        base_model = Qwen2VLForConditionalGeneration(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    base_model.to(device)
    base_model.eval()

    ckpt_file = _find_checkpoint_file(cfg.checkpoint)
    if ckpt_file.endswith(".safetensors") and device.type == "cuda" and cfg.load_weights_to_gpu:
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_file, device=str(device))
    else:
        state_dict = _load_state_dict(ckpt_file)

    model = QwenVLPRMBetaBinomV1Model(
        base_model=base_model,
        tokenizer=tokenizer,
        prm_token=prm_token,
        reward_tokens=reward_tokens,
    )
    model.load_state_dict(state_dict, strict=False)
    if torch_dtype is not None:
        _force_module_floating_dtype(model.base_model, torch_dtype)
        _force_module_floating_dtype(model.kappa_head, torch_dtype)
        _force_module_floating_dtype(model, torch_dtype)
    model.to(device)
    model.eval()
    bad_dtypes = _collect_unexpected_floating_dtypes(model, torch_dtype)
    if bad_dtypes:
        preview = ", ".join(bad_dtypes[:8])
        raise RuntimeError(
            f"Qwen PRM dtype normalization failed for target dtype={torch_dtype}. "
            f"Unexpected floating tensors: {preview}"
        )
    return LoadedPRM(
        backend="qwenvl_beta_binom_v1",
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        prm_token_id=int(prm_token_id),
        reward_token_ids=[int(x) for x in reward_token_ids],
        dataset_name=str(cfg.dataset_name),
        max_seq_length=int(cfg.max_seq_length),
        grid_max_cols=int(cfg.grid_max_cols),
    )


def load_prm_model(cfg: PRMConfig) -> Tuple[LoadedPRM, Any]:
    backend = str(cfg.backend).strip().lower()
    if backend == "internvl":
        loaded = _load_internvl_prm_model(cfg)
    elif backend == "qwenvl_beta_binom_v1":
        loaded = _load_qwenvl_prm_model(cfg)
    else:
        raise ValueError(f"Unsupported PRM backend: {cfg.backend}")
    return loaded, loaded.tokenizer


@torch.no_grad()
def _batch_prm_mu_kappa_sigma_internvl(
    loaded: LoadedPRM,
    pixel_values: torch.Tensor,
    prompts: Sequence[str],
    num_patches_list: Sequence[Union[int, Sequence[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from internvl.conversation import get_conv_template

    model = loaded.model
    tokenizer = loaded.tokenizer
    prm_token_id = tokenizer.convert_tokens_to_ids("<prm>")
    reward_token_ids = tokenizer.convert_tokens_to_ids(["Yes", "No"])
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id

    queries: List[str] = []
    for idx, question in enumerate(prompts):
        patch_spec = num_patches_list[idx]
        if isinstance(patch_spec, Sequence) and not isinstance(patch_spec, (str, bytes)):
            prompt_num_patches = [int(x) for x in patch_spec]
        else:
            prompt_num_patches = [int(patch_spec)]
        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question
        template = get_conv_template(model.template)
        template.append_message(template.roles[0], "")
        template.append_message(template.roles[1], question)
        query = template.get_prompt()
        for num_patches in prompt_num_patches:
            image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * int(num_patches) + "</img>"
            query = query.replace("<image>", image_tokens, 1)
        queries.append(query)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    tokenizer.padding_side = old_padding_side

    device = pixel_values.device
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    outputs = model.forward(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long, device=device),
        output_hidden_states=True,
        return_dict=True,
    )

    logits = outputs.logits
    placeholder_mask = input_ids == prm_token_id
    selected_logits = logits[placeholder_mask]
    selected_logits = selected_logits[..., reward_token_ids]
    if selected_logits.numel() == 0:
        empty = selected_logits.new_zeros((0,), dtype=torch.float32)
        return empty, empty, empty

    mu = torch.softmax(selected_logits, dim=-1)[..., 0]
    if hasattr(model, "kappa_head") and model.kappa_head is not None:
        hidden = outputs.hidden_states[-1]
        prm_h = hidden[placeholder_mask]
        z_kappa = model.kappa_head(prm_h).squeeze(-1)
        kappa_min = float(getattr(model, "beta_binom_kappa_min", 1e-3))
        kappa = torch.nn.functional.softplus(z_kappa) + kappa_min
    else:
        kappa_floor = float(getattr(model, "beta_binom_kappa_floor", 2.0))
        kappa = torch.nn.functional.softplus(selected_logits[..., 1]) + kappa_floor
    sigma = torch.sqrt(mu * (1.0 - mu) / (kappa + 1.0))
    return mu.float(), kappa.float(), sigma.float()


def _tile_pil_images(images: List[Image.Image], grid_max_cols: int = 3) -> Image.Image:
    images = [im.convert("RGB") for im in images]
    if len(images) == 1:
        return images[0]
    cols = min(grid_max_cols, len(images))
    rows = (len(images) + cols - 1) // cols
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    canvas = Image.new("RGB", (cols * max_w, rows * max_h), (255, 255, 255))
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * max_w + (max_w - img.width) // 2
        y = r * max_h + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


def _qwen_use_native_multi_image(dataset_name: str) -> bool:
    ds = str(dataset_name or "").lower()
    return "olympiadbench" in ds


@torch.no_grad()
def _batch_prm_mu_kappa_sigma_qwen(
    loaded: LoadedPRM,
    prompts: Sequence[str],
    image_paths: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = loaded.model
    processor = loaded.processor
    prm_token_id = int(loaded.prm_token_id)
    reward_token_ids = list(loaded.reward_token_ids or [])
    device = next(model.parameters()).device

    sample_images = [Image.open(p).convert("RGB") for p in image_paths]
    rendered_prompts: List[str] = []
    native_multi_image = bool(image_paths) and _qwen_use_native_multi_image(loaded.dataset_name)
    if native_multi_image:
        user_text = "<image>\n" * len(image_paths)
        msg_image_field: Union[str, List[str]] = list(image_paths)
        proc_images = [sample_images for _ in range(len(prompts))]
    elif image_paths:
        tiled_image = _tile_pil_images(sample_images, grid_max_cols=int(loaded.grid_max_cols))
        user_text = "<image>\n"
        msg_image_field = [image_paths[0]]
        proc_images = [[tiled_image] for _ in range(len(prompts))]
    else:
        user_text = ""
        msg_image_field = []
        proc_images = None

    for prompt in prompts:
        tmp = {
            "image": msg_image_field,
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": prompt},
            ],
        }
        msgs = _build_messages(tmp, Path("."))
        prompt_text = processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered_prompts.append(prompt_text)

    if proc_images is None:
        proc_out = processor(
            text=rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    else:
        proc_out = processor(
            text=rendered_prompts,
            images=proc_images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

    input_ids_cpu = proc_out.get("input_ids", None)
    if (
        input_ids_cpu is not None
        and loaded.max_seq_length > 0
        and int(input_ids_cpu.shape[1]) > int(loaded.max_seq_length)
    ):
        raise RuntimeError(
            f"Tokenized input too long for Qwen PRM: tokenized_len={int(input_ids_cpu.shape[1])}, "
            f"max_seq_length={int(loaded.max_seq_length)}"
        )

    proc_out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc_out.items()}
    outputs, last_hidden_state = model._forward_base(proc_out, need_last_hidden_state=True)
    if last_hidden_state is None:
        raise RuntimeError("QwenVL Beta evaluator requires last_hidden_state, but none was returned.")

    input_ids = proc_out["input_ids"]
    placeholder_mask = input_ids == prm_token_id
    logits = outputs.logits
    selected_logits = logits[placeholder_mask]
    if selected_logits.numel() == 0:
        empty = logits.new_zeros((0,), dtype=torch.float32)
        return empty, empty, empty
    selected_logits = selected_logits[..., reward_token_ids]
    mu = torch.softmax(selected_logits.float(), dim=-1)[..., 0]

    prm_hidden = last_hidden_state[placeholder_mask]
    if not hasattr(model, "kappa_head") or model.kappa_head is None:
        raise RuntimeError("QwenVL Beta evaluator expects a Beta PRM V1 checkpoint with kappa_head.")
    kappa_logits = model.kappa_head(prm_hidden.to(model.kappa_head[-1].weight.dtype)).squeeze(-1)
    kappa_min = float(getattr(model, "beta_binom_kappa_min", 1e-3))
    kappa = torch.nn.functional.softplus(kappa_logits.float()) + kappa_min
    sigma = torch.sqrt(mu * (1.0 - mu) / (kappa + 1.0))
    return mu.float(), kappa.float(), sigma.float()


@torch.no_grad()
def batch_prm_mu_kappa_sigma(
    model: LoadedPRM,
    tokenizer: Any,
    pixel_values: Optional[torch.Tensor],
    prompts: Sequence[str],
    num_patches_list: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    image_paths: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.backend == "internvl":
        if pixel_values is None or num_patches_list is None:
            raise ValueError("InternVL PRM requires pixel_values and num_patches_list.")
        return _batch_prm_mu_kappa_sigma_internvl(
            loaded=model,
            pixel_values=pixel_values,
            prompts=prompts,
            num_patches_list=num_patches_list,
        )
    if model.backend == "qwenvl_beta_binom_v1":
        return _batch_prm_mu_kappa_sigma_qwen(
            loaded=model,
            prompts=prompts,
            image_paths=list(image_paths or []),
        )
    raise ValueError(f"Unsupported loaded PRM backend: {model.backend}")


def score_steps_mu_minus_lambda_sigma(
    mu: Sequence[float], sigma: Sequence[float], risk_lambda: float
) -> List[float]:
    out = []
    lam = float(risk_lambda)
    for m, s in zip(mu, sigma):
        out.append(float(m) - lam * float(s))
    return out


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")
