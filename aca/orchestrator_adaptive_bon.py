from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
import torch
from PIL import Image
from tqdm import tqdm

from internvl.train.dataset import build_transform, dynamic_preprocess

from judge_client import JudgeConfig, judge_yes_no
from prm_infer_v1 import (
    PRMConfig,
    batch_prm_mu_kappa_sigma,
    load_prm_model,
    mean,
    score_steps_mu_minus_lambda_sigma,
)
from resume_utils import load_completed_outputs, make_item_tag
from utils_merge_and_cut import (
    CutConfig,
    merge_prefix_continuation,
    pick_cutpoint,
    split_steps_answer,
)


@dataclass
class Candidate:
    segments: List[str]
    token_len: int = 0
    origin: str = ""
    round_added: int = -1

    prm_mu: List[float] = field(default_factory=list)
    prm_kappa: List[float] = field(default_factory=list)
    prm_sigma: List[float] = field(default_factory=list)
    prm_score_steps: List[float] = field(default_factory=list)

    score: float = float("nan")
    u_sigma: float = float("nan")  # uncertainty aggregate over reasoning steps only
    lcb: float = float("nan")
    ucb: float = float("nan")

    @property
    def reasoning_steps(self) -> List[str]:
        steps, _ = split_steps_answer(self.segments)
        return steps

    @property
    def answer(self) -> str:
        _, ans = split_steps_answer(self.segments)
        return ans


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def gen_from_scratch(
    gen_url: str,
    question: str,
    image_path: Optional[Union[str, List[str]]],
    k: int,
    oversample_factor: float,
    sampling: Dict[str, Any],
    return_raw: bool = False,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    sampling_payload = dict(sampling)
    seed = sampling_payload.pop("seed", None)
    payload = {
        "question": question,
        "image_path": image_path,
        "k": int(k),
        "oversample_factor": float(oversample_factor),
        "sampling": sampling_payload,
        "seed": seed,
        "return_raw": bool(return_raw),
    }
    out = _post_json(gen_url.rstrip("/") + "/generate_from_scratch", payload)
    cands = [
        Candidate(
            segments=item["segments"],
            token_len=int(item.get("token_len", 0)),
            origin="scratch",
        )
        for item in out.get("selected", [])
    ]
    return cands, out


def gen_continue(
    gen_url: str,
    question: str,
    image_path: Optional[Union[str, List[str]]],
    prefix_steps: Sequence[str],
    m: int,
    oversample_factor: float,
    sampling: Dict[str, Any],
    return_raw: bool = False,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    sampling_payload = dict(sampling)
    seed = sampling_payload.pop("seed", None)
    payload = {
        "question": question,
        "image_path": image_path,
        "prefix_steps": list(prefix_steps),
        "m": int(m),
        "oversample_factor": float(oversample_factor),
        "sampling": sampling_payload,
        "seed": seed,
        "return_raw": bool(return_raw),
    }
    out = _post_json(gen_url.rstrip("/") + "/generate_continue", payload)
    cands = [
        Candidate(
            segments=item["segments"],
            token_len=int(item.get("token_len", 0)),
            origin="continue",
        )
        for item in out.get("selected", [])
    ]
    return cands, out


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _safe_mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _resolve_one_image_path(
    raw_image_path: str,
    input_base_dir: str,
    image_root: str,
) -> str:
    if os.path.isabs(raw_image_path):
        return raw_image_path

    cand1 = os.path.normpath(os.path.join(input_base_dir, raw_image_path))
    cand2 = os.path.normpath(os.path.join(image_root, raw_image_path)) if image_root else ""
    cand3 = os.path.normpath(os.path.join(os.getcwd(), raw_image_path))
    if os.path.exists(cand1):
        return cand1
    if cand2 and os.path.exists(cand2):
        return cand2
    if os.path.exists(cand3):
        return cand3
    return raw_image_path


def _normalize_correct_answer(correct_answer: Any) -> Optional[str]:
    if correct_answer is None:
        return None
    if isinstance(correct_answer, list):
        parts = [str(x).strip() for x in correct_answer if str(x).strip()]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts)
    text = str(correct_answer).strip()
    return text or None


def _prepare_question_for_images(question: str, num_images: int) -> str:
    if num_images <= 0:
        return question
    existing = question.count("<image>")
    need = max(0, int(num_images) - int(existing))
    if need <= 0:
        return question
    return ("<image>\n" * need) + question


def _normalize_item_inputs(
    item: Dict[str, Any],
    *,
    input_base_dir: str,
    image_root: str,
) -> Dict[str, Any]:
    question_raw = str(item.get("question", "") or "")

    raw_image_paths = item.get("image_paths", None)
    resolved_image_paths: List[str] = []
    if isinstance(raw_image_paths, list) and raw_image_paths:
        for raw_path in raw_image_paths:
            path_text = str(raw_path).strip()
            if path_text:
                resolved_image_paths.append(
                    _resolve_one_image_path(path_text, input_base_dir=input_base_dir, image_root=image_root)
                )
    else:
        raw_image_path = item.get("image_path") or item.get("image") or ""
        raw_image_path = str(raw_image_path).strip() if raw_image_path else ""
        if raw_image_path:
            resolved_image_paths.append(
                _resolve_one_image_path(raw_image_path, input_base_dir=input_base_dir, image_root=image_root)
            )

    if len(resolved_image_paths) == 0:
        resolved_image_input: Optional[Union[str, List[str]]] = None
    elif len(resolved_image_paths) == 1:
        resolved_image_input = resolved_image_paths[0]
    else:
        resolved_image_input = list(resolved_image_paths)

    return {
        "question_raw": question_raw,
        "question_model": _prepare_question_for_images(question_raw, len(resolved_image_paths)),
        "resolved_image_input": resolved_image_input,
        "resolved_image_paths": resolved_image_paths,
        "correct_answer_text": _normalize_correct_answer(item.get("correct_answer", None)),
    }


def _load_tokenizer_for_baseline(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load baseline tokenizer from {path!r}: {e}") from e


def _compute_baseline_a_effective_tokens(
    *,
    baseline_json_path: str,
    tokenizer,
    item_ids: List[str],
) -> Dict[str, Any]:
    """
    Approximate Baseline-A effective completion tokens offline:
      - baseline json has solutions_splits: List[rollout], each rollout is List[str] segments
      - we approximate completion tokens by tokenizing the concatenated rollout text (no prompt)
    """
    baseline_data = json.load(open(baseline_json_path, "r", encoding="utf-8"))
    if not isinstance(baseline_data, list):
        raise ValueError("baseline_a_json must be a JSON list")

    wanted = set(str(i) for i in item_ids)
    id_to_item: Dict[str, Any] = {}
    for it in baseline_data:
        if not isinstance(it, dict):
            continue
        key = str(it.get("id", ""))
        if key in wanted:
            id_to_item[key] = it

    per_item_tokens: List[int] = []
    covered = 0
    missing: List[str] = []

    for iid in item_ids:
        key = str(iid)
        it = id_to_item.get(key)
        if it is None:
            missing.append(key)
            continue
        sols = it.get("solutions_splits", None)
        if not isinstance(sols, list):
            missing.append(key)
            continue

        texts: List[str] = []
        for rollout in sols:
            if not isinstance(rollout, list):
                continue
            parts = [str(s).strip() for s in rollout if str(s).strip()]
            texts.append("\n".join(parts))

        enc = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        lens = [len(ids) for ids in enc.get("input_ids", [])]
        per_item_tokens.append(int(sum(lens)))
        covered += 1

    return {
        "path": baseline_json_path,
        "n_items_requested": int(len(item_ids)),
        "n_items_covered": int(covered),
        "missing_ids": missing[:50],  # cap
        "per_item_effective_tokens": per_item_tokens,
        "effective_tokens_total": int(sum(per_item_tokens)),
        "effective_tokens_mean": _safe_mean([float(x) for x in per_item_tokens]),
        "effective_tokens_median": _safe_median([float(x) for x in per_item_tokens]),
    }


def _compute_and_save_summary(
    *,
    outputs: List[Dict[str, Any]],
    output_json_path: str,
    save_intermediate_dir: str,
    baseline_a_json: str,
    baseline_tokenizer_path: str,
    prm_tokenizer,
    n_total: int,
) -> Dict[str, Any]:
    n = int(len(outputs))
    is_correct = [bool(it.get("is_correct_best")) for it in outputs]
    num_correct = int(sum(1 for x in is_correct if x))
    accuracy = float(num_correct / n) if n else float("nan")

    stop_pools: List[int] = []
    early_stop_flags: List[bool] = []
    early_stop_saved_budget_flags: List[bool] = []
    eff_tokens: List[int] = []
    act_tokens: List[int] = []
    item_ids: List[str] = []

    for it in outputs:
        item_ids.append(str(it.get("id", "")))
        tr = it.get("trace", {}) or {}
        rounds = tr.get("rounds", []) or []
        if rounds:
            stop_pool = int(rounds[-1].get("pool_size", 0) or 0)
            stop_pools.append(stop_pool)
            stopped_early = bool(rounds[-1].get("early_stop", False))
            early_stop_flags.append(stopped_early)
            early_stop_saved_budget_flags.append(bool(stopped_early and stop_pool < int(n_total)))
        else:
            stop_pools.append(0)
            early_stop_flags.append(False)
            early_stop_saved_budget_flags.append(False)

        e = int(tr.get("tokens_effective_from_scratch", 0) or 0) + int(
            tr.get("tokens_effective_continue", 0) or 0
        )
        a = int(tr.get("tokens_actual_from_scratch", 0) or 0) + int(
            tr.get("tokens_actual_continue", 0) or 0
        )
        eff_tokens.append(e)
        act_tokens.append(a)

    # Early-stop rate is defined as "stopped before reaching n_total" (i.e., saved sampling budget).
    early_stop_rate = (
        float(sum(1 for x in early_stop_saved_budget_flags if x) / n) if n else float("nan")
    )
    median_stop_pool = float(statistics.median(stop_pools)) if stop_pools else float("nan")

    summary: Dict[str, Any] = {
        "n_items": n,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "early_stop_rate": early_stop_rate,
        "median_stop_pool": median_stop_pool,
        "n_total": int(n_total),
        "effective_tokens": {
            "total": int(sum(eff_tokens)),
            "mean": _safe_mean([float(x) for x in eff_tokens]),
            "median": _safe_median([float(x) for x in eff_tokens]),
        },
        "actual_tokens": {
            "total": int(sum(act_tokens)),
            "mean": _safe_mean([float(x) for x in act_tokens]),
            "median": _safe_median([float(x) for x in act_tokens]),
        },
    }

    if baseline_a_json:
        base_tokenizer = _load_tokenizer_for_baseline(baseline_tokenizer_path) or prm_tokenizer
        baseline = _compute_baseline_a_effective_tokens(
            baseline_json_path=baseline_a_json, tokenizer=base_tokenizer, item_ids=item_ids
        )
        summary["baseline_a"] = baseline
        base_total = float(baseline.get("effective_tokens_total", 0) or 0)
        aca_total = float(summary["effective_tokens"]["total"])
        summary["effective_saving_vs_baseline_a"] = (
            float("nan") if base_total <= 0 else float(1.0 - (aca_total / base_total))
        )

    out_dir = os.path.dirname(os.path.abspath(output_json_path))
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    _write_json(summary_path, summary)

    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)
        _write_json(os.path.join(save_intermediate_dir, "summary.json"), summary)

    print(
        f"[summary] items={n} accuracy={accuracy:.4f} ({num_correct}/{n}) "
        f"early_stop_rate={early_stop_rate:.4f} median_stop_pool={int(median_stop_pool)}"
    )
    et = summary["effective_tokens"]
    print(
        f"[summary] effective_tokens total={int(et['total'])} mean={et['mean']:.1f} median={et['median']:.1f}"
    )
    if "effective_saving_vs_baseline_a" in summary:
        saving = summary["effective_saving_vs_baseline_a"]
        base_total = summary.get("baseline_a", {}).get("effective_tokens_total", 0)
        print(
            f"[summary] baseline_a_effective_tokens total≈{int(base_total)} saving≈{saving:.4f}"
        )
    print(f"[summary] saved: {summary_path}")

    return summary


def build_prm_prompt(question: str, segments: Sequence[str]) -> Tuple[str, int]:
    # segments includes answer at the end
    steps = [str(s).strip() for s in segments if str(s).strip()]
    solution = "<prm>".join(steps) + "<prm>" if steps else ""
    prompt = f"Question: {question}\nProcess: {solution}"
    return prompt, len(steps)


def load_pixel_values(
    image_path: Optional[Union[str, Sequence[str]]],
    input_size: int,
    dynamic: bool,
    use_thumbnail: bool,
    max_num: int,
) -> Tuple[torch.Tensor, List[int]]:
    transform = build_transform(is_train=False, input_size=input_size)
    if not image_path:
        # Create dummy image for text-only prompts.
        image = Image.new("RGB", (224, 224), (255, 255, 255))
        images = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
            max_num=1,
        )
        num_patches_list = [len(images)]
    else:
        raw_paths = (
            [str(p) for p in image_path]
            if isinstance(image_path, Sequence) and not isinstance(image_path, (str, bytes))
            else [str(image_path)]
        )
        images = []
        num_patches_list = []
        for path in raw_paths:
            image = Image.open(path).convert("RGB")
            if dynamic:
                curr_images = dynamic_preprocess(
                    image,
                    image_size=input_size,
                    use_thumbnail=use_thumbnail,
                    max_num=max_num,
                )
            else:
                curr_images = [image]
            images.extend(curr_images)
            num_patches_list.append(len(curr_images))
    pixel_values = [transform(im) for im in images]
    return torch.stack(pixel_values), num_patches_list


def score_and_annotate_candidates(
    model,
    tokenizer,
    pixel_values: Optional[torch.Tensor],
    num_patches_list: Optional[Sequence[int]],
    resolved_image_paths: Sequence[str],
    question: str,
    candidates: List[Candidate],
    risk_lambda: float,
    c_stop: float,
    mini_batch_size: int,
) -> None:
    prompts: List[str] = []
    lens: List[int] = []
    for c in candidates:
        p, ln = build_prm_prompt(question, c.segments)
        prompts.append(p)
        lens.append(ln)

    mu_all: List[float] = []
    kappa_all: List[float] = []
    sigma_all: List[float] = []

    for i in range(0, len(prompts), mini_batch_size):
        curr = prompts[i : i + mini_batch_size]
        curr_bs = len(curr)
        if getattr(model, "backend", "internvl") == "internvl":
            if pixel_values is None or num_patches_list is None:
                raise ValueError("InternVL PRM scoring requires pixel_values and num_patches_list.")
            curr_pixel_values = torch.cat([pixel_values] * curr_bs, dim=0)
            curr_num_patches_list = [list(num_patches_list)] * curr_bs
        else:
            curr_pixel_values = None
            curr_num_patches_list = None
        mu, kappa, sigma = batch_prm_mu_kappa_sigma(
            model=model,
            tokenizer=tokenizer,
            pixel_values=curr_pixel_values,
            prompts=curr,
            num_patches_list=curr_num_patches_list,
            image_paths=list(resolved_image_paths),
        )
        mu_all.extend(mu.tolist())
        kappa_all.extend(kappa.tolist())
        sigma_all.extend(sigma.tolist())

    # Slice back per candidate
    offset = 0
    for c, ln in zip(candidates, lens):
        c.prm_mu = mu_all[offset : offset + ln]
        c.prm_kappa = kappa_all[offset : offset + ln]
        c.prm_sigma = sigma_all[offset : offset + ln]
        c.prm_score_steps = score_steps_mu_minus_lambda_sigma(
            c.prm_mu, c.prm_sigma, risk_lambda=risk_lambda
        )
        c.score = mean(c.prm_score_steps) if c.prm_score_steps else float("-inf")

        # Uncertainty aggregate excludes the answer step (last segment)
        if ln >= 2:
            c.u_sigma = mean(c.prm_sigma[: ln - 1])
        else:
            c.u_sigma = mean(c.prm_sigma)
        c.lcb = c.score - float(c_stop) * float(c.u_sigma)
        c.ucb = c.score + float(c_stop) * float(c.u_sigma)

        offset += ln


def pick_best_idx(cands: List[Candidate]) -> int:
    if not cands:
        return -1
    return int(max(range(len(cands)), key=lambda i: float(cands[i].score)))


def maybe_early_stop(cands: List[Candidate], best_idx: int) -> bool:
    if best_idx < 0 or best_idx >= len(cands):
        return False
    best_lcb = float(cands[best_idx].lcb)
    max_other_ucb = float("-inf")
    for i, c in enumerate(cands):
        if i == best_idx:
            continue
        max_other_ucb = max(max_other_ucb, float(c.ucb))
    return best_lcb > max_other_ucb


def select_competitor_idx(
    cands: List[Candidate],
    best_idx: int,
    expand_policy: str = "ucb_runnerup",
) -> int:
    if not cands or best_idx < 0 or best_idx >= len(cands):
        return -1
    if expand_policy == "ucb_top1":
        return int(max(range(len(cands)), key=lambda i: float(cands[i].ucb)))
    if expand_policy != "ucb_runnerup":
        raise ValueError(f"Unsupported expand_policy: {expand_policy}")
    best_lcb = float(cands[best_idx].lcb)
    comp = -1
    best_ucb = float("-inf")
    for i, c in enumerate(cands):
        if i == best_idx:
            continue
        if float(c.ucb) >= best_lcb and float(c.ucb) > best_ucb:
            best_ucb = float(c.ucb)
            comp = int(i)
    return comp


def run_one_item(
    model,
    tokenizer,
    gen_url: str,
    judge_cfg: Optional[JudgeConfig],
    item: Dict[str, Any],
    item_idx: int,
    image_root: str,
    input_base_dir: str,
    sampling: Dict[str, Any],
    n0: int,
    n_total: int,
    m: int,
    oversample: float,
    risk_lambda: float,
    c_stop: float,
    expand_policy: str,
    disable_early_stop: bool,
    cut_cfg: CutConfig,
    mini_batch_size: int,
    pixel_values_cfg: Dict[str, Any],
    save_intermediate_dir: str,
    save_raw_oversample: bool,
) -> Dict[str, Any]:
    normalized = _normalize_item_inputs(
        item,
        input_base_dir=input_base_dir,
        image_root=image_root,
    )
    question = str(normalized["question_raw"])
    question_for_model = str(normalized["question_model"])
    resolved_image_input = normalized["resolved_image_input"]
    resolved_image_paths = list(normalized["resolved_image_paths"])
    image_path = resolved_image_paths[0] if len(resolved_image_paths) == 1 else ""
    correct_answer = normalized["correct_answer_text"]

    item_id = item.get("id", None)
    item_tag = str(item_id) if item_id is not None else f"idx_{int(item_idx)}"
    save_dir_item = (
        os.path.join(save_intermediate_dir, item_tag) if save_intermediate_dir else ""
    )
    if save_dir_item:
        _write_json(
            os.path.join(save_dir_item, "config.json"),
            {
                "id": item_id,
                "item_idx": int(item_idx),
                "question": question,
                "image_path": image_path,
                "image_paths": resolved_image_paths if len(resolved_image_paths) > 1 else None,
                "correct_answer": correct_answer,
                "n0": int(n0),
                "n_total": int(n_total),
                "m": int(m),
                "oversample": float(oversample),
                "risk_lambda": float(risk_lambda),
                "c_stop": float(c_stop),
                "cut_cfg": {
                    "p_bad": float(cut_cfg.p_bad),
                    "c_cut": float(cut_cfg.c_cut),
                    "min_step_chars": int(cut_cfg.min_step_chars),
                    "gibberish_min_len": int(cut_cfg.gibberish_min_len),
                    "gibberish_non_ascii_letter_ratio": float(
                        cut_cfg.gibberish_non_ascii_letter_ratio
                    ),
                    "gibberish_min_non_ascii_letters": int(
                        cut_cfg.gibberish_min_non_ascii_letters
                    ),
                },
                "sampling": dict(sampling),
                "save_raw_oversample": bool(save_raw_oversample),
                "ts": float(time.time()),
            },
        )

    if getattr(model, "backend", "internvl") == "internvl":
        pixel_values, num_patches_list = load_pixel_values(
            resolved_image_input,
            **pixel_values_cfg,
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    else:
        pixel_values = None
        num_patches_list = None

    trace: Dict[str, Any] = {
        "rounds": [],
        "tokens_effective_from_scratch": 0,
        "tokens_effective_continue": 0,
        "tokens_actual_from_scratch": 0,
        "tokens_actual_continue": 0,
    }

    # Round 0: from scratch
    pool, gen_meta = gen_from_scratch(
        gen_url=gen_url,
        question=question_for_model,
        image_path=resolved_image_input,
        k=int(n0),
        oversample_factor=float(oversample),
        sampling=sampling,
        return_raw=bool(save_dir_item and save_raw_oversample),
    )
    if save_dir_item:
        _append_jsonl(
            os.path.join(save_dir_item, "gen_calls.jsonl"),
            {
                "kind": "scratch",
                "k": int(n0),
                "oversample_factor": float(oversample),
                "sampling": dict(sampling),
                "meta": {
                    "raw_count": gen_meta.get("raw_count", None),
                    "kept_count": gen_meta.get("kept_count", None),
                    "effective_completion_tokens": gen_meta.get("effective_completion_tokens", None),
                    "actual_completion_tokens": gen_meta.get("actual_completion_tokens", None),
                },
                "raw": gen_meta.get("raw", None) if save_raw_oversample else None,
                "kept_indices": gen_meta.get("kept_indices", None) if save_raw_oversample else None,
                "user_prompt": gen_meta.get("user_prompt", None) if save_raw_oversample else None,
                "ts": float(time.time()),
            },
        )
    for c in pool:
        c.round_added = 0
    trace["tokens_effective_from_scratch"] += int(gen_meta.get("effective_completion_tokens", 0))
    trace["tokens_actual_from_scratch"] += int(gen_meta.get("actual_completion_tokens", 0))

    # Main loop
    logged_pool_size = 0
    while True:
        score_and_annotate_candidates(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            resolved_image_paths=resolved_image_paths,
            question=question_for_model,
            candidates=pool,
            risk_lambda=risk_lambda,
            c_stop=c_stop,
            mini_batch_size=mini_batch_size,
        )
        best_idx = pick_best_idx(pool)
        stop = (
            (maybe_early_stop(pool, best_idx) if len(pool) >= int(n0) else False)
            if not disable_early_stop
            else False
        )

        if save_dir_item and logged_pool_size < len(pool):
            for ci in range(logged_pool_size, len(pool)):
                c = pool[ci]
                _append_jsonl(
                    os.path.join(save_dir_item, "candidates.jsonl"),
                    {
                        "pool_idx": int(ci),
                        "origin": str(c.origin),
                        "round_added": int(c.round_added),
                        "segments": c.segments,
                        "token_len": int(c.token_len),
                        "prm_mu": c.prm_mu,
                        "prm_kappa": c.prm_kappa,
                        "prm_sigma": c.prm_sigma,
                        "prm_scores": c.prm_score_steps,
                        "score": float(c.score),
                        "u_sigma": float(c.u_sigma),
                        "lcb": float(c.lcb),
                        "ucb": float(c.ucb),
                    },
                )
            logged_pool_size = len(pool)

        round_info: Dict[str, Any] = {
            "pool_size": len(pool),
            "best_idx": int(best_idx),
            "best_score": float(pool[best_idx].score) if best_idx >= 0 else None,
            "best_lcb": float(pool[best_idx].lcb) if best_idx >= 0 else None,
            "best_ucb": float(pool[best_idx].ucb) if best_idx >= 0 else None,
            "early_stop": bool(stop),
        }

        if (not disable_early_stop and stop) or len(pool) >= int(n_total):
            trace["rounds"].append(round_info)
            if save_dir_item:
                _append_jsonl(os.path.join(save_dir_item, "rounds.jsonl"), round_info)
            break

        comp_idx = select_competitor_idx(pool, best_idx, expand_policy=expand_policy)
        if comp_idx < 0:
            if not disable_early_stop:
                # No plausible competitor -> stop.
                round_info["early_stop"] = True
                trace["rounds"].append(round_info)
                if save_dir_item:
                    _append_jsonl(os.path.join(save_dir_item, "rounds.jsonl"), round_info)
                break
            # Early-stop disabled: keep allocating budget by sampling from scratch.
            comp_idx = -1

        if comp_idx >= 0:
            comp = pool[comp_idx]
            t = pick_cutpoint(
                steps=comp.segments,
                mu=comp.prm_mu,
                sigma=comp.prm_sigma,
                cfg=cut_cfg,
            )
            prefix = comp.reasoning_steps[:t]
            round_info.update(
                {
                    "competitor_idx": int(comp_idx),
                    "competitor_score": float(comp.score),
                    "cut_t": int(t),
                    "prefix_len": int(len(prefix)),
                }
            )
        else:
            prefix = []
            round_info.update(
                {
                    "competitor_idx": -1,
                    "competitor_score": None,
                    "cut_t": 0,
                    "prefix_len": 0,
                }
            )

        remaining = int(n_total) - len(pool)
        add_m = min(int(m), max(0, remaining))
        if add_m <= 0:
            round_info["early_stop"] = True
            trace["rounds"].append(round_info)
            if save_dir_item:
                _append_jsonl(os.path.join(save_dir_item, "rounds.jsonl"), round_info)
            break

        # If prefix is empty, continuing is equivalent to scratch; use scratch endpoint for stability.
        if len(prefix) == 0:
            new_cands, gen2 = gen_from_scratch(
                gen_url=gen_url,
                question=question_for_model,
                image_path=resolved_image_input,
                k=int(add_m),
                oversample_factor=float(oversample),
                sampling=sampling,
                return_raw=bool(save_dir_item and save_raw_oversample),
            )
            for c in new_cands:
                c.origin = "scratch_fallback"
            trace["tokens_effective_from_scratch"] += int(gen2.get("effective_completion_tokens", 0))
            trace["tokens_actual_from_scratch"] += int(gen2.get("actual_completion_tokens", 0))
            merged_new = new_cands
            gen_kind = "scratch_fallback"
        else:
            new_cands, gen2 = gen_continue(
                gen_url=gen_url,
                question=question_for_model,
                image_path=resolved_image_input,
                prefix_steps=prefix,
                m=int(add_m),
                oversample_factor=float(oversample),
                sampling=sampling,
                return_raw=bool(save_dir_item and save_raw_oversample),
            )
            trace["tokens_effective_continue"] += int(gen2.get("effective_completion_tokens", 0))
            trace["tokens_actual_continue"] += int(gen2.get("actual_completion_tokens", 0))

            merged_new: List[Candidate] = []
            for nc in new_cands:
                merged = merge_prefix_continuation(prefix_steps=prefix, cont_segments=nc.segments)
                if len(merged) < 2:
                    continue
                merged_new.append(
                    Candidate(
                        segments=merged,
                        token_len=int(nc.token_len),
                        origin="continue",
                    )
                )
            gen_kind = "continue"

        if save_dir_item:
            _append_jsonl(
                os.path.join(save_dir_item, "gen_calls.jsonl"),
                {
                    "kind": str(gen_kind),
                    "add_m": int(add_m),
                    "oversample_factor": float(oversample),
                    "prefix_len": int(len(prefix)),
                    "sampling": dict(sampling),
                    "meta": {
                        "raw_count": gen2.get("raw_count", None),
                        "kept_count": gen2.get("kept_count", None),
                        "effective_completion_tokens": gen2.get("effective_completion_tokens", None),
                        "actual_completion_tokens": gen2.get("actual_completion_tokens", None),
                    },
                    "raw": gen2.get("raw", None) if save_raw_oversample else None,
                    "kept_indices": gen2.get("kept_indices", None) if save_raw_oversample else None,
                    "user_prompt": gen2.get("user_prompt", None) if save_raw_oversample else None,
                    "ts": float(time.time()),
                },
            )

        # No-progress guard: if continuation yields no valid candidates, fall back to scratch once.
        if len(merged_new) == 0 and gen_kind == "continue":
            fallback_cands, gen3 = gen_from_scratch(
                gen_url=gen_url,
                question=question_for_model,
                image_path=resolved_image_input,
                k=int(add_m),
                oversample_factor=float(oversample),
                sampling=sampling,
                return_raw=bool(save_dir_item and save_raw_oversample),
            )
            for c in fallback_cands:
                c.origin = "continue_fallback_scratch"
            trace["tokens_effective_from_scratch"] += int(
                gen3.get("effective_completion_tokens", 0)
            )
            trace["tokens_actual_from_scratch"] += int(gen3.get("actual_completion_tokens", 0))
            merged_new = fallback_cands
            gen_kind = "continue_fallback_scratch"
            if save_dir_item:
                _append_jsonl(
                    os.path.join(save_dir_item, "gen_calls.jsonl"),
                    {
                        "kind": "continue_fallback_scratch",
                        "add_m": int(add_m),
                        "oversample_factor": float(oversample),
                        "prefix_len": int(len(prefix)),
                        "sampling": dict(sampling),
                        "meta": {
                            "raw_count": gen3.get("raw_count", None),
                            "kept_count": gen3.get("kept_count", None),
                            "effective_completion_tokens": gen3.get("effective_completion_tokens", None),
                            "actual_completion_tokens": gen3.get("actual_completion_tokens", None),
                        },
                        "raw": gen3.get("raw", None) if save_raw_oversample else None,
                        "kept_indices": gen3.get("kept_indices", None) if save_raw_oversample else None,
                        "user_prompt": gen3.get("user_prompt", None) if save_raw_oversample else None,
                        "ts": float(time.time()),
                    },
                )

        if len(merged_new) == 0:
            # Still no progress -> exit to avoid dead loop.
            round_info["early_stop"] = True
            round_info["no_progress"] = True
            trace["rounds"].append(round_info)
            if save_dir_item:
                _append_jsonl(os.path.join(save_dir_item, "rounds.jsonl"), round_info)
            break

        round_info.update(
            {
                "gen_kind": gen_kind,
                "added": int(len(merged_new)),
            }
        )
        trace["rounds"].append(round_info)
        if save_dir_item:
            _append_jsonl(os.path.join(save_dir_item, "rounds.jsonl"), round_info)

        current_round = len(trace["rounds"]) - 1
        for c in merged_new:
            c.round_added = int(current_round)
        pool.extend(merged_new)
        # Loop continues

    # Final selection
    score_and_annotate_candidates(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        num_patches_list=num_patches_list,
        resolved_image_paths=resolved_image_paths,
        question=question_for_model,
        candidates=pool,
        risk_lambda=risk_lambda,
        c_stop=c_stop,
        mini_batch_size=mini_batch_size,
    )
    best_idx = pick_best_idx(pool)

    out_item: Dict[str, Any] = {
        "id": item.get("id", None),
        "question": question,
        "image_path": image_path,
        "image_paths": resolved_image_paths if len(resolved_image_paths) > 1 else None,
        "selected_best_idx": int(best_idx),
        "candidates": [],
        "trace": trace,
        "algorithm_trace": trace,
    }

    # Optional judge for kept candidates only (effective accounting).
    labels: Optional[List[int]] = None
    if judge_cfg is not None and correct_answer is not None:
        labels = []
        for c in pool:
            labels.append(
                int(
                    judge_yes_no(
                        cfg=judge_cfg,
                        question=question,
                        correct_answer=str(correct_answer),
                        model_answer=str(c.answer),
                    )
                )
            )
        out_item["labels"] = labels
        out_item["correct_answer"] = correct_answer
        out_item["is_correct_best"] = int(labels[best_idx]) if best_idx >= 0 else 0

    for ci, c in enumerate(pool):
        out_item["candidates"].append(
            {
                "idx": int(ci),
                "segments": c.segments,
                "answer": c.answer,
                "token_len": int(c.token_len),
                "origin": str(c.origin),
                "round_added": int(c.round_added),
                "prm_mu": c.prm_mu,
                "prm_kappa": c.prm_kappa,
                "prm_sigma": c.prm_sigma,
                "prm_scores": c.prm_score_steps,
                "score": float(c.score),
                "u_sigma": float(c.u_sigma),
                "lcb": float(c.lcb),
                "ucb": float(c.ucb),
                "label": int(labels[ci]) if labels is not None else None,
            }
        )
    if save_dir_item:
        _write_json(os.path.join(save_dir_item, "final.json"), out_item)
    return out_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON list")
    parser.add_argument("--output", required=True, help="Output JSON list")
    parser.add_argument("--prm-ckpt", required=True, help="BetaPRM V1 checkpoint")
    parser.add_argument("--prm-backend", default="internvl", choices=["internvl", "qwenvl_beta_binom_v1"])
    parser.add_argument("--prm-base-model", default="", help="Base QwenVL model path, required for Qwen PRM backend.")
    parser.add_argument("--dataset-name", default="", help="Dataset name for PRM backend policies, e.g. MathVerse or OlympiadBench.")
    parser.add_argument("--gen-url", required=True, help="Gen service base url, e.g. http://127.0.0.1:18080")
    parser.add_argument("--image-root", default="", help="Optional root dir for resolving relative image_path")
    parser.add_argument("--judge-url", default="", help="Optional judge OpenAI-compatible base url, e.g. http://127.0.0.1:8888/v1")
    parser.add_argument("--judge-model", default="", help="Judge served model name")
    parser.add_argument("--n0", type=int, default=4)
    parser.add_argument("--n-total", type=int, default=16)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--oversample", type=float, default=2.0)
    parser.add_argument("--mini-batch-size", type=int, default=4)

    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--c-stop", type=float, default=0.3)
    parser.add_argument(
        "--expand-policy",
        choices=["ucb_runnerup", "ucb_top1"],
        default="ucb_runnerup",
        help="How to choose the path to expand: current behavior uses the best non-best UCB challenger; ucb_top1 expands the global top-UCB path and may overlap with the score-best path.",
    )
    parser.add_argument("--disable-early-stop", action="store_true", default=False, help="If set, never stop early; keep sampling until n_total (unless no_progress).")
    parser.add_argument("--c-cut", type=float, default=2.0)
    parser.add_argument("--p-bad", type=float, default=0.5)

    # Gen sampling params (passed to gen service)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=-1, help="Optional seed for generator (passed to gen service). Use -1 to disable.")
    parser.add_argument("--save-intermediate-dir", default="", help="If set, save per-item intermediate logs under this directory.")
    parser.add_argument("--save-raw-oversample", action="store_true", default=False, help="If set with --save-intermediate-dir, also save oversampled raw gen outputs.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume by skipping items with valid per-item final.json under --save-intermediate-dir.")
    parser.add_argument("--baseline-a-json", default="", help="Optional Baseline-A rollout json to estimate baseline effective tokens offline.")
    parser.add_argument("--baseline-tokenizer", default="", help="Optional tokenizer path for Baseline-A token counting (defaults to PRM tokenizer).")

    # Image preprocess for PRM
    parser.set_defaults(dynamic=True)
    dyn = parser.add_mutually_exclusive_group()
    dyn.add_argument("--dynamic", dest="dynamic", action="store_true", help="Enable dynamic image preprocessing (default).")
    dyn.add_argument("--no-dynamic", dest="dynamic", action="store_false", help="Disable dynamic image preprocessing.")
    parser.add_argument("--input-size", type=int, default=0, help="PRM image input size. Use 0 to follow the PRM checkpoint config (recommended).")
    parser.set_defaults(use_thumbnail=None)
    thumb = parser.add_mutually_exclusive_group()
    thumb.add_argument("--use-thumbnail", dest="use_thumbnail", action="store_true", help="Force enable thumbnail in PRM preprocessing.")
    thumb.add_argument("--no-thumbnail", dest="use_thumbnail", action="store_false", help="Force disable thumbnail in PRM preprocessing.")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--prm-processor-use-fast", action="store_true", default=False)
    parser.add_argument("--prm-max-seq-length", type=int, default=32768)
    parser.add_argument("--prm-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--prm-max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--prm-video-min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--prm-video-max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--prm-video-min-frames", type=int, default=4)
    parser.add_argument("--prm-video-max-frames", type=int, default=128)
    parser.add_argument("--prm-video-fps", type=float, default=2.0)
    parser.add_argument("--prm-grid-max-cols", type=int, default=3)
    parser.add_argument("--prm-load-weights-to-gpu", action="store_true", default=False)
    parser.set_defaults(prm_bf16=None)
    prm_bf16 = parser.add_mutually_exclusive_group()
    prm_bf16.add_argument("--prm-bf16", dest="prm_bf16", action="store_true")
    prm_bf16.add_argument("--no-prm-bf16", dest="prm_bf16", action="store_false")

    args = parser.parse_args()
    if args.resume and not args.save_intermediate_dir:
        parser.error("--resume requires --save-intermediate-dir")

    data = json.load(open(args.input, "r", encoding="utf-8"))
    assert isinstance(data, list), "Input must be a JSON list"
    input_base_dir = os.path.dirname(os.path.abspath(args.input))

    prm_cfg = PRMConfig(
        checkpoint=args.prm_ckpt,
        backend=str(args.prm_backend),
        base_model_path=str(args.prm_base_model),
        dataset_name=str(args.dataset_name),
        processor_use_fast=bool(args.prm_processor_use_fast),
        max_seq_length=int(args.prm_max_seq_length),
        min_pixels=int(args.prm_min_pixels),
        max_pixels=int(args.prm_max_pixels),
        video_min_pixels=int(args.prm_video_min_pixels),
        video_max_pixels=int(args.prm_video_max_pixels),
        video_min_frames=int(args.prm_video_min_frames),
        video_max_frames=int(args.prm_video_max_frames),
        video_fps=float(args.prm_video_fps),
        grid_max_cols=int(args.prm_grid_max_cols),
        load_weights_to_gpu=bool(args.prm_load_weights_to_gpu),
        bf16=True if args.prm_bf16 is None else bool(args.prm_bf16),
    )
    model, tokenizer = load_prm_model(prm_cfg)

    if str(args.prm_backend) == "internvl":
        cfg_image_size = getattr(model.config, "force_image_size", None) or getattr(
            getattr(model.config, "vision_config", None), "image_size", None
        )
        cfg_use_thumbnail = bool(getattr(model.config, "use_thumbnail", False))
        eff_input_size = int(args.input_size) if int(args.input_size) > 0 else int(cfg_image_size or 224)
        eff_use_thumbnail = (
            bool(args.use_thumbnail) if args.use_thumbnail is not None else bool(cfg_use_thumbnail)
        )
        print(
            f"[prm] backend=internvl cfg_image_size={cfg_image_size} cfg_use_thumbnail={cfg_use_thumbnail} "
            f"-> eff_input_size={eff_input_size} eff_use_thumbnail={eff_use_thumbnail} dynamic={bool(args.dynamic)} max_num={int(args.max_num)}"
        )
    else:
        cfg_image_size = None
        cfg_use_thumbnail = None
        eff_input_size = 0
        eff_use_thumbnail = False
        print(
            f"[prm] backend={args.prm_backend} base_model={args.prm_base_model} dataset_name={args.dataset_name} "
            f"processor_use_fast={bool(args.prm_processor_use_fast)} max_seq_length={int(args.prm_max_seq_length)} "
            f"min_pixels={int(args.prm_min_pixels)} max_pixels={int(args.prm_max_pixels)} "
            f"grid_max_cols={int(args.prm_grid_max_cols)} "
            f"load_weights_to_gpu={bool(args.prm_load_weights_to_gpu)} "
            f"bf16={True if args.prm_bf16 is None else bool(args.prm_bf16)}"
        )

    judge_cfg: Optional[JudgeConfig] = None
    if args.judge_url and args.judge_model:
        judge_cfg = JudgeConfig(api_base=args.judge_url, model_name=args.judge_model)

    sampling = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
    }
    if int(args.seed) >= 0:
        sampling["seed"] = int(args.seed)

    cut_cfg = CutConfig(c_cut=float(args.c_cut), p_bad=float(args.p_bad))
    pv_cfg = {
        "input_size": int(eff_input_size),
        "dynamic": bool(args.dynamic),
        "use_thumbnail": bool(eff_use_thumbnail),
        "max_num": int(args.max_num),
    }

    if args.save_intermediate_dir:
        os.makedirs(args.save_intermediate_dir, exist_ok=True)
        _write_json(
            os.path.join(args.save_intermediate_dir, "run_config.json"),
            {
                "input": args.input,
                "output": args.output,
                "gen_url": args.gen_url,
                "prm_ckpt": args.prm_ckpt,
                "prm_backend": args.prm_backend,
                "prm_base_model": args.prm_base_model,
                "dataset_name": args.dataset_name,
                "judge_url": args.judge_url,
                "judge_model": args.judge_model,
                "image_root": args.image_root,
                "n0": int(args.n0),
                "n_total": int(args.n_total),
                "m": int(args.m),
                "oversample": float(args.oversample),
                "mini_batch_size": int(args.mini_batch_size),
                "risk_lambda": float(args.risk_lambda),
                "c_stop": float(args.c_stop),
                "expand_policy": str(args.expand_policy),
                "c_cut": float(args.c_cut),
                "p_bad": float(args.p_bad),
                "sampling": dict(sampling),
                "save_raw_oversample": bool(args.save_raw_oversample),
                "resume": bool(args.resume),
                "prm_image_cfg": {
                    "cfg_image_size": cfg_image_size,
                    "cfg_use_thumbnail": cfg_use_thumbnail,
                    "eff_input_size": int(eff_input_size),
                    "eff_use_thumbnail": bool(eff_use_thumbnail),
                    "dynamic": bool(args.dynamic),
                    "max_num": int(args.max_num),
                    "processor_use_fast": bool(args.prm_processor_use_fast),
                    "max_seq_length": int(args.prm_max_seq_length),
                    "min_pixels": int(args.prm_min_pixels),
                    "max_pixels": int(args.prm_max_pixels),
                    "video_min_pixels": int(args.prm_video_min_pixels),
                    "video_max_pixels": int(args.prm_video_max_pixels),
                    "video_min_frames": int(args.prm_video_min_frames),
                    "video_max_frames": int(args.prm_video_max_frames),
                    "video_fps": float(args.prm_video_fps),
                    "grid_max_cols": int(args.prm_grid_max_cols),
                    "load_weights_to_gpu": bool(args.prm_load_weights_to_gpu),
                    "bf16": True if args.prm_bf16 is None else bool(args.prm_bf16),
                },
                "ts": float(time.time()),
            },
        )

    completed_outputs: Dict[str, Dict[str, Any]] = {}
    invalid_resume_tags = set()
    if args.resume:
        completed_outputs, invalid_resume_tags = load_completed_outputs(
            data=data,
            save_intermediate_dir=str(args.save_intermediate_dir),
        )
        print(
            f"[resume] enabled save_dir={args.save_intermediate_dir} "
            f"valid_completed={len(completed_outputs)} invalid_cached={len(invalid_resume_tags)}"
        )

    outputs: List[Dict[str, Any]] = []
    resumed_count = 0
    new_count = 0
    for item_idx, item in enumerate(tqdm(data, desc="ACA-orchestrator")):
        item_tag = make_item_tag(item.get("id", None), int(item_idx))
        resumed = completed_outputs.get(item_tag)
        if resumed is not None:
            print(f"[resume] hit item={item_tag} -> reuse saved final.json")
            outputs.append(resumed)
            resumed_count += 1
            continue
        if args.resume and item_tag in invalid_resume_tags:
            print(f"[resume] invalid cache item={item_tag} -> rerun")
        elif args.resume:
            print(f"[resume] new run item={item_tag}")
        outputs.append(
            run_one_item(
                model=model,
                tokenizer=tokenizer,
                gen_url=args.gen_url,
                judge_cfg=judge_cfg,
                item=item,
                item_idx=int(item_idx),
                image_root=str(args.image_root),
                input_base_dir=input_base_dir,
                sampling=sampling,
                n0=int(args.n0),
                n_total=int(args.n_total),
                m=int(args.m),
                oversample=float(args.oversample),
                risk_lambda=float(args.risk_lambda),
                c_stop=float(args.c_stop),
                expand_policy=str(args.expand_policy),
                disable_early_stop=bool(args.disable_early_stop),
                cut_cfg=cut_cfg,
                mini_batch_size=int(args.mini_batch_size),
                pixel_values_cfg=pv_cfg,
                save_intermediate_dir=str(args.save_intermediate_dir),
                save_raw_oversample=bool(args.save_raw_oversample),
            )
        )
        new_count += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    if args.resume:
        print(f"[resume] summary reused={resumed_count} newly_run={new_count} total={len(outputs)}")

    _compute_and_save_summary(
        outputs=outputs,
        output_json_path=str(args.output),
        save_intermediate_dir=str(args.save_intermediate_dir),
        baseline_a_json=str(args.baseline_a_json),
        baseline_tokenizer_path=str(args.baseline_tokenizer),
        prm_tokenizer=tokenizer,
        n_total=int(args.n_total),
    )


if __name__ == "__main__":
    main()
