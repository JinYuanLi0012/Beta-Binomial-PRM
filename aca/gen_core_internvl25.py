from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def select_best_rollouts(seg_lists: Sequence[Sequence[str]], k: int) -> List[List[str]]:
    seen = set()
    deduped: List[List[str]] = []
    for segs in seg_lists:
        segs_norm = tuple(_normalize_text(x) for x in segs if _normalize_text(x))
        if not segs_norm:
            continue
        if segs_norm in seen:
            continue
        seen.add(segs_norm)
        deduped.append(list(segs_norm))

    valid = [s for s in deduped if len(s) >= 2 and all(x.strip() for x in s)]
    valid.sort(key=lambda s: (len(s), sum(len(x) for x in s)), reverse=True)
    result = valid[:k]
    if len(result) < k:
        rest = [
            s
            for s in deduped
            if len(s) >= 1 and all(x.strip() for x in s) and s not in result
        ]
        rest.sort(key=lambda s: (len(s), sum(len(x) for x in s)), reverse=True)
        result.extend(rest[: k - len(result)])
    return result


def make_cot_prompt(question: str) -> str:
    return f"""<image>
Please solve the problem and output STRICTLY and ONLY using these XML tags:
- Wrap each reasoning step in <step>...</step>.
- Put the final answer in ONE <answer>...</answer> tag as the LAST tag.
Do NOT output any text outside these tags. No extra text after </answer>.

Question: {question}"""


def make_continue_prompt(
    question: str,
    prefix_steps: Sequence[str],
    extra_instruction: Optional[str] = None,
) -> str:
    prev = "".join([f"<step>{_normalize_text(s)}</step>" for s in prefix_steps if _normalize_text(s)])
    extra = ""
    if extra_instruction and _normalize_text(extra_instruction):
        extra = _normalize_text(extra_instruction) + "\n"
    return f"""<image>
You are given a math problem. Below are the previous reasoning steps already finished.
Continue from the NEXT step (do NOT repeat any previous steps).
{extra}Output STRICTLY and ONLY using these XML tags:
- Use <step>...</step> for each NEW step you add.
- End with exactly ONE <answer>...</answer> tag as the LAST tag.
Do NOT output any text outside these tags. No extra text after </answer>.

Question: {question}

Previous Steps:
{prev}"""


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenResult:
    segments: List[str]
    token_len: int
    raw_text: str = ""


class InternVL25Generator:
    """
    vLLM Python generator for InternVL2.5 with robust XML segment parsing and per-sample token length.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for generator. Install vllm + transformers to run gen_server."
            ) from e

        self.model_path = model
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 8}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self._step_answer_pattern = re.compile(
            r"<step>(.*?)</step>|<answer>(.*?)</answer>", re.DOTALL
        )
        self._special_tokens = [
            t for t in self.tokenizer.all_special_tokens if t not in ["<step>", "</step>", "<answer>", "</answer>"]
        ]
        self._special_re = re.compile("|".join(map(re.escape, self._special_tokens))) if self._special_tokens else None

        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
        )

    def _to_chat_prompt(self, user_content: str, image_count: int) -> str:
        prompt_with_images = user_content
        if image_count > 0:
            existing = prompt_with_images.count("<image>")
            need = max(0, image_count - existing)
            if need > 0:
                prompt_with_images = (" ".join(["<image>"] * need) + "\n") + prompt_with_images
        elif "<image>" not in prompt_with_images:
            prompt_with_images = "<image>\n" + prompt_with_images

        messages = [
            {
                "role": "system",
                "content": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
            },
            {"role": "user", "content": prompt_with_images},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _load_images(image_path: Optional[Union[str, List[str]]]) -> Tuple[Optional[Union[Any, List[Any]]], int]:
        if not image_path:
            return None, 0
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("PIL is required to load images for generation.") from e

        if isinstance(image_path, list):
            imgs = [Image.open(p).convert("RGB") for p in image_path]
            return imgs, len(imgs)
        img = Image.open(image_path).convert("RGB")
        return img, 1

    def _parse_segments(self, text: str) -> List[str]:
        if text is None:
            return []
        resp = str(text)
        if self._special_re is not None:
            resp = re.sub(self._special_re, "", resp)

        matches = re.findall(self._step_answer_pattern, resp)
        segs = (
            [m[0] if m[0] else m[1] for m in matches] if matches else []
        )
        segs = [_normalize_text(s) for s in segs if _normalize_text(s)]

        if not segs:
            m = re.search(
                r"Answer:\s*(?:The final answer is\s*)?(.*)",
                resp,
                re.IGNORECASE,
            )
            if m and _normalize_text(m.group(1)):
                return [_normalize_text(m.group(1))]
        if not segs and _normalize_text(resp):
            return [_normalize_text(resp)]
        return segs

    def generate(
        self,
        user_prompt: str,
        image_path: Optional[Union[str, List[str]]],
        n: int,
        sampling: SamplingConfig,
    ) -> List[GenResult]:
        from vllm import SamplingParams

        imgs, image_count = self._load_images(image_path)
        chat_prompt = self._to_chat_prompt(user_prompt, image_count=image_count)

        inp: Dict[str, Any] = {"prompt": chat_prompt}
        if imgs is not None:
            inp["multi_modal_data"] = {"image": imgs}

        sp_kwargs = dict(
            temperature=float(sampling.temperature),
            max_tokens=int(sampling.max_new_tokens),
            top_p=float(sampling.top_p),
            top_k=int(sampling.top_k),
            repetition_penalty=float(sampling.repetition_penalty),
            skip_special_tokens=False,
            n=max(1, int(n)),
        )
        if sampling.seed is not None and int(sampling.seed) >= 0:
            # vLLM supports seed in most versions; keep backward compatibility.
            try:
                sampling_params = SamplingParams(**sp_kwargs, seed=int(sampling.seed))
            except TypeError:
                sampling_params = SamplingParams(**sp_kwargs)
        else:
            sampling_params = SamplingParams(**sp_kwargs)
        outputs = self.llm.generate([inp], sampling_params=sampling_params)
        if not outputs:
            return []

        res: List[GenResult] = []
        for out in outputs[0].outputs:
            token_len = len(getattr(out, "token_ids", []) or [])
            raw_text = str(getattr(out, "text", "") or "")
            segs = self._parse_segments(raw_text)
            res.append(GenResult(segments=segs, token_len=int(token_len), raw_text=raw_text))
        return res


def oversample_and_select(
    raw: Sequence[GenResult],
    k: int,
) -> Tuple[List[GenResult], int]:
    """
    Select best k rollouts using select_best_rollouts, while preserving per-result token_len.
    Returns (selected_results, effective_completion_tokens).
    """
    k = int(k)
    if k <= 0:
        return [], 0
    seg_lists = [r.segments for r in raw]
    selected_seg_lists = select_best_rollouts(seg_lists, k=k)

    selected: List[GenResult] = []
    effective_tokens = 0
    used = set()
    for segs in selected_seg_lists:
        key = tuple(_normalize_text(x) for x in segs if _normalize_text(x))
        picked = None
        for i, r in enumerate(raw):
            if i in used:
                continue
            rkey = tuple(_normalize_text(x) for x in r.segments if _normalize_text(x))
            if rkey == key:
                picked = i
                break
        if picked is None:
            selected.append(GenResult(segments=list(segs), token_len=0))
        else:
            used.add(picked)
            selected.append(raw[picked])
            effective_tokens += int(raw[picked].token_len)

    return selected, int(effective_tokens)
