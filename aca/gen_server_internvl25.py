from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from gen_core_internvl25 import (
    GenResult,
    InternVL25Generator,
    SamplingConfig,
    make_continue_prompt,
    make_cot_prompt,
    oversample_and_select,
)


class SamplingCfgIn(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.05


class GenerateScratchIn(BaseModel):
    question: str
    image_path: Optional[Union[str, List[str]]] = None
    k: int = Field(..., ge=1)
    oversample_factor: float = Field(2.0, ge=1.0)
    sampling: SamplingCfgIn = SamplingCfgIn()
    seed: Optional[int] = None
    return_raw: bool = False


class GenerateContinueIn(BaseModel):
    question: str
    image_path: Optional[Union[str, List[str]]] = None
    prefix_steps: List[str] = []
    extra_instruction: Optional[str] = None
    m: int = Field(..., ge=1)
    oversample_factor: float = Field(2.0, ge=1.0)
    sampling: SamplingCfgIn = SamplingCfgIn()
    seed: Optional[int] = None
    return_raw: bool = False


class GenOutItem(BaseModel):
    segments: List[str]
    token_len: int
    raw_text: Optional[str] = None


class GenerateOut(BaseModel):
    selected: List[GenOutItem]
    raw_count: int
    kept_count: int
    effective_completion_tokens: int
    actual_completion_tokens: int
    raw: Optional[List[GenOutItem]] = None
    kept_indices: Optional[List[int]] = None
    user_prompt: Optional[str] = None


def _kept_indices(raw: List[GenResult], selected: List[GenResult]) -> List[int]:
    """
    Map selected results back to raw indices. Uses normalized segments + a used-set
    to disambiguate duplicates (should be rare after dedup).
    """
    from gen_core_internvl25 import _normalize_text

    used = set()
    indices: List[int] = []
    for s in selected:
        skey = tuple(_normalize_text(x) for x in (s.segments or []) if _normalize_text(x))
        picked = -1
        for i, r in enumerate(raw):
            if i in used:
                continue
            rkey = tuple(_normalize_text(x) for x in (r.segments or []) if _normalize_text(x))
            if rkey == skey:
                picked = int(i)
                break
        if picked >= 0:
            used.add(picked)
        indices.append(int(picked))
    return indices


def _to_sampling(cfg: SamplingCfgIn, seed: Optional[int]) -> SamplingConfig:
    return SamplingConfig(
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        top_k=int(cfg.top_k),
        max_new_tokens=int(cfg.max_new_tokens),
        repetition_penalty=float(cfg.repetition_penalty),
        seed=None if seed is None else int(seed),
    )


def _sum_tokens(raw: List[GenResult]) -> int:
    return int(sum(int(r.token_len) for r in raw))


def build_app(generator: InternVL25Generator) -> FastAPI:
    app = FastAPI(title="InternVL2.5 Gen Service (ACA)")

    @app.post("/generate_from_scratch", response_model=GenerateOut)
    def generate_from_scratch(req: GenerateScratchIn):
        k = int(req.k)
        first_n = max(k, int(round(k * float(req.oversample_factor))))
        sampling = _to_sampling(req.sampling, seed=req.seed)

        user_prompt = make_cot_prompt(req.question)
        raw = generator.generate(
            user_prompt=user_prompt,
            image_path=req.image_path,
            n=first_n,
            sampling=sampling,
        )

        selected, effective = oversample_and_select(raw, k=k)
        actual = _sum_tokens(raw)
        out = GenerateOut(
            selected=[GenOutItem(segments=r.segments, token_len=int(r.token_len)) for r in selected],
            raw_count=int(len(raw)),
            kept_count=int(len(selected)),
            effective_completion_tokens=int(effective),
            actual_completion_tokens=int(actual),
        )
        if bool(req.return_raw):
            out.raw = [
                GenOutItem(segments=r.segments, token_len=int(r.token_len), raw_text=r.raw_text)
                for r in raw
            ]
            out.kept_indices = _kept_indices(raw, selected)
            out.user_prompt = user_prompt
        return out

    @app.post("/generate_continue", response_model=GenerateOut)
    def generate_continue(req: GenerateContinueIn):
        m = int(req.m)
        first_n = max(m, int(round(m * float(req.oversample_factor))))
        sampling = _to_sampling(req.sampling, seed=req.seed)

        user_prompt = make_continue_prompt(
            req.question,
            req.prefix_steps,
            extra_instruction=req.extra_instruction,
        )
        raw = generator.generate(
            user_prompt=user_prompt,
            image_path=req.image_path,
            n=first_n,
            sampling=sampling,
        )

        selected, effective = oversample_and_select(raw, k=m)
        actual = _sum_tokens(raw)
        out = GenerateOut(
            selected=[GenOutItem(segments=r.segments, token_len=int(r.token_len)) for r in selected],
            raw_count=int(len(raw)),
            kept_count=int(len(selected)),
            effective_completion_tokens=int(effective),
            actual_completion_tokens=int(actual),
        )
        if bool(req.return_raw):
            out.raw = [
                GenOutItem(segments=r.segments, token_len=int(r.token_len), raw_text=r.raw_text)
                for r in raw
            ]
            out.kept_indices = _kept_indices(raw, selected)
            out.user_prompt = user_prompt
        return out

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": generator.model_path, "tp": generator.tensor_parallel_size}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="InternVL2.5 model path or name")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--image-limit", type=int, default=8, help="limit_mm_per_prompt image count")
    args = parser.parse_args()

    generator = InternVL25Generator(
        model=args.model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        limit_mm_per_prompt={"image": int(args.image_limit)},
    )
    app = build_app(generator)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
