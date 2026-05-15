from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class JudgeConfig:
    api_base: str
    model_name: str
    timeout: int = 30
    retries: int = 5


def judge_yes_no(
    cfg: JudgeConfig,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> int:
    """
    Returns 1 if judge says correct, else 0.
    OpenAI-compatible endpoint: POST {api_base}/chat/completions
    """
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    prompt = f"""You are given a question, the correct answer and a model's answer. Please determine if the model's answer matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer.
Only the correctness of the model's answer matters.
Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{correct_answer}
--------------------------------
Model's Answer:
{model_answer}
--------------------------------"""
    payload = {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 8,
        "temperature": 0.0,
    }

    last_err: Optional[Exception] = None
    for _ in range(max(1, int(cfg.retries))):
        try:
            r = requests.post(url, json=payload, timeout=int(cfg.timeout))
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            content_str = content if isinstance(content, str) else str(content)
            lc = content_str.lower().strip()
            if "yes" in lc:
                return 1
            if "no" in lc:
                return 0
        except Exception as e:
            last_err = e
            continue
    return 0

