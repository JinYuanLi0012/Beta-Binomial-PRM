from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def split_steps_answer(segments: Sequence[str]) -> Tuple[List[str], str]:
    segs = [_norm(s) for s in segments if _norm(s)]
    if not segs:
        return [], ""
    if len(segs) == 1:
        return [], segs[0]
    return list(segs[:-1]), segs[-1]


def merge_prefix_continuation(
    prefix_steps: Sequence[str],
    cont_segments: Sequence[str],
) -> List[str]:
    """
    prefix_steps: reasoning steps only (no answer).
    cont_segments: parsed segments from generator, typically [new_steps..., answer]
    Returns full segments: [steps..., answer]
    """
    prefix = [_norm(s) for s in prefix_steps if _norm(s)]
    cont_steps, cont_answer = split_steps_answer(cont_segments)
    cont_steps = [_norm(s) for s in cont_steps if _norm(s)]

    # Drop any overlap between end(prefix) and start(cont_steps).
    max_overlap = min(len(prefix), len(cont_steps))
    drop = 0
    for ov in range(max_overlap, 0, -1):
        if prefix[-ov:] == cont_steps[:ov]:
            drop = ov
            break
    if drop > 0:
        cont_steps = cont_steps[drop:]

    answer = _norm(cont_answer)
    if not answer:
        # Fallback: if no explicit answer, use last continuation step as answer.
        if cont_steps:
            answer = cont_steps[-1]
            cont_steps = cont_steps[:-1]
    if not answer:
        # Still empty => invalid continuation; return prefix only (caller may decide to discard).
        return list(prefix)

    return list(prefix) + list(cont_steps) + [answer]


_CRITICAL_RE = re.compile(r"(\d|=|\+|\-|\*|/|\^|\\frac|sqrt|\\sqrt)")


@dataclass(frozen=True)
class CutConfig:
    c_cut: float = 2.0
    p_bad: float = 0.5
    min_step_chars: int = 3
    gibberish_min_len: int = 5
    gibberish_non_ascii_letter_ratio: float = 0.6
    gibberish_min_non_ascii_letters: int = 2


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF


def _is_gibberish(s: str, cfg: CutConfig) -> bool:
    """
    Heuristic filter for malformed/garbled step strings.
    Designed to be conservative: it should mostly remove obvious noise (e.g. '145Ёу').
    """
    s = _norm(s)
    if not s:
        return True
    if len(s) < int(cfg.min_step_chars):
        return True

    if "\ufffd" in s:
        return True

    if len(s) < int(cfg.gibberish_min_len):
        return False

    letters_total = 0
    non_ascii_letters = 0
    for ch in s:
        if ch.isalpha():
            letters_total += 1
            if ord(ch) > 127 and not _is_cjk(ch):
                non_ascii_letters += 1

    if non_ascii_letters >= int(cfg.gibberish_min_non_ascii_letters) and letters_total > 0:
        ratio = non_ascii_letters / float(letters_total)
        if ratio >= float(cfg.gibberish_non_ascii_letter_ratio):
            return True

    return False


def pick_cutpoint(
    steps: Sequence[str],
    mu: Sequence[float],
    sigma: Sequence[float],
    cfg: CutConfig,
) -> int:
    """
    steps: full segments including answer at the end.
    mu/sigma: aligned to each <prm> (same length as steps).
    Returns t in [0, len(reasoning_steps)] where prefix = reasoning_steps[:t].
    Never returns an index in the answer step.
    """
    steps = list(steps)
    T = min(len(steps), len(mu), len(sigma))
    if T <= 1:
        return 0

    reasoning_T = T - 1  # exclude answer
    c_cut = float(cfg.c_cut)
    p_bad = float(cfg.p_bad)

    # Candidate indices among reasoning steps only.
    cand_idx: List[int] = []
    for i in range(reasoning_T):
        st = _norm(steps[i])
        if not st:
            continue
        if len(st) < int(cfg.min_step_chars):
            continue
        if _is_gibberish(st, cfg):
            continue
        cand_idx.append(i)

    if not cand_idx:
        return 0

    # Rule 1: earliest confidently bad step by LCB < p_bad.
    for i in cand_idx:
        lcb = float(mu[i]) - c_cut * float(sigma[i])
        if lcb < p_bad:
            return int(i)

    # Rule 2: otherwise, choose the most uncertain critical step (argmax sigma).
    critical = [i for i in cand_idx if _CRITICAL_RE.search(_norm(steps[i]))]
    pool = critical if critical else cand_idx
    best_i = max(pool, key=lambda i: float(sigma[i]))
    return int(best_i)
