from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Orchestrator output JSON")
    args = parser.parse_args()

    data: List[Dict[str, Any]] = json.load(open(args.input, "r", encoding="utf-8"))
    assert isinstance(data, list)

    acc_flags: List[float] = []
    eff_tokens: List[float] = []
    act_tokens: List[float] = []
    pool_sizes: List[float] = []
    used_continue: List[float] = []
    stop_at: Dict[int, int] = {}

    for it in data:
        trace = it.get("trace", {}) or {}
        eff = float(trace.get("tokens_effective_from_scratch", 0)) + float(
            trace.get("tokens_effective_continue", 0)
        )
        act = float(trace.get("tokens_actual_from_scratch", 0)) + float(
            trace.get("tokens_actual_continue", 0)
        )
        eff_tokens.append(eff)
        act_tokens.append(act)

        cands = it.get("candidates", []) or []
        pool_sizes.append(float(len(cands)))

        rounds = trace.get("rounds", []) or []
        used_continue.append(
            1.0 if any(r.get("gen_kind") == "continue" for r in rounds) else 0.0
        )

        # stop point = pool_size in last round
        if rounds:
            last = rounds[-1]
            ps = int(last.get("pool_size", len(cands)))
            stop_at[ps] = stop_at.get(ps, 0) + 1

        if "is_correct_best" in it:
            acc_flags.append(float(it.get("is_correct_best", 0)))

    print(f"items: {len(data)}")
    if acc_flags:
        print(f"accuracy: {_mean(acc_flags):.4f}")
    else:
        print("accuracy: N/A (no judge labels)")

    print(f"avg_pool_size: {_mean(pool_sizes):.2f}")
    print(f"avg_effective_tokens: {_mean(eff_tokens):.1f}")
    print(f"avg_actual_tokens: {_mean(act_tokens):.1f}")
    print(f"used_continue_rate: {_mean(used_continue):.3f}")

    if stop_at:
        items = sorted(stop_at.items(), key=lambda x: x[0])
        print("stop_at_pool_size:")
        for k, v in items:
            print(f"  - {k}: {v} ({v/len(data):.3f})")


if __name__ == "__main__":
    main()

