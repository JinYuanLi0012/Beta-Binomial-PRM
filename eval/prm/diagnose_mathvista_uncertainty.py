import argparse
import json
import math
import os
from typing import Callable, Dict, List, Tuple


def _safe_mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _parse_float_grid(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float grid: {s}")
    return vals


def _rankdata(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    mx = _safe_mean(x)
    my = _safe_mean(y)
    num = 0.0
    vx = 0.0
    vy = 0.0
    for a, b in zip(x, y):
        da = a - mx
        db = b - my
        num += da * db
        vx += da * da
        vy += db * db
    den = math.sqrt(vx * vy)
    return float(num / den) if den > 0 else 0.0


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    w = pos - lo
    return float(ys[lo] * (1.0 - w) + ys[hi] * w)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _score_linear(mu_steps: List[float], sigma_steps: List[float], lam: float) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    step_scores = [m - lam * s for m, s in zip(mu_steps, sigma_steps)]
    return _safe_mean(step_scores)


def _score_topq_penalty(
    mu_steps: List[float], sigma_steps: List[float], lam: float, q: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    tau = _quantile(sigma_steps, q)
    step_scores = [m - lam * max(0.0, s - tau) for m, s in zip(mu_steps, sigma_steps)]
    return _safe_mean(step_scores)


def _score_risk_budget(
    mu_steps: List[float], sigma_steps: List[float], lam: float, tau: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    risk_ratio = _safe_mean([1.0 if s > tau else 0.0 for s in sigma_steps])
    return _safe_mean(mu_steps) - lam * risk_ratio


def _score_soft_gate(
    mu_steps: List[float], sigma_steps: List[float], tau: float, temp: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    t = max(float(temp), 1e-6)
    ws = [_sigmoid((tau - s) / t) for s in sigma_steps]
    den = sum(ws)
    if den <= 0:
        return _safe_mean(mu_steps)
    return float(sum(w * m for w, m in zip(ws, mu_steps)) / den)


def _score_quantile_aggregate(
    mu_steps: List[float], sigma_steps: List[float], lam: float, q: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    step_scores = [m - lam * s for m, s in zip(mu_steps, sigma_steps)]
    return _quantile(step_scores, q)


def _score_softmin_aggregate(
    mu_steps: List[float], sigma_steps: List[float], lam: float, temp: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    t = max(float(temp), 1e-6)
    step_scores = [m - lam * s for m, s in zip(mu_steps, sigma_steps)]
    # score = -1/t * log(mean(exp(-t*score_i))) with numerical stability.
    vals = [-t * s for s in step_scores]
    vmax = max(vals)
    lse = vmax + math.log(_safe_mean([math.exp(v - vmax) for v in vals]))
    return float(-lse / t)


def _score_mink_aggregate(
    mu_steps: List[float], sigma_steps: List[float], lam: float, k_frac: float
) -> float:
    if not mu_steps:
        return -1e9
    if not sigma_steps:
        sigma_steps = [0.0] * len(mu_steps)
    step_scores = sorted([m - lam * s for m, s in zip(mu_steps, sigma_steps)])
    kf = min(max(float(k_frac), 0.0), 1.0)
    k = max(1, int(math.ceil(len(step_scores) * kf)))
    return _safe_mean(step_scores[:k])


def _bon_accuracy(
    data: List[dict], score_fn: Callable[[List[float], List[float]], float]
) -> Tuple[float, int, int]:
    correct = 0
    total = 0
    for item in data:
        labels = item.get("labels", [])
        mu_rollouts = item.get("prm_mu", [])
        sigma_rollouts = item.get("prm_sigma", [])
        if not labels or not mu_rollouts:
            continue
        n = min(len(labels), len(mu_rollouts))
        scores = []
        for i in range(n):
            mu_steps = mu_rollouts[i] if i < len(mu_rollouts) else []
            sigma_steps = sigma_rollouts[i] if i < len(sigma_rollouts) else []
            scores.append(score_fn(mu_steps, sigma_steps))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        correct += 1 if int(labels[best_idx]) == 1 else 0
        total += 1
    acc = (correct / total) if total > 0 else 0.0
    return acc, correct, total


def _collect_rollout_points(data: List[dict]) -> Tuple[List[float], List[int]]:
    # x: rollout uncertainty proxy (mean sigma), y: rollout error (1 if wrong)
    xs = []
    ys = []
    for item in data:
        labels = item.get("labels", [])
        sigma_rollouts = item.get("prm_sigma", [])
        n = min(len(labels), len(sigma_rollouts))
        for i in range(n):
            sigma_mean = _safe_mean([float(v) for v in sigma_rollouts[i]])
            label = int(labels[i])
            error = 0 if label == 1 else 1
            xs.append(sigma_mean)
            ys.append(error)
    return xs, ys


def _collect_all_step_sigmas(data: List[dict]) -> List[float]:
    vals = []
    for item in data:
        sigma_rollouts = item.get("prm_sigma", [])
        for steps in sigma_rollouts:
            vals.extend(float(v) for v in steps)
    return vals


def _bucket_error_rates(xs: List[float], ys: List[int], num_buckets: int = 5) -> List[Dict]:
    if not xs or len(xs) != len(ys):
        return []
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    buckets = []
    n = len(xs)
    for b in range(num_buckets):
        l = b * n // num_buckets
        r = (b + 1) * n // num_buckets
        idxs = order[l:r]
        if not idxs:
            continue
        sigma_vals = [xs[i] for i in idxs]
        err_vals = [ys[i] for i in idxs]
        buckets.append(
            {
                "bucket": b + 1,
                "size": len(idxs),
                "sigma_min": min(sigma_vals),
                "sigma_max": max(sigma_vals),
                "sigma_mean": _safe_mean(sigma_vals),
                "error_rate": _safe_mean([float(v) for v in err_vals]),
            }
        )
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True, help="Path to evaluator output json.")
    parser.add_argument("--out-json", type=str, default="", help="Optional output path.")
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0,0.1,0.2,0.35,0.5,0.7,1.0",
        help="Comma-separated risk lambdas for BoN sweep.",
    )
    parser.add_argument("--num-buckets", type=int, default=5)
    parser.add_argument(
        "--topq-grid",
        type=str,
        default="0.7,0.8,0.9",
        help="Comma-separated q grid for top-q penalty threshold.",
    )
    parser.add_argument(
        "--budget-q-grid",
        type=str,
        default="0.7,0.8,0.9",
        help="Comma-separated global sigma quantile grid for risk-budget threshold.",
    )
    parser.add_argument(
        "--soft-q-grid",
        type=str,
        default="0.7,0.8,0.9",
        help="Comma-separated global sigma quantile grid for soft-gate threshold.",
    )
    parser.add_argument(
        "--soft-temp-grid",
        type=str,
        default="0.01,0.02,0.05,0.1",
        help="Comma-separated temperature grid for soft gate.",
    )
    parser.add_argument(
        "--quantile-grid",
        type=str,
        default="0.2,0.3,0.4,0.5",
        help="Comma-separated quantile grid for quantile aggregation.",
    )
    parser.add_argument(
        "--softmin-temp-grid",
        type=str,
        default="5,10,20",
        help="Comma-separated temperature grid for soft-min aggregation.",
    )
    parser.add_argument(
        "--mink-frac-grid",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated fraction grid for min-k aggregation.",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    lambdas = _parse_float_grid(args.lambdas)
    topq_grid = _parse_float_grid(args.topq_grid)
    budget_q_grid = _parse_float_grid(args.budget_q_grid)
    soft_q_grid = _parse_float_grid(args.soft_q_grid)
    soft_temp_grid = _parse_float_grid(args.soft_temp_grid)
    quantile_grid = _parse_float_grid(args.quantile_grid)
    softmin_temp_grid = _parse_float_grid(args.softmin_temp_grid)
    mink_frac_grid = _parse_float_grid(args.mink_frac_grid)
    all_step_sigmas = _collect_all_step_sigmas(data)
    verbose_progress = True

    # Explicit baseline: mu-only BoN score = mean(mu)
    mu_only_acc, mu_only_correct, mu_only_total = _bon_accuracy(
        data, score_fn=lambda mu, sigma: _safe_mean(mu)
    )
    mu_only_result = {
        "accuracy": mu_only_acc,
        "correct": mu_only_correct,
        "total": mu_only_total,
    }

    # Strategy 1: global linear penalty
    linear_sweep = []
    linear_total = len(lambdas)
    linear_idx = 0
    for lam in lambdas:
        linear_idx += 1
        acc, correct, total = _bon_accuracy(
            data, score_fn=lambda mu, sigma, ll=lam: _score_linear(mu, sigma, ll)
        )
        linear_sweep.append(
            {
                "lambda": lam,
                "accuracy": acc,
                "correct": correct,
                "total": total,
            }
        )
        if verbose_progress:
            print(
                f"[prog] linear {linear_idx}/{linear_total} "
                f"lambda={lam} acc={acc:.4f} ({correct}/{total})"
            )
    linear_best = max(linear_sweep, key=lambda x: x["accuracy"]) if linear_sweep else None
    if verbose_progress and linear_best is not None:
        print(
            f"[prog] linear done: best lambda={linear_best['lambda']} "
            f"acc={linear_best['accuracy']:.4f}"
        )

    # Strategy 2: top-q gated penalty
    topq_sweep = []
    topq_total = len(topq_grid) * len(lambdas)
    topq_idx = 0
    for q in topq_grid:
        for lam in lambdas:
            topq_idx += 1
            acc, correct, total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, ll=lam, qq=q: _score_topq_penalty(
                    mu, sigma, ll, qq
                ),
            )
            topq_sweep.append(
                {
                    "lambda": lam,
                    "q": q,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] topq {topq_idx}/{topq_total} "
                    f"q={q} lambda={lam} acc={acc:.4f} ({correct}/{total})"
                )
    topq_best = max(topq_sweep, key=lambda x: x["accuracy"]) if topq_sweep else None
    if verbose_progress and topq_best is not None:
        print(
            f"[prog] topq done: best q={topq_best['q']} lambda={topq_best['lambda']} "
            f"acc={topq_best['accuracy']:.4f}"
        )

    # Strategy 3: risk-budget penalty
    budget_sweep = []
    budget_total = len(budget_q_grid) * len(lambdas)
    budget_idx = 0
    for q in budget_q_grid:
        tau = _quantile(all_step_sigmas, q)
        for lam in lambdas:
            budget_idx += 1
            acc, correct, total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, ll=lam, tt=tau: _score_risk_budget(
                    mu, sigma, ll, tt
                ),
            )
            budget_sweep.append(
                {
                    "lambda": lam,
                    "tau": tau,
                    "q": q,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] budget {budget_idx}/{budget_total} "
                    f"q={q} tau={tau:.4f} lambda={lam} acc={acc:.4f} ({correct}/{total})"
                )
    budget_best = max(budget_sweep, key=lambda x: x["accuracy"]) if budget_sweep else None
    if verbose_progress and budget_best is not None:
        print(
            f"[prog] budget done: best q={budget_best['q']} lambda={budget_best['lambda']} "
            f"tau={budget_best['tau']:.4f} acc={budget_best['accuracy']:.4f}"
        )

    # Strategy 4: soft-gated weighted mean grid (q x temp)
    soft_sweep = []
    soft_total_combos = len(soft_q_grid) * len(soft_temp_grid)
    soft_idx = 0
    for q in soft_q_grid:
        tau = _quantile(all_step_sigmas, q)
        for temp in soft_temp_grid:
            soft_idx += 1
            soft_acc, soft_correct, soft_total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, tt=tau, tp=temp: _score_soft_gate(
                    mu, sigma, tt, tp
                ),
            )
            soft_sweep.append(
                {
                    "tau": tau,
                    "q": q,
                    "temp": temp,
                    "accuracy": soft_acc,
                    "correct": soft_correct,
                    "total": soft_total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] soft-gate {soft_idx}/{soft_total_combos} "
                    f"q={q} tau={tau:.4f} temp={temp} acc={soft_acc:.4f} "
                    f"({soft_correct}/{soft_total})"
                )
    soft_best = max(soft_sweep, key=lambda x: x["accuracy"]) if soft_sweep else None
    if verbose_progress and soft_best is not None:
        print(
            f"[prog] soft-gate done: best q={soft_best['q']} temp={soft_best['temp']} "
            f"tau={soft_best['tau']:.4f} acc={soft_best['accuracy']:.4f}"
        )

    # Strategy 5: non-average quantile aggregation (on step scores)
    quantile_sweep = []
    quantile_total = len(quantile_grid) * len(lambdas)
    quantile_idx = 0
    for q in quantile_grid:
        for lam in lambdas:
            quantile_idx += 1
            acc, correct, total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, ll=lam, qq=q: _score_quantile_aggregate(
                    mu, sigma, ll, qq
                ),
            )
            quantile_sweep.append(
                {
                    "lambda": lam,
                    "q": q,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] quantile {quantile_idx}/{quantile_total} "
                    f"q={q} lambda={lam} acc={acc:.4f} ({correct}/{total})"
                )
    quantile_best = (
        max(quantile_sweep, key=lambda x: x["accuracy"]) if quantile_sweep else None
    )
    if verbose_progress and quantile_best is not None:
        print(
            f"[prog] quantile done: best q={quantile_best['q']} lambda={quantile_best['lambda']} "
            f"acc={quantile_best['accuracy']:.4f}"
        )

    # Strategy 6: non-average soft-min aggregation (on step scores)
    softmin_sweep = []
    softmin_total = len(softmin_temp_grid) * len(lambdas)
    softmin_idx = 0
    for temp in softmin_temp_grid:
        for lam in lambdas:
            softmin_idx += 1
            acc, correct, total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, ll=lam, tt=temp: _score_softmin_aggregate(
                    mu, sigma, ll, tt
                ),
            )
            softmin_sweep.append(
                {
                    "lambda": lam,
                    "temp": temp,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] softmin {softmin_idx}/{softmin_total} "
                    f"temp={temp} lambda={lam} acc={acc:.4f} ({correct}/{total})"
                )
    softmin_best = max(softmin_sweep, key=lambda x: x["accuracy"]) if softmin_sweep else None
    if verbose_progress and softmin_best is not None:
        print(
            f"[prog] softmin done: best temp={softmin_best['temp']} lambda={softmin_best['lambda']} "
            f"acc={softmin_best['accuracy']:.4f}"
        )

    # Strategy 7: non-average min-k aggregation (on step scores)
    mink_sweep = []
    mink_total = len(mink_frac_grid) * len(lambdas)
    mink_idx = 0
    for k_frac in mink_frac_grid:
        for lam in lambdas:
            mink_idx += 1
            acc, correct, total = _bon_accuracy(
                data,
                score_fn=lambda mu, sigma, ll=lam, kk=k_frac: _score_mink_aggregate(
                    mu, sigma, ll, kk
                ),
            )
            mink_sweep.append(
                {
                    "lambda": lam,
                    "k_frac": k_frac,
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }
            )
            if verbose_progress:
                print(
                    f"[prog] mink {mink_idx}/{mink_total} "
                    f"k_frac={k_frac} lambda={lam} acc={acc:.4f} ({correct}/{total})"
                )
    mink_best = max(mink_sweep, key=lambda x: x["accuracy"]) if mink_sweep else None
    if verbose_progress and mink_best is not None:
        print(
            f"[prog] mink done: best k_frac={mink_best['k_frac']} lambda={mink_best['lambda']} "
            f"acc={mink_best['accuracy']:.4f}"
        )

    xs, ys = _collect_rollout_points(data)
    pearson = _pearson(xs, [float(y) for y in ys]) if xs else 0.0
    spearman = _spearman(xs, [float(y) for y in ys]) if xs else 0.0
    buckets = _bucket_error_rates(xs, ys, num_buckets=args.num_buckets)

    report = {
        "input_json": args.input_json,
        "num_items": len(data),
        "num_rollouts": len(xs),
        "mu_only_baseline": mu_only_result,
        "strategy_eval": {
            "linear_mu_minus_lambda_sigma": {
                "sweep": linear_sweep,
                "best": linear_best,
            },
            "topq_gated_penalty": {
                "sweep": topq_sweep,
                "best": topq_best,
            },
            "risk_budget_penalty": {
                "sweep": budget_sweep,
                "best": budget_best,
            },
            "soft_gate_weighted_mean": {
                "sweep": soft_sweep,
                "best": soft_best,
            },
            "quantile_aggregate": {
                "sweep": quantile_sweep,
                "best": quantile_best,
            },
            "softmin_aggregate": {
                "sweep": softmin_sweep,
                "best": softmin_best,
            },
            "mink_aggregate": {
                "sweep": mink_sweep,
                "best": mink_best,
            },
        },
        # Backward-compatible fields
        "lambda_sweep": linear_sweep,
        "best_lambda": linear_best,
        "sigma_error_correlation": {
            "pearson": pearson,
            "spearman": spearman,
            "note": "y=1 means rollout is wrong (label!=1). Positive means sigma aligns with risk.",
        },
        "sigma_error_buckets": buckets,
    }

    print(f"[diag] items={report['num_items']} rollouts={report['num_rollouts']}")
    print(
        f"[diag] mu-only baseline acc={mu_only_result['accuracy']:.4f} "
        f"({mu_only_result['correct']}/{mu_only_result['total']})"
    )
    if linear_best is not None:
        print(
            f"[diag] linear best lambda={linear_best['lambda']} acc={linear_best['accuracy']:.4f} "
            f"({linear_best['correct']}/{linear_best['total']})"
        )
    if topq_best is not None:
        print(
            f"[diag] topq best lambda={topq_best['lambda']} q={topq_best['q']:.2f} "
            f"acc={topq_best['accuracy']:.4f} ({topq_best['correct']}/{topq_best['total']})"
        )
    if budget_best is not None:
        print(
            f"[diag] budget best lambda={budget_best['lambda']} q={budget_best['q']:.2f} "
            f"tau={budget_best['tau']:.4f} acc={budget_best['accuracy']:.4f} "
            f"({budget_best['correct']}/{budget_best['total']})"
        )
    if soft_best is not None:
        print(
            f"[diag] soft-gate best q={soft_best['q']:.2f} tau={soft_best['tau']:.4f} temp={soft_best['temp']:.4f} "
            f"acc={soft_best['accuracy']:.4f} ({soft_best['correct']}/{soft_best['total']})"
        )
    if quantile_best is not None:
        print(
            f"[diag] quantile best lambda={quantile_best['lambda']} q={quantile_best['q']:.2f} "
            f"acc={quantile_best['accuracy']:.4f} ({quantile_best['correct']}/{quantile_best['total']})"
        )
    if softmin_best is not None:
        print(
            f"[diag] softmin best lambda={softmin_best['lambda']} temp={softmin_best['temp']:.4f} "
            f"acc={softmin_best['accuracy']:.4f} ({softmin_best['correct']}/{softmin_best['total']})"
        )
    if mink_best is not None:
        print(
            f"[diag] mink best lambda={mink_best['lambda']} k_frac={mink_best['k_frac']:.2f} "
            f"acc={mink_best['accuracy']:.4f} ({mink_best['correct']}/{mink_best['total']})"
        )
    print(
        f"[diag] corr(sigma,error): pearson={report['sigma_error_correlation']['pearson']:.4f}, "
        f"spearman={report['sigma_error_correlation']['spearman']:.4f}"
    )
    for b in buckets:
        print(
            f"[diag] bucket{b['bucket']}: sigma_mean={b['sigma_mean']:.4f}, "
            f"error_rate={b['error_rate']:.4f}, size={b['size']}"
        )

    out_json = args.out_json
    if not out_json:
        stem, ext = os.path.splitext(args.input_json)
        out_json = f"{stem}_uncertainty_diag.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[diag] saved: {out_json}")


if __name__ == "__main__":
    main()
