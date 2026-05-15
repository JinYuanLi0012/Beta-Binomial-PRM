import argparse
import json
import math
import os
import random
from functools import reduce


def _aggregate_metric(metric_name, rollouts):
    if metric_name == 'min':
        return [min(x) if x else 0 for x in rollouts]
    if metric_name == 'last':
        return [x[-1] if x else 0 for x in rollouts]
    if metric_name == 'product':
        return [reduce(lambda a, b: a * b, x) if x else 0 for x in rollouts]
    if metric_name == 'average':
        return [sum(x) / len(x) if x else 0 for x in rollouts]
    if metric_name == 'sum_logprob':
        return [
            sum(math.log(a) if a != 0 else -9999 for a in x) if x else 0
            for x in rollouts
        ]
    if metric_name == 'max':
        return [max(x) if x else 0 for x in rollouts]
    if metric_name == 'sum_logit':
        return [
            sum(math.log(a / (1 - a)) if a != 1 else 9999 for a in x) if x else 0
            for x in rollouts
        ]
    if metric_name == 'mean_odd':
        return [
            (sum((a / (1 - a)) if a != 1 else 9999 for a in x) / len(x)) if x else 0
            for x in rollouts
        ]
    raise ValueError(f'Unknown metric: {metric_name}')


def _evaluate_rollout_metric(data, value_key, metric_name):
    cnt = 0
    for item in data:
        agg_scores = _aggregate_metric(metric_name, item[value_key])
        if item['labels'][agg_scores.index(max(agg_scores))] == 1:
            cnt += 1
    return {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }


def calculate_accuracy(data):
    results = {}

    cnts = []
    for _ in range(5000):
        cnt = 0
        for item in data:
            if random.choice(item['labels']) == 1:
                cnt += 1
        cnts.append(cnt)

    cnt = sum(cnts) / len(cnts)
    print(f'random {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['random'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(16, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@16 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@16'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(8, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@8 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@8'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(4, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@4 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@4'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(2, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@2 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@2'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    metric_names = [
        'min',
        'last',
        'product',
        'average',
        'max',
    ]

    for metric_name in metric_names:
        metric_result = _evaluate_rollout_metric(data, 'prm_scores', metric_name)
        print(
            f"prm accuracy {metric_name} {metric_result['correct']} / {len(data)}, "
            f"rate: {metric_result['accuracy'] * 100}%"
        )
        results[metric_name] = metric_result

    if all('prm_mu' in item for item in data):
        for metric_name in metric_names:
            metric_result = _evaluate_rollout_metric(data, 'prm_mu', metric_name)
            result_key = f'{metric_name}_mu'
            print(
                f"prm accuracy {result_key} {metric_result['correct']} / {len(data)}, "
                f"rate: {metric_result['accuracy'] * 100}%"
            )
            results[result_key] = metric_result

    selected_metrics = metric_names
    accs = [results[k]['accuracy'] for k in selected_metrics if k in results]
    ave = sum(accs) / len(accs) if accs else 0.0
    print(f'prm accuracy ave over {selected_metrics}: {ave * 100}%')
    results['ave'] = {'metrics': selected_metrics, 'accuracy': ave}

    mu_metric_names = [f'{name}_mu' for name in metric_names if f'{name}_mu' in results]
    if mu_metric_names:
        mu_accs = [results[k]['accuracy'] for k in mu_metric_names]
        ave_mu = sum(mu_accs) / len(mu_accs) if mu_accs else 0.0
        print(f'prm accuracy ave_mu over {mu_metric_names}: {ave_mu * 100}%')
        results['ave_mu'] = {'metrics': mu_metric_names, 'accuracy': ave_mu}

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()

    result_file = os.path.join(args.output_dir, args.output_file)

    print(f'Reading {result_file}...')
    results = calculate_accuracy(json.load(open(result_file)))

    print(f"Saving results to {result_file.replace('.json', f'_score.json')}...")
    json.dump(
        results,
        open(result_file.replace('.json', f'_score.json'), 'w'),
        indent=4,
        ensure_ascii=False,
    )
    print(f'Results saved.')
