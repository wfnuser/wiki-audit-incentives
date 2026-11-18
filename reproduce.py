"""Reproduce contract calibration and complete-case evaluation without API access."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from experiments import large_sample_statistics as stats

ROOT = Path(__file__).resolve().parent


def load(name):
    rows = [json.loads(line) for line in (ROOT / 'data' / f'{name}.jsonl').read_text().splitlines()]
    stats.row_options(rows)  # Validate IDs, source groups, and binary outcomes.
    return rows


def screening(rows):
    selected = [r for r in rows if r['s'] == 1]
    def counts(rs):
        return dict(submitted=len(rs), accepted=sum(r['a'] for r in rs),
                    true_positives=sum(r['a'] * r['y'] for r in rs),
                    false_positives=sum(r['a'] * (1-r['y']) for r in rs))
    full, target = counts(rows), counts(selected)
    random_tp = random_fp = 0.0
    by_task = {}
    for task in sorted({r['task'] for r in rows}):
        group = [r for r in rows if r['task'] == task]
        chosen = [r for r in group if r['s']]
        f, t = counts(group), counts(chosen)
        rate = len(chosen) / len(group)
        random_tp += rate * f['true_positives']
        random_fp += rate * f['false_positives']
        by_task[task] = dict(full=f, selected=t)
    return dict(full=full, selected=target, by_task=by_task,
                matched_random_expectation=dict(submitted=len(selected),
                    true_positives=random_tp, false_positives=random_fp),
                review_reduction=1-len(selected)/len(rows),
                true_positive_retention=target['true_positives']/full['true_positives'])


def reproduce(replicates):
    manifest = json.loads((ROOT/'data/SHA256.json').read_text())
    for filename, expected in manifest.items():
        if hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Data checksum mismatch: {filename}')
    calibration, evaluation = load('calibration'), load('evaluation')
    contracts = json.loads((ROOT/'data/contracts.json').read_text())
    worlds = stats.completion_counts(calibration)
    fits = dict(primary=stats.calibrate(worlds),
                robust_tv_001=stats.calibrate(worlds, radius=Fraction(1,100)))
    if fits != contracts:
        raise ValueError('Recalibrated contracts differ from frozen contracts')
    complete = [r for r in evaluation if r['s'] is not None and r['a'] is not None]
    train_complete = [r for r in calibration if r['s'] is not None and r['a'] is not None]
    results = {}
    for name, contract in contracts.items():
        R, D, c = (Fraction(contract[k]) for k in ('R', 'D', 'c'))
        results[name] = dict(contract=contract,
            calibration_complete_case=stats.law_metrics(stats.completion_counts(train_complete)[0], R,D,c),
            evaluation_complete_case=stats.law_metrics(stats.completion_counts(complete)[0], R,D,c),
            evaluation_assigned_sample_bounds={policy:stats.policy_bounds(evaluation,policy,R,D,c)
                for policy in ('idle','blind','M:00','M:01','M:10','M:11')})
        if replicates:
            results[name]['evaluation_complete_case_bootstrap'] = stats.bootstrap(complete,replicates,20260917086,R,D,c)
    token_rows = [r for r in complete if r['m_tokens'] is not None and r['a_tokens'] is not None]
    full = sum(r['a_tokens'] for r in token_rows)
    investigation = sum(r['m_tokens'] for r in token_rows)
    selected = sum(r['a_tokens'] for r in token_rows if r['s'])
    return dict(sample_sizes={name:dict(assigned=len(rows),
                   complete=sum(r['s'] is not None and r['a'] is not None for r in rows),
                   sources=len({(r['task'],r['source_hash']) for r in rows}),
                   tasks=dict(Counter(r['task'] for r in rows)))
                   for name,rows in [('calibration',calibration),('evaluation',evaluation)]},
                calibration_completion_count=len(worlds),contracts=results,
                evaluation_complete_case_screening=screening(complete),
                token_replay=dict(responses=len(token_rows),full_adjudication=full,
                   investigation=investigation,selected_adjudication=selected,
                   routed_total=investigation+selected,relative_change=(investigation+selected)/full-1))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=0,help='Number of source-cluster bootstrap replicates (paper setting: 10000).')
    parser.add_argument('--output',type=Path,help='Write JSON to a new file; existing results are never overwritten.')
    args=parser.parse_args()
    if args.bootstrap < 0: parser.error('--bootstrap must be nonnegative')
    if args.output and args.output.exists(): parser.error('Output exists; select a new path.')
    result=reproduce(args.bootstrap)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    else: print(json.dumps(result,indent=2,allow_nan=False))
