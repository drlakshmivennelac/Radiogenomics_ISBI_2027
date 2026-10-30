"""Recompute MU-Glioma-Post external IDH metrics for the selected three-member ensemble.

Members (equal weights, 1/3 each): cross-attention seed 42 (v2_reference_seed42),
cross-attention seed 44 (v2_reference_seed44), mean-concat seed 42 (v2_mean_concat_seed42).

Source: outputs/mu_external_four_equal, which stores one (scans x [IDH, MGMT]) probability
array per member and fold. Each member is averaged over its five fold checkpoints (15
checkpoints in total), then the three members are averaged. Threshold = frozen OOF Youden
point of the three-member ensemble (0.14600733667612076). No MU labels are used for tuning.

Sanity check: averaging all four members (20 checkpoints) must reproduce the saved
four-member idh_probability column in predictions.csv.

Usage: python recompute_mu_three_member.py <path to outputs/mu_external_four_equal> <output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THREE = ['v2_reference_seed42', 'v2_reference_seed44', 'v2_mean_concat_seed42']
FOUR = ['v2_reference_seed42', 'v2_reference_seed43', 'v2_reference_seed44', 'v2_mean_concat_seed42']
THRESHOLD = 0.14600733667612076


def member_mean(src, name):
    return np.mean([np.load(src / f'{name}_fold_{k}.npy')[:, 0] for k in range(1, 6)], axis=0)


def metrics(sub, col, t):
    y = sub.idh.values.astype(int)
    p = sub[col].values
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return dict(n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()), threshold=t,
                tp=tp, tn=tn, fp=fp, fn=fn,
                auroc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=tp / (tp + fp) if tp + fp else None, npv=tn / (tn + fn) if tn + fn else None,
                positive_f1=2 * tp / (2 * tp + fp + fn), brier=float(np.mean((p - y) ** 2)),
                unique_patients=int(sub.subject_id.nunique()))


def patient_bootstrap_auc(sub, col, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = sub.subject_id.unique()
    groups = {s: sub[sub.subject_id == s] for s in ids}
    vals = []
    for _ in range(reps):
        b = pd.concat([groups[s] for s in rng.choice(ids, len(ids))])
        if b.idh.nunique() == 2:
            vals.append(roc_auc_score(b.idh.astype(int), b[col]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])], len(vals)


def main(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src / 'predictions.csv')
    probs = {m: member_mean(src, m) for m in FOUR}
    four = np.mean([probs[m] for m in FOUR], axis=0)
    check = float(np.max(np.abs(four - df.idh_probability.values)))
    if check > 1e-9:
        raise ValueError(f'Four-member reconstruction does not match saved predictions: {check}')
    df['idh_probability_three'] = np.mean([probs[m] for m in THREE], axis=0)
    for m in THREE:
        df[f'{m}_probability'] = probs[m]
    labelled = df[df.idh_observed].copy()
    earliest = labelled.sort_values(['subject_id', 'timepoint']).groupby('subject_id').head(1)
    result = dict(members=THREE, weights=[1 / 3] * 3, checkpoints=15, threshold=THRESHOLD,
                  four_member_reconstruction_max_abs_diff=check,
                  note='Recomputed from saved per-checkpoint external predictions; not a separately logged run. '
                       'Post-treatment cohort; CIs are patient-cluster percentile bootstrap (2,000), conditional on fixed models.')
    for name, sub in [('primary_earliest_scan_per_patient', earliest), ('secondary_all_labelled_scans', labelled)]:
        r = metrics(sub, 'idh_probability_three', THRESHOLD)
        r['auroc_95ci'], r['valid_replicates'] = patient_bootstrap_auc(sub, 'idh_probability_three')
        result[name] = r
    (out / 'mu_three_member_results.json').write_text(json.dumps(result, indent=2))
    keep = ['patient_id', 'subject_id', 'timepoint', 'days_from_diagnosis', 'idh', 'idh_observed',
            'idh_probability_three'] + [f'{m}_probability' for m in THREE]
    df[keep].to_csv(out / 'mu_three_member_scan_predictions.csv', index=False)
    print(json.dumps({k: result[k] for k in ('primary_earliest_scan_per_patient', 'secondary_all_labelled_scans')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
