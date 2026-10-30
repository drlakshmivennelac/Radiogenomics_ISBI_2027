"""Build IDH_Only_v2_Recipe.ipynb: retrain the CA42 member with an IDH-only head on the paper's v2 recipe and folds
(reserved 200 excluded), and compare with the joint-head CA42 out-of-fold predictions."""
import sys, nbformat as nbf
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))
md(r'''
# IDH-only head vs joint IDH/MGMT head on the paper's recipe

Retrains the CA42 member (`v2_reference_seed42`: cross-attention, tumour crop, seed 42) with **only the IDH head**,
on exactly the same 1,435 development patients and the same five folds. The 200 reserved patients are not in that
manifest, so this run never sees them. About 4-5 GPU hours; re-running the training cell resumes an interrupted run.

The only change is the head (and checkpoint selection on IDH validation loss, which is what the joint loss reduces to
when MGMT is absent). The comparison cell then pairs the new out-of-fold predictions with CA42's.
''')
code(r'''
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
PROJECT = Path.cwd()
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
BASE = Path.home() / 'Downloads' / 'outputs' / 'clinical_experiments'
SOURCE = BASE / 'v2_reference_seed42'
OUT = BASE / 'v2_idh_only_seed42'
HOLDOUT = Path.home() / 'Downloads' / 'outputs' / 'balanced_holdout_ids.json'
DEVICE, SEED, N_BOOT = 'cuda', 42, 2000
''')
code(r'''
from clinical_quality import read_manifest
reserved = set(json.loads(HOLDOUT.read_text())['test_patient_ids'])
rows = [r for r in read_manifest(SOURCE) if r['role'] == 'development']
assert not reserved & {r['patient_id'] for r in rows}, 'reserved patients in the training manifest'
print(len(rows), 'development patients;', sum(r['idh'] == 1 for r in rows), 'IDH-mutant; none of the 200 reserved')
''')
code(r'''
from research_experiments import train_experiment, config_from_run, ExperimentModel
from experiments_v2 import TumorCropDataset
cfg, model, prep = config_from_run(SOURCE)
cfg.seed = SEED; cfg.monitor = 'idh_loss'
print('model', model); print('preprocessing', {k: v for k, v in prep.items() if k != 'cache_dir'})
train_experiment(rows, cfg, dict(model), dict(prep), OUT, target='idh', device=DEVICE,
                 model_factory=ExperimentModel, dataset_factory=TumorCropDataset)
''')
code(r'''
j = pd.read_csv(SOURCE / 'oof_predictions.csv'); i = pd.read_csv(OUT / 'oof_predictions.csv')
m = j.merge(i[['patient_id', 'idh_probability']], on='patient_id', suffixes=('_joint', '_idh_only'))
m = m[m.idh_label.notna()]
assert not set(m.patient_id) & reserved
y = m.idh_label.astype(int).values; a, b = m.idh_probability_joint.values, m.idh_probability_idh_only.values
rng = np.random.default_rng(SEED); st = (m.site + '_' + m.idh_label.astype(int).astype(str)).values
G = [np.where(st == s)[0] for s in np.unique(st)]
d = [roc_auc_score(y[k], b[k]) - roc_auc_score(y[k], a[k]) for k in
     (np.concatenate([rng.choice(g, len(g)) for g in G]) for _ in range(N_BOOT))]
res = dict(n=len(y), idh_mutant=int(y.sum()), auroc_joint=roc_auc_score(y, a), auroc_idh_only=roc_auc_score(y, b),
           diff_idh_only_minus_joint=roc_auc_score(y, b) - roc_auc_score(y, a), diff_ci=list(np.percentile(d, [2.5, 97.5])),
           brier_joint=float(np.mean((a - y) ** 2)), brier_idh_only=float(np.mean((b - y) ** 2)))
for s_ in sorted(m.site.unique()):
    k = (m.site == s_).values
    res[f'auroc_{s_}'] = dict(joint=roc_auc_score(y[k], a[k]), idh_only=roc_auc_score(y[k], b[k]))
(OUT / 'comparison_with_joint.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
''')
nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
nbf.write(nb, sys.argv[1] if len(sys.argv) > 1 else 'IDH_Only_v2_Recipe.ipynb'); print('written')
