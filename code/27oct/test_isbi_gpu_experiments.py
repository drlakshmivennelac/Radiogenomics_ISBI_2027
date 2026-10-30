"""Synthetic end-to-end test of isbi_gpu_experiments (CPU, tiny model, ~5 min). No real data or checkpoints.

Builds a fake 3-site cohort of NIfTI volumes, a fake 'source' member run, then exercises:
  E1 run_occlusion + load_occlusion; E2 ds_rows / ds_audit / frozen_inference; E4 loso_member + loso_combine.
Run: python test_isbi_gpu_experiments.py
"""
import csv, json, shutil, tempfile, sys, os
from pathlib import Path
import numpy as np, nibabel as nib, torch, pandas as pd

import isbi_gpu_experiments as G
from research_experiments import ExperimentModel, StudyConfig
from dataclasses import asdict

TMP = Path(tempfile.mkdtemp(prefix='isbi_gpu_test_'))
rng = np.random.default_rng(0)
SHAPE = (40, 44, 36)
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
MODEL = dict(base_channels=8, dim=32, heads=4, token_grid=[2, 2, 2], dropout=0.1, head_dropout=0.2, backbone_dropout=0.1)
PREP = dict(output_shape=[64, 64, 64], spacing_mm=2.0, cache_dir=str(TMP / 'train_cache'), cache_dtype='float32', padding_voxels=4)


def volume(mutant, enhancing=True):
    brain = np.zeros(SHAPE, bool); brain[5:35, 5:39, 4:32] = True
    t1 = np.where(brain, 100 + rng.normal(0, 5, SHAPE), 0).astype(np.float32)
    fl = np.where(brain, 80 + rng.normal(0, 5, SHAPE), 0).astype(np.float32)
    lab = np.zeros(SHAPE, np.int16)
    c = rng.integers(14, 24, 3); r = rng.integers(3, 6)
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    lab[d <= r + 2] = 2; lab[d <= r] = 1
    if enhancing and not mutant: lab[(d <= r) & (d > r - 1.5)] = 3
    t1[lab == 3] += 60; fl[lab > 0] += 40 + (30 if mutant else 0)
    return t1, fl, lab


def save(a, p):
    p.parent.mkdir(parents=True, exist_ok=True); nib.save(nib.Nifti1Image(a, AFF), str(p)); return str(p)


# ---------------------------------------------------------------- fake development cohort + source run
rows = []
for site, n in [('UCSF', 16), ('UTSW', 16), ('UPENN', 16)]:
    for i in range(n):
        idh = int(i % 3 == 0); mgmt = int(i % 2 == 0)
        t1, fl, lab = volume(idh)
        pid = f'{site}:{site}-{i:03d}'; d = TMP / 'data' / site / f'{i:03d}'
        rows.append(dict(site=site, subject_id=f'{site}-{i:03d}', patient_id=pid, t1c_path=save(t1, d / 't1c.nii.gz'),
                         flair_path=save(fl, d / 'flair.nii.gz'), brain_mask_path='', idh=idh, mgmt=mgmt,
                         idh_observed=True, mgmt_observed=True, role='development', cv_fold=0,
                         tumor_mask_path=save(lab, d / 'mask.nii.gz'), tumor_mask_source='provided'))
from experiments_v3 import v3_folds
cfg = StudyConfig(max_epochs=2, min_epochs=1, patience=1, warmup_epochs=1, cosine_t_max=1, amp=False, seed=42)
for r in rows: r['eligible'] = True
rows = v3_folds(rows, cfg)
BASE = TMP / 'clinical_experiments'
for member, seed, fusion in [('v2_reference_seed42', 42, 'cross_attention'), ('v2_reference_seed44', 44, 'cross_attention'),
                             ('v2_mean_concat_seed42', 42, 'mean_concat')]:
    src = BASE / member; src.mkdir(parents=True)
    keys = list(rows[0])
    with (src / 'fold_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    mc = dict(MODEL, fusion_mode=fusion)
    (src / 'study_config.json').write_text(json.dumps(dict(training=asdict(StudyConfig(**{**asdict(cfg), 'seed': seed})), model=mc, preprocessing=PREP)))
    # five fake fold checkpoints + OOF file, as the real runs have
    oof = []
    for k in range(5):
        torch.manual_seed(seed + k)
        m = ExperimentModel(target='joint', **mc)
        tr = [r['patient_id'] for r in rows if r['cv_fold'] != k]; va = [r['patient_id'] for r in rows if r['cv_fold'] == k]
        (src / f'fold_{k + 1}').mkdir()
        torch.save(dict(model_state=m.state_dict(), model_config=mc, preprocessing=PREP, target='joint', train_ids=tr,
                        validation_ids=va, fold=k + 1), src / f'fold_{k + 1}' / 'best.pt')
        for r in rows:
            if r['cv_fold'] == k:
                oof.append(dict(patient_id=r['patient_id'], site=r['site'], idh_label=r['idh'], idh_probability=0.5, fold=k + 1))
    pd.DataFrame(oof).to_csv(src / 'oof_predictions.csv', index=False)
    (src / 'summary.json').write_text('{}')

# ---------------------------------------------------------------- E1
print('E1 ...', flush=True)
from ensemble_oof_xrai import cohort
cases = cohort(runs=[BASE / m for m in G.MEMBERS], threshold=G.THRESHOLD)
out1 = G.run_occlusion(cases[:10], TMP / 'e1', TMP / 'novelty_cache', device='cpu', chunk=4)
long, ens = G.load_occlusion(out1)
assert len(ens) == 10 and (ens.n_members == 3).all(), ens
assert set(G.CONDITIONS) <= set(ens.columns)
assert (ens.lesion_voxels > 0).all()
# the original condition must equal a direct forward pass of that checkpoint
from research_experiments import load_checkpoint
from ensemble_oof_xrai import case_dataset
c0 = cases[0]; m0 = c0['members'][0]
model, state = load_checkpoint(m0['checkpoint'], 'cpu'); b = case_dataset(state, c0['row'], TMP / 'novelty_cache')[0]
with torch.no_grad(): direct = float(torch.sigmoid(model(b['t1c'][None], b['flair'][None])[0, 0]))
got = float(long[(long.patient_id == c0['patient_id']) & (long.member == 'v2_reference_seed42')]['original'].iloc[0])
assert abs(direct - got) < 1e-5, (direct, got)
# masks sit inside the crop and conditions really differ
with np.load(next((out1 / 'crop_masks').glob('*.npz'))) as z: assert z['labels'].shape == (64, 64, 64) and (z['labels'] > 0).sum() > 50
assert (long['tumour_ring_only'] != long['original']).any() and (long['t1c_blank'] != long['original']).any()
print('  E1 ok:', ens[G.CONDITIONS].mean().round(3).to_dict())

# ---------------------------------------------------------------- E2
print('E2 ...', flush=True)
DS = TMP / 'ds007045'
part = []
for i, sub in enumerate(['sub-BO1', 'sub-KA05', 'sub-NSK04', 'sub-TO015']):
    t1, fl, lab = volume(0); p = G.ds_paths(DS, sub)
    save(t1, p['t1c_path']); save(fl, p['flair_path']); save(fl * .9, p['t2_path']); save(lab, p['tumor_mask_path'])
    part.append(dict(participant_id=sub, sex=1, age=50 + i, MGMT=i % 2, field_strength='1,5' if i % 2 else '3', scanner_model='x'))
pd.DataFrame(part).to_csv(DS / 'participants.tsv', sep='\t', index=False)
rows2, missing = G.ds_rows(DS)
assert len(rows2) == 4 and missing.empty and rows2[0]['field_strength'] in (1.5, 3.0) and rows2[1]['centre'] == 'KA'
au = G.ds_audit(rows2); assert au.same_grid.all() and (au.error == '').all(), au
pr = G.frozen_inference(rows2, TMP / 'e2', device='cpu', base=BASE)
assert len(pr) == 4 and pr.probability.between(0, 1).all()
pr2 = G.frozen_inference(rows2, TMP / 'e2', device='cpu', base=BASE)       # resume path
assert np.allclose(pr.probability, pr2.probability)
print('  E2 ok:', pr.probability.round(3).tolist())

# ---------------------------------------------------------------- E4
if os.environ.get('SKIP_E4'): print('KEPT', TMP); sys.exit(0)
print('E4 (tiny training, 2 epochs x 5 folds x 2 members) ...', flush=True)
ref = TMP / 'loso_ref'
G.loso_member('v2_reference_seed42', 'UCSF', ref, device='cpu', base=BASE)
G.loso_member('v2_mean_concat_seed42', 'UCSF', TMP / 'loso_mc', device='cpu', base=BASE, reference_folds=ref / 'inner_cv')
try:
    G.loso_member('v2_reference_seed44', 'UCSF', TMP / 'loso_bad', device='cpu', base=BASE, fold_seed=7, reference_folds=ref / 'inner_cv')
    raise AssertionError('fold mismatch not detected')
except ValueError as e:
    assert 'inner folds differ' in str(e)
o = pd.read_csv(TMP / 'loso_mc' / 'outer_site_predictions.csv'); assert len(o) == 16 and set(o.site) == {'UCSF'}
cfg_saved = json.loads((TMP / 'loso_mc' / 'inner_cv' / 'study_config.json').read_text())
assert cfg_saved['model']['fusion_mode'] == 'mean_concat' and cfg_saved['training']['seed'] == 42
# resume: a second call must not retrain
G.loso_member('v2_reference_seed42', 'UCSF', ref, device='cpu', base=BASE)
# combine with a fake Z
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
fit = lambda X, y: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y)
Z = pd.DataFrame([dict(patient_id=r['patient_id'], site=r['site'], y=r['idh'], age=rng.normal(50, 10), probability=.5,
                       **{c: rng.normal() for c in ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'necrosis_frac',
                                                    'edema_ratio_log', 'mismatch_index', 'mismatch_frac']}, no_enhancement=0.0) for r in rows])
r_, d_, info = G.loso_combine('UCSF', [ref, TMP / 'loso_mc'], Z, fit, n_boot=50)
assert info['members'] == 2 and len(r_) == 7
print('  E4 ok')
import os
if not os.environ.get("KEEP"): shutil.rmtree(TMP)
else: print("KEPT", TMP)
print('ALL TESTS PASSED')
