"""Build IDH_Reserved_Test.ipynb: one-time evaluation of the frozen ensemble + hybrid on the 200 reserved patients
(balanced_holdout_ids.json) that no v2 checkpoint and no combiner has seen.
Reuses the hybrid notebook's configuration, cohort and helper cells so the development frame is rebuilt identically.
Run: python build_reserved_test_notebook.py [output.ipynb]
"""
import sys
from pathlib import Path
import nbformat as nbf

HYB = nbf.read(Path(__file__).with_name('IDH_Hybrid_Analysis.ipynb'), 4)
def after(prefix):
    for i, c in enumerate(HYB.cells):
        if c.cell_type == 'markdown' and c.source.lstrip().startswith(prefix): return HYB.cells[i + 1].source
    raise KeyError(prefix)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# Reserved internal test: the frozen ensemble and hybrid on 200 unseen patients

`balanced_holdout_ids.json` reserved 200 patients (100 IDH-mutant, 100 wildtype; UCSF 61, UTSW 76, UPENN 63) before
any v2 training. They are absent from every v2 fold manifest, from the ensemble's out-of-fold predictions and from the
1,393-patient frame on which the hybrid was fitted. This notebook evaluates, **once and unchanged**:

* the three-member ensemble (CA42 + CA44 + MC42; each member = mean of its 5 fold checkpoints; equal weights),
* every frozen combiner in `outputs/idh_hybrid/frozen_models.json`, with the frozen Youden and 90%-sensitivity thresholds,
* the parameter-free confidence gate (RSNA) between features + age and the recalibrated ensemble.

Nothing is fitted or tuned on these patients. Imaging features are z-scored with the **development** statistics of the
patient's own site (the test set is case-control enriched, so its own statistics would be distorted).

**Read the results as a case-control test.** Prevalence is 50% by design, so AUROC, sensitivity and specificity carry
over; raw PPV, NPV and Brier do not. PPV and NPV are also reported re-weighted to the development prevalence.

**Order of work.** Section 2 writes `protocol.json` (frozen-model hash, checkpoints, patient list, endpoints) *before*
any prediction exists. Re-running is safe: predictions are cached and the analysis is deterministic; a changed protocol
is refused.
''')

md('## 1. Configuration')
code(after('## 1.') + r'''

# ---- reserved-test settings
OUTPUT_DIR = PROJECT / 'outputs' / 'idh_reserved_test'
FROZEN = PROJECT / 'outputs' / 'idh_hybrid' / 'frozen_models.json'
BASE = Path.home() / 'Downloads' / 'outputs' / 'clinical_experiments'
THREE = ['v2_reference_seed42', 'v2_reference_seed44', 'v2_mean_concat_seed42']
HOLDOUT = Path.home() / 'Downloads' / 'outputs' / 'balanced_holdout_ids.json'
IMAGE_CACHE = Path.home() / 'Downloads' / 'outputs' / 'reserved_test_cache'   # preprocessed crops; kept OUT of OneDrive
PATH_REMAP = {r'C:\Users\claks\Downloads\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5': r'C:\Users\claks\Downloads\UCSF-PDGM'}
N_REPRO = 8                 # development patients re-predicted per member to check the inference path reproduces the OOF
REPRO_TOL = 0.03            # max |p_new - p_oof| allowed (mixed-precision training-time evaluation vs fp32 here)
DEVICE = 'cuda'
''')

md('## 2. Setup, development frame and protocol')
code(after('## 2.'))
code(after('## 3.'))
code(r'''
import sys, os, gc, hashlib, csv as _csv
from scipy.stats import beta
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True); IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
frozen = json.loads(Path(FROZEN).read_text())
frozen_sha = hashlib.sha256(Path(FROZEN).read_bytes()).hexdigest()
hold = json.loads(Path(HOLDOUT).read_text()); test_ids = sorted(hold['test_patient_ids'])
assert len(test_ids) == 200, len(test_ids)

def resolve(p):
    if not p: return ''
    q = Path(p)
    if not q.exists():
        for old, new in PATH_REMAP.items():
            if str(q).lower().startswith(old.lower()):
                c = Path(new + str(q)[len(old):])
                if c.exists(): return str(c)
    return str(q)

# ---- leakage checks against everything the hybrid was built on
dev_ids = set(d.patient_id)
oof_ids = set(pd.read_csv(RADGENOME_OOF).patient_id)
assert not dev_ids & set(test_ids), 'reserved patients in the hybrid development frame'
assert not oof_ids & set(test_ids), 'reserved patients in the ensemble OOF predictions'
checkpoints = []
for name in THREE:
    run = BASE / name
    man = pd.read_csv(run / 'fold_manifest.csv', dtype=str, keep_default_na=False)
    assert not set(man.patient_id) & set(test_ids), f'{name}: reserved patients in its fold manifest'
    for k in range(1, 6):
        p = run / f'fold_{k}' / 'best.pt'; assert p.exists(), p
        checkpoints.append(dict(member=name, fold=k, path=str(p), bytes=p.stat().st_size, mtime=int(p.stat().st_mtime)))
print(f'No overlap with the {len(dev_ids)}-patient hybrid frame, the {len(oof_ids)} OOF predictions or any of the 3 fold manifests.')

# ---- earlier uses of the reserved set (for the paper's disclosure)
earlier = []
for root, dirs, files in os.walk(Path.home() / 'Downloads' / 'outputs'):
    dirs[:] = [x for x in dirs if 'cache' not in x.lower()]
    if Path(root).name in ('balanced_test', 'final_test') or 'selection.json' in files:
        f = Path(root) / 'metrics.json'
        auc = json.loads(f.read_text()).get('idh', {}).get('auroc') if f.exists() else None
        earlier.append(dict(folder=str(Path(root).relative_to(Path.home())), idh_auroc=auc,
                            date=time.strftime('%Y-%m-%d', time.localtime(Path(root).stat().st_mtime))))
earlier = pd.DataFrame(earlier)
print('Earlier evaluations on the reserved set (none of these models is used here):')
display(earlier if len(earlier) else 'none found')

# ---- the protocol is written before any prediction is made
protocol = dict(frozen_models=str(FROZEN), frozen_models_sha256=frozen_sha, holdout_fingerprint=hold.get('fingerprint'),
                test_ids_sha256=hashlib.sha256('\n'.join(test_ids).encode()).hexdigest(),
                checkpoints=[{k: v for k, v in c.items() if k != 'mtime'} for c in checkpoints],
                ensemble='mean over 3 members of the mean over 5 fold checkpoints (probabilities)',
                features='development extraction (features.csv); z-scored with development site statistics',
                primary='AUROC of the frozen Hybrid',
                secondary=['paired AUROC differences: Hybrid minus each other frozen model',
                           'sensitivity/specificity at frozen Youden and 90%-sensitivity thresholds',
                           'PPV/NPV re-weighted to development prevalence', 'grade-4 and per-site subgroups'],
                bootstrap=dict(reps=N_BOOT, seed=SEED, strata='site x IDH'))
pf = OUTPUT_DIR / 'protocol.json'
if pf.exists():
    old = json.loads(pf.read_text()); old.pop('written', None)
    if old != protocol: raise ValueError('protocol.json differs from the current settings: the protocol may not change after it is written')
    print('Protocol unchanged since', json.loads(pf.read_text())['written'])
else:
    pf.write_text(json.dumps(dict(protocol, written=time.strftime('%Y-%m-%d %H:%M')), indent=2)); print('Wrote', pf)
''')

md(r'''
## 3. Ensemble predictions for the reserved patients

Same loader, preprocessing and checkpoints as the RHUH run. Each checkpoint's `train_ids`/`validation_ids` are checked
against the reserved list before it is used. Probabilities are cached per checkpoint in `IMAGE_CACHE`.
First, a reproduction check: a few development patients from each member's fold-1 validation set are re-predicted
and compared with that member's saved out-of-fold probabilities.
''')
code(r'''
import torch
from torch.utils.data import DataLoader
from research_experiments import load_checkpoint, dataset_for

mask_tab = pd.read_csv(MASK_TABLE, dtype=str, keep_default_na=False)
mrow = raw.set_index('patient_id')
def to_row(pid, tumor_mask):
    r = mrow.loc[pid]
    lab = lambda v: None if str(v).strip() in ('', 'None', 'nan') else int(float(v))
    return dict(patient_id=pid, site=r.site, subject_id=r.get('subject_id', ''), idh=lab(r.idh), mgmt=lab(r.mgmt),
                idh_observed=lab(r.idh) is not None, mgmt_observed=lab(r.mgmt) is not None,
                t1c_path=resolve(r.t1c_path), flair_path=resolve(r.flair_path),
                brain_mask_path=resolve(r.brain_mask_path), tumor_mask_path=resolve(tumor_mask))
tm = mask_tab.set_index('patient_id').tumor_mask_path
assert set(mask_tab.loc[mask_tab.role == 'balanced_test', 'patient_id']) == set(test_ids), 'mask table test roles differ from the holdout file'
rows = [to_row(p, tm.get(p, '')) for p in test_ids]
missing = [r['patient_id'] for r in rows if not (Path(r['t1c_path']).exists() and Path(r['flair_path']).exists())]
assert not missing, f'{len(missing)} reserved patients have missing images, e.g. {missing[:3]} (add the folder to PATH_REMAP)'
print(len(rows), 'reserved patients;', sum(r['idh'] == 1 for r in rows), 'IDH-mutant; tumour mask for',
      sum(bool(r['tumor_mask_path']) and Path(r['tumor_mask_path']).exists() for r in rows))

def predict_rows(model, state, rows_, tag):
    state = dict(state); state['preprocessing'] = dict(state['preprocessing'], cache_dir=str(IMAGE_CACHE / tag))
    out = []
    with torch.inference_mode():
        for b in DataLoader(dataset_for(state, rows_), batch_size=1, shuffle=False, num_workers=0):
            p = model(b['t1c'].to(DEVICE), b['flair'].to(DEVICE)).float().sigmoid()[:, 0].cpu().numpy()
            if not np.isfinite(p).all(): raise FloatingPointError('non-finite prediction')
            out.append(p)
    return np.concatenate(out)

signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
arrays, repro = [], []
for j, c in enumerate(checkpoints):
    dest = IMAGE_CACHE / f'{c["member"]}_fold{c["fold"]}.npz'
    state = torch.load(c['path'], map_location='cpu', weights_only=True)
    seen = set(state['train_ids']) | set(state['validation_ids'])
    if seen & set(test_ids): raise ValueError(f'{c["member"]} fold {c["fold"]}: reserved patient inside the checkpoint')
    need_repro = c['fold'] == 1
    if dest.exists() and not need_repro:
        with np.load(dest) as z:
            assert str(z['signature']) == signature, 'cached predictions come from a different protocol'; a = z['p'].copy()
        print(f'{j + 1:2d}/15 {c["member"]} fold {c["fold"]}: cached')
    else:
        model, state = load_checkpoint(c['path'], DEVICE)
        if need_repro:          # reproduce saved OOF probabilities for a few fold-1 validation patients
            man = pd.read_csv(BASE / c['member'] / 'fold_manifest.csv', dtype=str, keep_default_na=False).set_index('patient_id')
            oofm = pd.read_csv(BASE / c['member'] / 'oof_predictions.csv').set_index('patient_id')
            pids = sorted(state['validation_ids'])[:N_REPRO]
            dev_rows = [to_row(p, man.loc[p, 'tumor_mask_path']) for p in pids]
            pn = predict_rows(model, state, dev_rows, 'repro_' + c['member'])
            diff = np.abs(pn - oofm.loc[pids, 'idh_probability'].values)
            repro.append(dict(member=c['member'], patients=len(pids), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean())))
            print(f'   reproduction {c["member"]}: max |p - p_oof| = {diff.max():.4f}')
        if dest.exists():
            with np.load(dest) as z: a = z['p'].copy()
        else:
            t0 = time.time(); a = predict_rows(model, state, rows, 'reserved')
            np.savez(dest, p=a, signature=signature)
            print(f'{j + 1:2d}/15 {c["member"]} fold {c["fold"]}: {len(a)} patients in {time.time() - t0:.0f}s')
        del model; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    del state
    assert a.shape == (len(rows),) and np.all((a >= 0) & (a <= 1)); arrays.append(a)
repro = pd.DataFrame(repro); display(repro)
if len(repro) and repro.max_abs_diff.max() > REPRO_TOL:
    raise AssertionError('Inference does not reproduce the saved OOF probabilities: stop and check preprocessing/paths before reading any result')
member = np.stack(arrays).reshape(3, 5, len(rows)).mean(1)
ens = pd.DataFrame(dict(patient_id=test_ids, probability=member.mean(0), **{n + '_probability': member[i] for i, n in enumerate(THREE)}))
ens.to_csv(OUTPUT_DIR / 'reserved_ensemble_predictions.csv', index=False)
''')

md(r'''
## 4. Apply the frozen models

Features come from the development extraction (`features.csv`), with the same exclusions as development (extraction
errors, missing conventional or mismatch features, missing age). The confidence gate's threshold is the Youden point of
the gate on the development out-of-fold arm probabilities (nothing from the reserved set).
''')
code(r'''
site_mu = {c: d.groupby('site')[c].mean() for c in CONV + MISM if c != 'no_enhancement'}
site_sd = {c: d.groupby('site')[c].std(ddof=0) + 1e-9 for c in CONV + MISM if c != 'no_enhancement'}

t = cohort[cohort.patient_id.isin(test_ids)].merge(ens, on='patient_id')
steps = [('reserved patients', len(t))]
t = t[t.y.isin([0, 1])]; steps.append(('IDH known', len(t)))
t_all = t.copy()                                                    # ensemble-only secondary analysis uses everyone
t = t.merge(feats, on='patient_id'); steps.append(('with imaging features (no extraction error)', len(t)))
t = t[t.age.notna()]; steps.append(('with age', len(t)))
t['necrosis_frac'] = t.necrosis_frac.fillna(0.0)
t = t.dropna(subset=CONV + MISM).reset_index(drop=True); steps.append(('complete conventional + mismatch features', len(t)))
t['y'] = t.y.astype(int)
excl = t_all[~t_all.patient_id.isin(t.patient_id)]
display(pd.DataFrame(steps, columns=['step', 'patients']))
print('Excluded from the hybrid analysis by site and IDH:', excl.groupby(['site', 'y']).size().to_dict())
for c in site_mu:
    t[c] = (t[c] - t.site.map(site_mu[c])) / t.site.map(site_sd[c])
p_ = t.probability.clip(1e-6, 1 - 1e-6); t['dl_logit'] = np.log(p_ / (1 - p_))
display(pd.crosstab([t.site, t.grade_group], t.y.map({1: 'IDH-mutant', 0: 'IDH-wildtype'}), margins=True))

def frozen_p(frame, name):
    spec = frozen['models'][name]
    if name == RAW: return frame.probability.values
    X = frame[spec['inputs']].values.astype(float)
    z = (X - np.array(spec['scaler_mean'])) / np.array(spec['scaler_scale'])
    return 1 / (1 + np.exp(-(z @ np.array(spec['coef']) + spec['intercept'])))

def gate(pm, pdl):
    cm, cd = np.abs(pm - .5), np.abs(pdl - .5)
    w = np.where(cm + cd > 0, cm / np.maximum(cm + cd, 1e-12), .5); return w * pm + (1 - w) * pdl
GATE = 'Confidence gate (RSNA)'
dev_cv = run_cv(Z, {k: MODELS[k] for k in ['Features + age', 'Ensemble (recalibrated)']}, include_raw=False)
gate_dev = gate(dev_cv['Features + age']['p'], dev_cv['Ensemble (recalibrated)']['p'])
TH = {m: (frozen['models'][m]['threshold_youden'], frozen['models'][m]['threshold_sens90']) for m in ORDER}
TH[GATE] = (youden_threshold(Z.y.values, gate_dev), sens_threshold(Z.y.values, gate_dev))
print(f'Gate on development OOF: AUROC {roc_auc_score(Z.y.values, gate_dev):.3f}; thresholds {TH[GATE][0]:.3f} / {TH[GATE][1]:.3f}')

NAMES = ORDER + [GATE]
P = {m: frozen_p(t, m) for m in ORDER}
P[GATE] = gate(P['Features + age'], P['Ensemble (recalibrated)'])
pd.DataFrame(dict(patient_id=t.patient_id, site=t.site, y=t.y, grade_group=t.grade_group, age=t.age,
                  **{m: P[m] for m in NAMES})).to_csv(OUTPUT_DIR / 'reserved_predictions.csv', index=False)
''')

md('## 5. Results')
code(r'''
PI = float(d.y.mean())          # development prevalence, for re-weighted PPV / NPV
y = t.y.values
strata = (t.site + '_' + t.y.astype(str)).values
groups = [np.where(strata == s)[0] for s in np.unique(strata)]
rng = np.random.default_rng(SEED)
BOOT = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(N_BOOT)]

def exact_ci(k, n):
    if n == 0: return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1.0 if k == n else beta.ppf(.975, k + 1, n - k))
def ci_str(lo, hi): return f'{lo:.3f}-{hi:.3f}'

def summary(m, yy, pp, boot=None):
    out = dict(model=m, n=len(yy), idh_mutant=int(yy.sum()), auroc=roc_auc_score(yy, pp),
               average_precision=average_precision_score(yy, pp), brier_case_control=float(np.mean((pp - yy) ** 2)),
               calibration_slope=calib(yy, pp)[1])
    if boot is not None:
        a = [roc_auc_score(yy[i], pp[i]) for i in boot]; out['auroc_ci'] = ci_str(*np.percentile(a, [2.5, 97.5]))
    for tag, th in zip(('youden', 'sens90'), TH[m]):
        b = binary_metrics(yy, (pp >= th).astype(int)); se, sp = b['sensitivity'], b['specificity']
        out[f'threshold@{tag}'] = th
        out[f'sensitivity@{tag}'] = se; out[f'sens_ci@{tag}'] = ci_str(*exact_ci(b['tp'], b['tp'] + b['fn']))
        out[f'specificity@{tag}'] = sp; out[f'spec_ci@{tag}'] = ci_str(*exact_ci(b['tn'], b['tn'] + b['fp']))
        out[f'ppv_dev_prev@{tag}'] = se * PI / (se * PI + (1 - sp) * (1 - PI)) if se * PI + (1 - sp) * (1 - PI) else np.nan
        out[f'npv_dev_prev@{tag}'] = sp * (1 - PI) / (sp * (1 - PI) + (1 - se) * PI) if sp * (1 - PI) + (1 - se) * PI else np.nan
        out[f'tp/fp/tn/fn@{tag}'] = f"{b['tp']}/{b['fp']}/{b['tn']}/{b['fn']}"
    return out

res = pd.DataFrame([summary(m, y, P[m], BOOT) for m in NAMES])
res.to_csv(OUTPUT_DIR / 'reserved_metrics.csv', index=False)
show = ['model', 'n', 'idh_mutant', 'auroc', 'auroc_ci', 'sensitivity@youden', 'sens_ci@youden', 'specificity@youden',
        'spec_ci@youden', 'ppv_dev_prev@youden', 'npv_dev_prev@youden', 'calibration_slope']
display(res[show].round(3))

pairs = [('Hybrid', m) for m in NAMES if m != 'Hybrid'] + [('Ensemble + age', 'Ensemble (recalibrated)'), (GATE, 'Ensemble (recalibrated)')]
drows = []
for a, b in pairs:
    v = [roc_auc_score(y[i], P[a][i]) - roc_auc_score(y[i], P[b][i]) for i in BOOT]
    drows.append(dict(comparison=f'{a} minus {b}', delta_auroc=roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]),
                      ci=ci_str(*np.percentile(v, [2.5, 97.5])), p_le_0=float(np.mean(np.array(v) <= 0))))
diffs = pd.DataFrame(drows); diffs.to_csv(OUTPUT_DIR / 'reserved_paired_differences.csv', index=False); display(diffs.round(3))

# ensemble alone on everyone with an IDH label (no feature or age requirement)
ya, pa = t_all.y.astype(int).values, t_all.probability.values
ens_all = dict(n=len(ya), idh_mutant=int(ya.sum()), auroc=roc_auc_score(ya, pa))
print(f"Ensemble (raw) on all {ens_all['n']} labelled reserved patients: AUROC {ens_all['auroc']:.3f}")
''')

code(r'''
# subgroups (descriptive): grade 4, each site
sub_rows = []
for label, mask in [('grade 4', (t.grade_group == 'grade 4').values)] + [(f'site {s}', (t.site == s).values) for s in sorted(t.site.unique())]:
    yy = y[mask]
    if yy.sum() < 2 or (1 - yy).sum() < 2: continue
    gi = [np.where(mask & (y == v))[0] for v in (0, 1)]
    bb = [np.concatenate([rng.choice(g, len(g)) for g in gi]) for _ in range(N_BOOT)]
    for m in ['Age', 'Features + age', 'Ensemble (recalibrated)', 'Ensemble + age', 'Hybrid', GATE]:
        a = [roc_auc_score(y[i], P[m][i]) for i in bb]
        sub_rows.append(dict(subgroup=label, model=m, n=int(mask.sum()), idh_mutant=int(yy.sum()),
                             auroc=roc_auc_score(yy, P[m][mask]), auroc_ci=ci_str(*np.percentile(a, [2.5, 97.5]))))
subs = pd.DataFrame(sub_rows); subs.to_csv(OUTPUT_DIR / 'reserved_subgroups.csv', index=False)
display(subs.pivot_table(index='model', columns='subgroup', values='auroc').round(3))

fig, ax = plt.subplots(figsize=(4.6, 4.4))
for m, col in [('Age', '#9e9e9e'), ('Features + age', '#3b8e3f'), ('Ensemble (recalibrated)', '#2a6fdb'), ('Hybrid', '#c0392b')]:
    fpr, tpr, _ = roc_curve(y, P[m]); ax.plot(fpr, tpr, color=col, lw=1.8, label=f'{m} ({roc_auc_score(y, P[m]):.3f})')
ax.plot([0, 1], [0, 1], ':', color='#bbbbbb'); ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'Reserved test (n={len(y)}, {int(y.sum())} mutant)', fontsize=10); ax.legend(fontsize=7, loc='lower right')
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'reserved_roc.png', dpi=200); plt.show()
''')

code(r'''
fmt = lambda v: f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
def table(df_, cols_):
    return ['| ' + ' | '.join(cols_) + ' |', '|' + '---|' * len(cols_)] + ['| ' + ' | '.join(fmt(r[c]) for c in cols_) + ' |' for r in df_.to_dict('records')]
L = ['# Reserved internal test (one-time evaluation of the frozen models)', '',
     f'Run: {time.strftime("%Y-%m-%d %H:%M")}. Protocol written: {json.loads(pf.read_text())["written"]}. '
     f'frozen_models.json sha256 {frozen_sha[:12]}.', '',
     f'Patients: {len(test_ids)} reserved; {len(t)} analysed ({int(y.sum())} IDH-mutant). Excluded (no features/age): '
     f'{len(excl)} ({excl.groupby(["site", "y"]).size().to_dict()}).', '',
     'Case-control design (about 50% prevalence): AUROC, sensitivity and specificity transfer; PPV/NPV are re-weighted to '
     f'the development prevalence ({PI:.3f}); Brier is shown for the case-control mix only.', '',
     '## Inference reproduction check', '', repro.round(4).to_string(index=False), '',
     '## All models (frozen thresholds)', ''] + table(res, show) + [
     '', '## Paired AUROC differences (site x IDH stratified bootstrap)', ''] + table(diffs, ['comparison', 'delta_auroc', 'ci', 'p_le_0']) + [
     '', f'Ensemble (raw) on all {ens_all["n"]} labelled reserved patients: AUROC {ens_all["auroc"]:.3f}.', '',
     '## Subgroups (descriptive)', ''] + table(subs, ['subgroup', 'model', 'n', 'idh_mutant', 'auroc', 'auroc_ci']) + [
     '', '## Earlier evaluations on the reserved set', '',
     (earlier.to_string(index=False) if len(earlier) else 'none found'), '',
     '- None of the earlier models is part of the paper; disclose them anyway.',
     '- Ensemble members, ensemble weights, combiners and thresholds were all fixed on development data before this run.']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(L), encoding='utf-8'); print('\n'.join(L))
''')

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'IDH_Reserved_Test.ipynb'
nbf.write(nb, out); print('Wrote', out)
