"""Build IDH_Hybrid_Analysis.ipynb: where do handcrafted imaging features (+ age) help, where does the RadGenome
deep-learning ensemble help, and how good is the hybrid?  Uses the IDH correlates notebook's features.csv and the
selected three-member ensemble's out-of-fold predictions.  Run: python build_hybrid_notebook.py [output.ipynb]
"""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# Hybrid IDH model: handcrafted imaging features, age and the RadGenome ensemble

**Question.** Handcrafted, literature-based imaging features plus age and the deep-learning ensemble each predict IDH
well. Where does each help, and is the combination better calibrated and more accurate than either?

**Inputs.**
* `features.csv` from `IDH_Imaging_Correlates.ipynb`
* The selected three-member ensemble's out-of-fold IDH probabilities (cross-attention seeds 42 and 44 + mean-concat seed 42, 1/3 each)
* Age and WHO grade from each cohort's clinical file

No images are read, so the whole notebook runs in a few minutes.

**Models** (all except the raw ensemble are L2 logistic regressions):

| Model | Inputs |
|---|---|
| Age | age |
| Features | conventional lesion features + T2-FLAIR mismatch (available at every site) |
| Features + age | the above + age |
| Ensemble (raw) | ensemble probability as produced, with no refitting |
| Ensemble (recalibrated) | ensemble logit, logistic recalibration |
| Ensemble + age | |
| Ensemble + features | image-only hybrid |
| **Hybrid** | ensemble logit + features + age |
| Hybrid + DWI/PWI | adds whole-lesion ADC and perfusion; UCSF + UPENN only, exploratory |

**Evaluation (no test labels touch any choice).**
1. **Primary:** site- and class-stratified 5-fold cross-validation, repeated 5 times. Every operating threshold is chosen
   *inside* each training fold, by inner 5-fold CV (Youden, and a 90%-sensitivity threshold), then applied to the test
   fold. Each patient's binary call is the majority over the repeats; probabilities are averaged over the repeats.
2. **Leave-one-site-out** for the fitted models. The ensemble logit itself comes from pooled-site CV, so this tests the
   *combination* across sites, not the ensemble.
3. 95% CIs from 2,000 site x class stratified patient bootstraps; paired bootstraps for differences between models.

**Metrics.** AUROC, average precision, Brier score, ECE (10 equal-width bins and 10 equal-count bins), calibration
intercept and slope, sensitivity, specificity, PPV, NPV, F1 (IDH-mutant), macro-F1, balanced accuracy and MCC.

**Where each helps.**
* Subgroups: site, grade group, enhancing vs non-enhancing, age band, tumour volume and mask provenance.
* Discordance between the ensemble and features + age.
* Net reclassification of the hybrid over the ensemble.
* Hybrid coefficients.
* Decision curves.

**Caveats to carry into the paper.**
* The ensemble's members and its original threshold were selected on these same out-of-fold results, so ensemble-based
  numbers are optimistic.
* This is internal, multi-site evidence; external validation is a separate notebook.
''')

md('## 1. Configuration')
code(r'''
from pathlib import Path
PROJECT = Path.cwd()
MANIFEST = PROJECT / 'outputs' / 'patient_manifest.csv'
FEATURES = PROJECT / 'outputs' / 'idh_imaging_correlates' / 'features.csv'
RADGENOME_OOF = PROJECT / 'outputs' / 'selected_three_reliability_20260926_192650' / 'patient_predictions.csv'
MASK_TABLE = PROJECT / 'outputs' / 'v2_mask_content_audit.csv'      # mask provenance for the subgroup analysis
OUTPUT_DIR = PROJECT / 'outputs' / 'idh_hybrid'
ASSUME_GRADE4_SITES = {'UPENN'}
FROZEN_ENSEMBLE_THRESHOLD = 0.14600733667612076   # reported for reference only (selected on the same OOF data)

N_FOLDS, REPEATS, N_BOOT, SEED = 5, 5, 2000, 20261021
SENS_TARGET = 0.90
C_REG = 1.0                      # L2 strength for every logistic model (fixed in advance, not tuned)
''')

md('## 2. Setup and cohort')
code(r'''
import json, re, time, warnings, platform
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=RuntimeWarning)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__)

def norm_name(c): return ' ' + re.sub(r'[^a-z0-9]+', ' ', c.lower()) + ' '
def binary(v):
    v = str(v).strip().lower()
    return 1 if v in ('1', '1.0', 'true', 'mutant') else 0 if v in ('0', '0.0', 'false', 'wildtype', 'wt') else np.nan
def grade4(v):
    v = str(v).strip().upper().replace('GRADE', '').replace('WHO', '').replace('CNS', '').strip(' :-')
    if v in ('4', '4.0', 'IV') or 'GLIOBLASTOMA' in v or v == 'GBM': return 1.0
    if v in ('1', '2', '3', '1.0', '2.0', '3.0', 'I', 'II', 'III'): return 0.0
    return np.nan

def metadata_lookup(raw):
    grade, age = {}, {}
    for meta, g in raw.groupby('metadata_path'):
        if not meta or not Path(meta).exists():
            print('  metadata not found:', meta); continue
        md_ = pd.read_csv(meta, sep='\t' if meta.lower().endswith(('.tsv', '.txt')) else ',', dtype=str,
                          keep_default_na=False, encoding_errors='replace')
        to_pid = {}
        for r in g.itertuples():
            for k in {str(getattr(r, 'subject_id', '')), r.patient_id.split(':', 1)[-1]}:
                if k:
                    to_pid[k.strip()] = r.patient_id
                    to_pid.setdefault(re.sub(r'_\d+$', '', k.strip()), r.patient_id)
        idcol = max(md_.columns, key=lambda c: md_[c].str.strip().isin(to_pid).sum())
        gcol = next((c for c in md_.columns if ' grade ' in norm_name(c)), None)
        acol = next((c for c in md_.columns if ' age ' in norm_name(c)
                     and pd.to_numeric(md_[c], errors='coerce').notna().mean() > .5), None)
        for row in md_.to_dict('records'):
            pid = to_pid.get(str(row[idcol]).strip())
            if pid:
                if gcol: grade[pid] = row[gcol]
                if acol: age[pid] = row[acol]
        print(f'  {Path(meta).name}: id {idcol!r}, grade {gcol!r}, age {acol!r}')
    return grade, age

raw = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
g_map, a_map = metadata_lookup(raw)
cohort = pd.DataFrame(dict(patient_id=raw.patient_id, site=raw.site.str.upper(), y=raw.idh.map(binary),
                           age=pd.to_numeric(raw.patient_id.map(a_map), errors='coerce'),
                           grade4=raw.patient_id.map(g_map).map(grade4)))
cohort.loc[cohort.grade4.isna() & cohort.site.isin(ASSUME_GRADE4_SITES), 'grade4'] = 1.0
cohort['grade_group'] = cohort.grade4.map({1.0: 'grade 4', 0.0: 'grade 2-3'}).fillna('unknown')
if Path(MASK_TABLE).exists():
    mt = pd.read_csv(MASK_TABLE, dtype=str, keep_default_na=False)[['patient_id', 'source']]
    cohort = cohort.merge(mt.rename(columns={'source': 'mask_source'}), on='patient_id', how='left')
else:
    cohort['mask_source'] = 'unknown'

feats = pd.read_csv(FEATURES)
feats = feats[feats.error.isna() | (feats.error.astype(str).str.len() == 0)].drop(columns=['site'])
oof = pd.read_csv(RADGENOME_OOF)
steps = [('manifest', len(cohort))]
d = cohort[cohort.y.isin([0, 1])]; steps.append(('IDH known', len(d)))
d = d.merge(oof[['patient_id', 'probability', 'fold']], on='patient_id'); steps.append(('with ensemble OOF prediction', len(d)))
d = d.merge(feats, on='patient_id'); steps.append(('with imaging features', len(d)))
d = d[d.age.notna()]; steps.append(('with age', len(d)))
CONV = ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'no_enhancement', 'necrosis_frac', 'edema_ratio_log']
MISM = ['mismatch_index', 'mismatch_frac']
ADV = ['nadc_whole_median', 'nadc_nonenh_p10', 'nperf_whole_p90']
d['necrosis_frac'] = d.necrosis_frac.fillna(0.0)
d = d.dropna(subset=CONV + MISM).reset_index(drop=True); steps.append(('complete conventional + mismatch features', len(d)))
d['y'] = d.y.astype(int)
p_ = d.probability.clip(1e-6, 1 - 1e-6); d['dl_logit'] = np.log(p_ / (1 - p_))
display(pd.DataFrame(steps, columns=['step', 'patients']))
display(pd.crosstab([d.site, d.grade_group], d.y.map({1: 'IDH-mutant', 0: 'IDH-wildtype'}), margins=True))
''')

md(r'''
## 3. Metrics, cross-validation and bootstrap helpers

Feature columns are z-scored **within each site** without labels, the same protocol as the correlates notebook. Age and
the ensemble logit are left on their natural scale; the pipeline standardises everything again inside each training fold.
''')
code(r'''
def site_z(frame, cols_):
    out = frame.copy()
    for c in cols_:
        if c == 'no_enhancement': continue
        out[c] = frame.groupby('site')[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return out
Z = site_z(d, CONV + MISM + [c for c in ADV if c in d])

MODELS = {
    'Age': ['age'],
    'Features': CONV + MISM,
    'Features + age': CONV + MISM + ['age'],
    'Ensemble (recalibrated)': ['dl_logit'],
    'Ensemble + age': ['dl_logit', 'age'],
    'Ensemble + features': ['dl_logit'] + CONV + MISM,
    'Hybrid': ['dl_logit'] + CONV + MISM + ['age'],
}
RAW = 'Ensemble (raw)'
ORDER = ['Age', 'Features', 'Features + age', RAW, 'Ensemble (recalibrated)', 'Ensemble + age', 'Ensemble + features', 'Hybrid']

def fit(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, max_iter=5000)).fit(X, y)

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p); k = np.argmax(tpr - fpr); return float(min(thr[k], 1.0))
def sens_threshold(y, p, target=SENS_TARGET):
    fpr, tpr, thr = roc_curve(y, p); ok = np.where(tpr >= target)[0]; return float(min(thr[ok[0]], 1.0))

def inner_thresholds(X, y, strat, seed, raw_p=None):
    # thresholds chosen on the TRAINING fold only, from inner-CV probabilities
    if raw_p is not None:
        return youden_threshold(y, raw_p), sens_threshold(y, raw_p)
    pin = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        pin[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return youden_threshold(y, pin), sens_threshold(y, pin)

def run_cv(frame, models, repeats=REPEATS, include_raw=True):
    y = frame.y.values; strat = (frame.site + '_' + frame.y.astype(str)).values
    names = list(models) + ([RAW] if include_raw else [])
    P = {m: np.zeros((repeats, len(frame))) for m in names}
    DY = {m: np.zeros((repeats, len(frame)), int) for m in names}
    DS = {m: np.zeros((repeats, len(frame)), int) for m in names}
    TH = {m: [] for m in names}
    for r in range(repeats):
        for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(frame, strat)):
            for m in names:
                if m == RAW:
                    p_te = frame.probability.values[te]
                    ty, ts = inner_thresholds(None, y[tr], None, None, raw_p=frame.probability.values[tr])
                else:
                    X = frame[models[m]].values
                    p_te = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                    ty, ts = inner_thresholds(X[tr], y[tr], strat[tr], SEED + 100 * r + k)
                P[m][r, te] = p_te; DY[m][r, te] = p_te >= ty; DS[m][r, te] = p_te >= ts
                TH[m].append((ty, ts))
    out = {}
    for m in names:
        out[m] = dict(p=P[m].mean(0), pred_youden=(DY[m].mean(0) >= .5).astype(int),
                      pred_sens90=(DS[m].mean(0) >= .5).astype(int), thresholds=np.array(TH[m]))
    return out

def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
    m = LogisticRegression(penalty=None, max_iter=5000).fit(lg.reshape(-1, 1), y)
    slope = float(m.coef_[0][0])
    # calibration-in-the-large on the log-odds scale: observed minus mean predicted (approximation of the offset model)
    citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    return citl, slope

def ece(y, p, bins=10, adaptive=False):
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))

def binary_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan
    sens, spec = div(tp, tp + fn), div(tn, tn + fp); ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn); f1n = div(2 * tn, 2 * tn + fp + fn)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1,
                macro_f1=(f1 + f1n) / 2 if np.isfinite(f1) and np.isfinite(f1n) else np.nan,
                balanced_accuracy=(sens + spec) / 2, mcc=div(tp * tn - fp * fn, mcc_den))

def all_metrics(y, res):
    p = res['p']; citl, slope = calib(y, p)
    out = dict(n=len(y), idh_mutant=int(y.sum()), auroc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
               brier=float(np.mean((p - y) ** 2)), ece=ece(y, p), ece_adaptive=ece(y, p, adaptive=True),
               calibration_in_the_large=citl, calibration_slope=slope)
    out.update({f'{k}@youden': v for k, v in binary_metrics(y, res['pred_youden']).items()})
    out.update({f'{k}@sens90': v for k, v in binary_metrics(y, res['pred_sens90']).items()})
    return out

KEY = ['auroc', 'average_precision', 'brier', 'ece', 'ece_adaptive', 'calibration_slope',
       'sensitivity@youden', 'specificity@youden', 'ppv@youden', 'npv@youden', 'f1@youden', 'macro_f1@youden',
       'balanced_accuracy@youden', 'mcc@youden', 'sensitivity@sens90', 'specificity@sens90', 'ppv@sens90', 'f1@sens90']

def bootstrap(frame, results, names, reps=N_BOOT, seed=SEED, compare=()):
    rng = np.random.default_rng(seed); y = frame.y.values
    groups = [np.where(((frame.site + '_' + frame.y.astype(str)) == g).values)[0]
              for g in sorted(set(frame.site + '_' + frame.y.astype(str)))]
    samples = {m: {k: [] for k in KEY} for m in names}
    diffs = {(a, b): {k: [] for k in KEY} for a, b in compare}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        cache = {}
        for m in set(names) | {x for pair in compare for x in pair}:
            r = results[m]
            cache[m] = all_metrics(y[idx], dict(p=r['p'][idx], pred_youden=r['pred_youden'][idx], pred_sens90=r['pred_sens90'][idx]))
        for m in names:
            for k in KEY: samples[m][k].append(cache[m][k])
        for a, b in compare:
            for k in KEY: diffs[(a, b)][k].append(cache[a][k] - cache[b][k])
    def interval(k, v, pt):
        q = np.nanpercentile(v, [2.5, 97.5])
        if k.startswith('ece'):        # ECE is biased upward under resampling, so bootstrap intervals are not valid: report the point only
            return np.array([np.nan, np.nan])
        return q
    full = {m: all_metrics(y, results[m]) for m in names}
    ci = {m: {k: interval(k, v, full[m][k]) for k, v in samples[m].items()} for m in names}
    dci = {pair: {k: (np.nanmean(v), *np.nanpercentile(v, [2.5, 97.5])) for k, v in dd.items()} for pair, dd in diffs.items()}
    return ci, dci

def fmt_ci(v, ci): return f'{v:.3f}' if np.isnan(ci[0]) else f'{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})'
''')

md('## 4. Primary results: repeated, nested cross-validation (all sites)')
code(r'''
t0 = time.time()
cv = run_cv(Z, MODELS)
t1 = time.time()
print(f'CV done in {(t1 - t0) / 60:.1f} min')
y = Z.y.values
point = {m: all_metrics(y, cv[m]) for m in ORDER}
COMPARE = [('Hybrid', RAW), ('Hybrid', 'Features + age'), ('Hybrid', 'Ensemble + age'), ('Hybrid', 'Ensemble + features'),
           (RAW, 'Features + age'), ('Ensemble + age', RAW), ('Features + age', 'Age')]
ci, dci = bootstrap(Z, cv, ORDER, compare=COMPARE)
rows = []
for m in ORDER:
    rows.append({'model': m, **{k: fmt_ci(point[m][k], ci[m][k]) for k in KEY}, 'n': point[m]['n'], 'idh_mutant': point[m]['idh_mutant']})
table = pd.DataFrame(rows)
table.to_csv(OUTPUT_DIR / 'primary_metrics.csv', index=False)
pd.DataFrame({'patient_id': Z.patient_id, 'site': Z.site, 'y': Z.y, 'grade_group': Z.grade_group, 'age': Z.age,
              **{f'p_{m}': cv[m]['p'] for m in ORDER}, **{f'call_{m}': cv[m]['pred_youden'] for m in ORDER}}
             ).to_csv(OUTPUT_DIR / 'cv_predictions.csv', index=False)
pd.DataFrame([{'model': m, **point[m]} for m in ORDER]).to_csv(OUTPUT_DIR / 'primary_metrics_raw.csv', index=False)
show = ['model', 'auroc', 'average_precision', 'brier', 'ece', 'calibration_slope', 'sensitivity@youden',
        'specificity@youden', 'ppv@youden', 'f1@youden', 'balanced_accuracy@youden', 'mcc@youden']
display(table[show])
display(table[['model', 'sensitivity@sens90', 'specificity@sens90', 'ppv@sens90', 'f1@sens90', 'npv@youden', 'macro_f1@youden', 'ece_adaptive']])
drows = [dict(comparison=f'{a} minus {b}', **{k: f'{v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})' for k, v in dci[(a, b)].items()
                                                 if k in ('auroc', 'average_precision', 'brier', 'ece', 'sensitivity@youden', 'specificity@youden', 'f1@youden')})
         for a, b in COMPARE]
diff_table = pd.DataFrame(drows); diff_table.to_csv(OUTPUT_DIR / 'paired_differences.csv', index=False)
display(diff_table)
raw_frozen = binary_metrics(y, (Z.probability.values >= FROZEN_ENSEMBLE_THRESHOLD).astype(int))
print('Raw ensemble at its frozen threshold (selected on these same data, optimistic):',
      {k: round(v, 3) for k, v in raw_frozen.items() if isinstance(v, float)})
''')

code(r'''
# Figures: ROC curves and reliability diagrams
plot_models = ['Features + age', RAW, 'Hybrid']
colors = {'Features + age': '#9a9a96', RAW: '#5b5b58', 'Hybrid': '#2a6fdb'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axes[0]
for m in plot_models:
    fpr, tpr, _ = roc_curve(y, cv[m]['p'])
    ax.plot(fpr, tpr, color=colors[m], lw=2 if m == 'Hybrid' else 1.4, label=f"{m}  AUROC {point[m]['auroc']:.3f}")
ax.plot([0, 1], [0, 1], color='#bbb', lw=.8, ls='--')
ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity'); ax.set_title('ROC (repeated nested CV)', fontsize=10)
ax.legend(fontsize=8, frameon=False, loc='lower right')
ax = axes[1]
edges = np.linspace(0, 1, 11)
for m in plot_models:
    p = cv[m]['p']; idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
    ax.plot(xs, ys, marker='o', ms=4, color=colors[m], lw=2 if m == 'Hybrid' else 1.4, label=f"{m}  ECE {point[m]['ece']:.3f}")
ax.plot([0, 1], [0, 1], color='#bbb', lw=.8, ls='--')
ax.set_xlabel('Mean predicted probability (bins with >= 10 patients)'); ax.set_ylabel('Observed IDH-mutant fraction')
ax.set_title('Calibration', fontsize=10); ax.legend(fontsize=8, frameon=False, loc='upper left')
for a in axes:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'roc_calibration.png', dpi=200); plt.show()
''')

md(r'''
## 5. Leave-one-site-out

Each fitted model is trained on two sites and tested on the third, with thresholds chosen inside the training sites.
UPENN has very few IDH-mutant cases, so its estimates are wide.
''')
code(r'''
loso_rows = []
for test in sorted(Z.site.unique()):
    tr, te = Z[Z.site != test], Z[Z.site == test]
    if te.y.sum() < 5: continue
    strat_tr = (tr.site + '_' + tr.y.astype(str)).values
    for m in ORDER:
        if m == RAW:
            p = te.probability.values; ty, ts = youden_threshold(tr.y.values, tr.probability.values), sens_threshold(tr.y.values, tr.probability.values)
        else:
            Xtr, Xte = tr[MODELS[m]].values, te[MODELS[m]].values
            p = fit(Xtr, tr.y.values).predict_proba(Xte)[:, 1]
            ty, ts = inner_thresholds(Xtr, tr.y.values, strat_tr, SEED)
        r = all_metrics(te.y.values, dict(p=p, pred_youden=(p >= ty).astype(int), pred_sens90=(p >= ts).astype(int)))
        loso_rows.append(dict(test_site=test, model=m, **{k: r[k] for k in ['n', 'idh_mutant', 'auroc', 'average_precision', 'brier', 'ece',
                              'calibration_slope', 'sensitivity@youden', 'specificity@youden', 'f1@youden', 'mcc@youden']}))
loso = pd.DataFrame(loso_rows); loso.to_csv(OUTPUT_DIR / 'leave_one_site_out.csv', index=False)
display(loso.pivot(index='model', columns='test_site', values='auroc').loc[ORDER].round(3))
display(loso.pivot(index='model', columns='test_site', values='ece').loc[ORDER].round(3))
''')

md(r'''
## 6. Where does each component help?

**6a Subgroups.** AUROC of each model (from the primary CV predictions) within each subgroup, plus the paired difference
Hybrid minus raw ensemble. Subgroups with fewer than 5 patients in either class are skipped.
''')
code(r'''
Z['enhancement'] = np.where(Z.no_enhancement == 1, 'non-enhancing (<0.5 mL)', 'enhancing')
Z['age_band'] = pd.cut(Z.age, [0, 40, 60, 120], right=False, labels=['<40', '40-59', '>=60']).astype(str)
Z['volume_tertile'] = pd.qcut(Z.vol_whole, 3, labels=['small', 'medium', 'large']).astype(str)
Z['mask_source'] = Z.mask_source.fillna('unknown')
SUB_MODELS = ['Age', 'Features + age', RAW, 'Ensemble + age', 'Hybrid']
rng_s = np.random.default_rng(SEED + 7); sub_rows = []
for var in ['site', 'grade_group', 'enhancement', 'age_band', 'volume_tertile', 'mask_source']:
    for lvl, g in Z.groupby(var):
        yy = g.y.values
        if yy.sum() < 5 or (len(yy) - yy.sum()) < 5: continue
        ix = g.index.values
        row = dict(variable=var, level=lvl, n=len(g), idh_mutant=int(yy.sum()))
        for m in SUB_MODELS: row[m] = roc_auc_score(yy, cv[m]['p'][ix])
        diffs_ = []
        i1, i0 = np.where(yy == 1)[0], np.where(yy == 0)[0]
        for _ in range(1000):
            b = np.concatenate([rng_s.choice(i1, len(i1)), rng_s.choice(i0, len(i0))])
            diffs_.append(roc_auc_score(yy[b], cv['Hybrid']['p'][ix][b]) - roc_auc_score(yy[b], cv[RAW]['p'][ix][b]))
        row['hybrid_minus_ensemble'] = f"{np.mean(diffs_):+.3f} ({np.percentile(diffs_, 2.5):+.3f} to {np.percentile(diffs_, 97.5):+.3f})"
        row['features+age_minus_ensemble'] = row['Features + age'] - row[RAW]
        sub_rows.append(row)
subgroups = pd.DataFrame(sub_rows); subgroups.to_csv(OUTPUT_DIR / 'subgroup_auroc.csv', index=False)
display(subgroups.round(3))
''')

code(r'''
# Subgroup chart: AUROC of features+age, raw ensemble and hybrid per subgroup
sg = subgroups.copy(); sg['label'] = sg.variable + ': ' + sg.level.astype(str) + '  (n=' + sg.n.astype(str) + ', mut=' + sg.idh_mutant.astype(str) + ')'
fig, ax = plt.subplots(figsize=(8, 0.34 * len(sg) + 1.2))
yy_ = np.arange(len(sg))[::-1]
for m, c, mk in [('Features + age', '#9a9a96', 'o'), (RAW, '#5b5b58', 's'), ('Hybrid', '#2a6fdb', 'D')]:
    ax.scatter(sg[m], yy_, color=c, marker=mk, s=26, label=m, zorder=3)
lo_, hi_ = sg[['Features + age', RAW, 'Hybrid']].min(axis=1).values, sg[['Features + age', RAW, 'Hybrid']].max(axis=1).values
ax.hlines(yy_, lo_, hi_, color='#ddd', lw=1, zorder=1)
ax.set_yticks(yy_); ax.set_yticklabels(sg.label, fontsize=8); ax.set_xlabel('AUROC within subgroup')
ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.06), ncol=3)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'subgroup_auroc.png', dpi=200); plt.show()
''')

md(r'''
**6b Discordance.** Uses each model's nested Youden calls. Where the ensemble and features + age disagree, which one is
right, what do those patients look like, and what does the hybrid do with them?
''')
code(r'''
e_ok = cv[RAW]['pred_youden'] == y; f_ok = cv['Features + age']['pred_youden'] == y; h_ok = cv['Hybrid']['pred_youden'] == y
groups_ = {'both right': e_ok & f_ok, 'only ensemble right': e_ok & ~f_ok, 'only features+age right': ~e_ok & f_ok, 'both wrong': ~e_ok & ~f_ok}
disc = []
for name, msk in groups_.items():
    g = Z[msk]
    disc.append(dict(group=name, patients=int(msk.sum()), idh_mutant=int(g.y.sum()), hybrid_right=int(h_ok[msk].sum()),
                     median_age=g.age.median(), grade_2_3_share=(g.grade_group == 'grade 2-3').mean(),
                     non_enhancing_share=(g.no_enhancement == 1).mean(), median_enhancing_frac=g.enhancing_frac_whole.median(),
                     median_mismatch_index=g.mismatch_index.median(),
                     sites=', '.join(f'{k}:{v}' for k, v in g.site.value_counts().items())))
discordance = pd.DataFrame(disc); discordance.to_csv(OUTPUT_DIR / 'discordance.csv', index=False)
display(discordance.round(3))
''')

md('**6c Net reclassification** of the hybrid over the raw ensemble, at each model\'s nested Youden threshold.')
code(r'''
def nri(y, a, b):            # a = new model calls, b = old model calls
    ev, ne = y == 1, y == 0
    up_ev = ((a == 1) & (b == 0) & ev).sum() - ((a == 0) & (b == 1) & ev).sum()
    up_ne = ((a == 0) & (b == 1) & ne).sum() - ((a == 1) & (b == 0) & ne).sum()
    return up_ev / ev.sum(), up_ne / ne.sum()
a_, b_ = cv['Hybrid']['pred_youden'], cv[RAW]['pred_youden']
ev, ne = nri(y, a_, b_)
rng_n = np.random.default_rng(SEED + 3); bs = []
i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
for _ in range(N_BOOT):
    bidx = np.concatenate([rng_n.choice(i1, len(i1)), rng_n.choice(i0, len(i0))]); bs.append(nri(y[bidx], a_[bidx], b_[bidx]))
bs = np.array(bs)
nri_table = pd.DataFrame([dict(component='IDH-mutant (events)', nri=ev, ci_low=np.percentile(bs[:, 0], 2.5), ci_high=np.percentile(bs[:, 0], 97.5)),
                          dict(component='IDH-wildtype (non-events)', nri=ne, ci_low=np.percentile(bs[:, 1], 2.5), ci_high=np.percentile(bs[:, 1], 97.5)),
                          dict(component='Total', nri=ev + ne, ci_low=np.percentile(bs.sum(1), 2.5), ci_high=np.percentile(bs.sum(1), 97.5))])
nri_table.to_csv(OUTPUT_DIR / 'nri_hybrid_vs_ensemble.csv', index=False); display(nri_table.round(3))
''')

md('**6d What the hybrid relies on.** Standardised coefficients from the hybrid fitted on all patients, with bootstrap CIs.')
code(r'''
Xh = Z[MODELS['Hybrid']].values
coef = fit(Xh, y)[-1].coef_[0]
rng_c = np.random.default_rng(SEED + 5); cb = []
for _ in range(500):
    b = rng_c.choice(len(y), len(y)); cb.append(fit(Xh[b], y[b])[-1].coef_[0])
cb = np.array(cb)
coefs = pd.DataFrame(dict(input=MODELS['Hybrid'], std_coef=coef, ci_low=np.percentile(cb, 2.5, 0), ci_high=np.percentile(cb, 97.5, 0),
                          odds_ratio_per_sd=np.exp(coef))).sort_values('std_coef', key=np.abs, ascending=False)
coefs.to_csv(OUTPUT_DIR / 'hybrid_coefficients.csv', index=False); display(coefs.round(3))
''')

md('**6e Decision curves.** Net benefit across threshold probabilities, with treat-all and treat-none as references.')
code(r'''
ths = np.linspace(.02, .6, 59); prev = y.mean()
fig, ax = plt.subplots(figsize=(6.5, 4))
for m, c in [('Features + age', '#9a9a96'), (RAW, '#5b5b58'), ('Hybrid', '#2a6fdb')]:
    p = cv[m]['p']
    nb = [((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in ths]
    ax.plot(ths, nb, color=c, lw=2 if m == 'Hybrid' else 1.4, label=m)
ax.plot(ths, [prev - (1 - prev) * t / (1 - t) for t in ths], color='#ccc', lw=1, ls='--', label='treat all as mutant')
ax.axhline(0, color='#888', lw=.8); ax.set_ylim(-.02, prev + .03)
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.legend(fontsize=8, frameon=False)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'decision_curves.png', dpi=200); plt.show()
''')

md(r'''
## 7. Exploratory: does diffusion and perfusion add to the hybrid?

UCSF and UPENN only, same repeated nested CV on the patients who have the maps. Perfusion is ASL at UCSF and DSC at
UPENN, each normalised and z-scored within site.
''')
code(r'''
adv_cols = [c for c in ADV if c in Z]
sub = Z.dropna(subset=adv_cols).reset_index(drop=True)
if sub.site.nunique() >= 1 and sub.y.sum() >= 20:
    M7 = {'Hybrid': MODELS['Hybrid'], 'Hybrid + DWI/PWI': MODELS['Hybrid'] + adv_cols, 'Features + age + DWI/PWI': MODELS['Features + age'] + adv_cols}
    cv7 = run_cv(sub, M7)
    y7 = sub.y.values; pt7 = {m: all_metrics(y7, cv7[m]) for m in list(M7) + [RAW]}
    ci7, dci7 = bootstrap(sub, cv7, list(M7) + [RAW], compare=[('Hybrid + DWI/PWI', 'Hybrid')])
    adv_table = pd.DataFrame([{'model': m, **{k: fmt_ci(pt7[m][k], ci7[m][k]) for k in ['auroc', 'average_precision', 'brier', 'ece', 'sensitivity@youden', 'specificity@youden', 'f1@youden']},
                               'n': pt7[m]['n'], 'idh_mutant': pt7[m]['idh_mutant']} for m in list(M7) + [RAW]])
    display(adv_table); adv_table.to_csv(OUTPUT_DIR / 'advanced_imaging_subset.csv', index=False)
    print('Hybrid + DWI/PWI minus Hybrid:', {k: tuple(round(x, 3) for x in v) for k, v in dci7[('Hybrid + DWI/PWI', 'Hybrid')].items() if k in ('auroc', 'brier', 'ece')})
else:
    adv_table = None; print('Too few patients with diffusion and perfusion')
''')

md(r'''
## 8. Freeze the models for external validation

Each model is fitted on all development patients. The coefficients, scaling, feature list and the median nested
threshold (Youden and 90%-sensitivity) are saved to `frozen_models.json`, which `IDH_External_Validation.ipynb` loads
unchanged. External imaging features must be z-scored within the external cohort, the same way as each development site.
''')
code(r'''
frozen = dict(created=time.strftime('%Y-%m-%d %H:%M'), feature_protocol='imaging features z-scored within cohort; age in years; dl_logit = logit(ensemble probability)',
              ensemble='CA42 + CA44 + MC42, equal weights', models={})
for m, cols_ in MODELS.items():
    pipe = fit(Z[cols_].values, y); sc, lr = pipe[0], pipe[-1]
    th = cv[m]['thresholds']
    frozen['models'][m] = dict(inputs=cols_, scaler_mean=sc.mean_.tolist(), scaler_scale=sc.scale_.tolist(),
                               coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                               threshold_youden=float(np.median(th[:, 0])), threshold_sens90=float(np.median(th[:, 1])))
th = cv[RAW]['thresholds']
frozen['models'][RAW] = dict(inputs=['probability'], threshold_youden=float(np.median(th[:, 0])), threshold_sens90=float(np.median(th[:, 1])))
(OUTPUT_DIR / 'frozen_models.json').write_text(json.dumps(frozen, indent=2))
print('Saved', OUTPUT_DIR / 'frozen_models.json')
''')

md('## 9. Report')
code(r'''
def to_md(frame):
    cell = lambda v: '' if (isinstance(v, float) and np.isnan(v)) else (f'{v:.3f}' if isinstance(v, float) else str(v))
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '---|' * len(frame.columns)] +
                     ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in frame.itertuples(index=False)])
lines = ['# Hybrid IDH model: results', '', f'Run: {time.strftime("%Y-%m-%d %H:%M")}', '',
         f'Cohort: {len(Z)} patients ({int(y.sum())} IDH-mutant) with features, age and ensemble OOF predictions.', '',
         '## Primary metrics (repeated nested 5-fold CV; 95% CI)', '', to_md(table[show]), '',
         '## High-sensitivity operating point (90% target, chosen in training folds)', '',
         to_md(table[['model', 'sensitivity@sens90', 'specificity@sens90', 'ppv@sens90', 'f1@sens90']]), '',
         '## Paired differences (bootstrap mean and 95% CI)', '', to_md(diff_table), '',
         '## Leave-one-site-out AUROC', '', to_md(loso.pivot(index='model', columns='test_site', values='auroc').loc[ORDER].reset_index()), '',
         '## Leave-one-site-out ECE', '', to_md(loso.pivot(index='model', columns='test_site', values='ece').loc[ORDER].reset_index()), '',
         '## Subgroups (AUROC)', '', to_md(subgroups), '',
         '## Discordance: ensemble vs features + age', '', to_md(discordance), '',
         '## Net reclassification: hybrid vs raw ensemble', '', to_md(nri_table), '',
         '## Hybrid coefficients (standardised)', '', to_md(coefs), '']
if adv_table is not None:
    lines += ['## Exploratory: adding diffusion and perfusion (UCSF + UPENN)', '', to_md(adv_table), '']
lines += ['## Caveats', '',
          '- Ensemble members and its original threshold were chosen on these out-of-fold results: ensemble-based estimates are optimistic.',
          '- Leave-one-site-out refits the combination only; the ensemble logit comes from pooled-site CV.',
          '- Internal evidence only; see the external validation notebook.',
          '- Age carries much of the signal; report image-only models (Features, Ensemble + features) alongside the hybrid.']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:30]))
''')

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'IDH_Hybrid_Analysis.ipynb'
nbf.write(nb, out); print('Wrote', out)
