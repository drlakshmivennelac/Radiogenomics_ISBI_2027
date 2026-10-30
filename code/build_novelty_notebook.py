"""Build IDH_Hybrid_Novelty.ipynb: four analyses that ask what the deep-learning ensemble adds beyond age and radiology.

A. Is the ensemble learning age?           (CPU, minutes)
B. Disagreement-gated deferral             (CPU, minutes)
C. Where the ensemble looks (XRAI regions) (CPU, ~10 min for 132 saved maps)
D. Embedding probes                        (GPU, ~15-45 min, resumable)

Parts 0-B reuse the cohort, features and nested-CV code of IDH_Hybrid_Analysis.ipynb verbatim, so the hybrid's
cross-validated predictions are reproduced exactly (same seeds). Run: python build_novelty_notebook.py [output.ipynb]
"""
import sys, subprocess, tempfile
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
with tempfile.TemporaryDirectory() as tmp:
    hyb = Path(tmp) / 'hyb.ipynb'
    subprocess.run([sys.executable, str(HERE / 'build_hybrid_notebook.py'), str(hyb)], check=True, stdout=subprocess.DEVNULL)
    H = nbf.read(hyb, 4)

def after(prefix):
    for i, c in enumerate(H.cells):
        if c.cell_type == 'markdown' and c.source.lstrip().startswith(prefix):
            return H.cells[i + 1].source
    raise KeyError(prefix)

CONFIG = after('## 1.').replace("OUTPUT_DIR = PROJECT / 'outputs' / 'idh_hybrid'", "OUTPUT_DIR = PROJECT / 'outputs' / 'idh_novelty'")
assert 'idh_novelty' in CONFIG
SETUP = after('## 2.')
HELPERS = after('## 3.')

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# What does the deep-learning ensemble add beyond age and radiology?

Four analyses for the hybrid IDH paper. Each asks a question that fusion papers usually skip.

| Part | Question | Data used | Runtime (RTX 5080) |
|---|---|---|---|
| A | Is the ensemble partly an age detector? How much does each component know that the others do not? | cross-validated predictions | 1-3 min |
| B | Can disagreement between the ensemble and features + age decide which patients go to a radiologist? | cross-validated predictions | 1-3 min |
| C | Where does the ensemble look: enhancing tumour, non-enhancing tumour, oedema, or outside the lesion? | the 132 saved XRAI maps | 5-15 min, CPU |
| D | Does the network's internal representation encode the radiological features, age and site? | GPU forward pass, 1,435 patients x 3 members | 15-45 min, resumable |

**Run order.** Run top to bottom. Parts A-C need no GPU. Part D is switched by `RUN_PROBES`; it saves embeddings per
checkpoint, so an interrupted run resumes where it stopped. Restart the kernel after a CUDA error.

**What is pre-specified.** The tests marked *pre-declared* below (A: within-class age correlation, age-matched AUROC;
C1-C4) were written before seeing any results. Part B's conflict score was refined after a test run, so B is exploratory. Everything else is
exploratory and is labelled so in the report.

**Caveats.** All results are internal (pooled-site cross-validation). The XRAI sample is outcome-enriched
(33 each of TP, TN, FP, FN), so it can compare groups but cannot estimate population frequencies. Probe accuracy shows
that information is linearly decodable from the representation, not that the IDH head uses it.
''')

md('## 1. Configuration')
code(CONFIG + r'''

# ---- added for this notebook ----
XRAI_DIR = PROJECT / 'outputs' / 'selected_three_xrai_balanced_10pct'   # saved three-member XRAI maps
AUDIT_CSV = None          # radiologist feature audit, once available: a CSV with patient_id (or review_id + AUDIT_KEY)
AUDIT_KEY = None          # optional CSV mapping review_id -> patient_id (from the blinded-review private folder)
RUN_PROBES = True         # Part D (GPU)
INPUT_CACHE = Path.home() / 'Downloads' / 'outputs' / 'novelty_input_cache'   # preprocessed crops (~16 MB/patient), kept OUT of OneDrive; delete afterwards
RANDOM_INIT_SEED = 0      # untrained-network control for Part D
PATH_REMAP = {r'C:\Users\claks\Downloads\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5': r'C:\Users\claks\Downloads\UCSF-PDGM'}
AGE_BANDS = [0, 40, 55, 200]            # pre-declared: <40, 40-54, >=55 years
MATCH_CALIPER, MATCH_K = 2.0, 3         # age matching: same site, |age difference| <= 2 years, up to 3 wildtype per mutant
RING_MM = 10.0                          # peritumoral ring for the XRAI analysis
N_BOOT_FAST = 1000
''')

md('## 2. Cohort (identical to the hybrid notebook)')
code(SETUP)
code(HELPERS)

md(r'''
## 3. Reproduce the hybrid's cross-validated predictions

Same seeds and code as `IDH_Hybrid_Analysis.ipynb`, so the AUROCs below should equal that notebook's Table 1
(Hybrid 0.961). If they differ, the inputs changed since that run. The predictions are saved as `cv_predictions.csv`.
''')
code(r'''
from scipy.stats import spearmanr, mannwhitneyu, rankdata, chi2 as chi2_dist
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
t0 = time.time()
y = Z.y.values
cv = run_cv(Z, MODELS)
pd.DataFrame({'patient_id': Z.patient_id, 'site': Z.site, 'y': y, 'age': Z.age,
              **{f'p[{m}]': cv[m]['p'] for m in ORDER}, **{f'call[{m}]': cv[m]['pred_youden'] for m in ORDER}}
             ).to_csv(OUTPUT_DIR / 'cv_predictions.csv', index=False)
AUC0 = pd.Series({m: roc_auc_score(y, cv[m]['p']) for m in ORDER}).round(3)
display(AUC0.to_frame('AUROC'))
print(f'{len(Z)} patients, {int(y.sum())} IDH-mutant | {time.time() - t0:.0f}s')

STRAT = (Z.site + '_' + Z.y.astype(str)).values
GROUPS = [np.where(STRAT == g)[0] for g in np.unique(STRAT)]
ALL = np.arange(len(Z))
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def safe_auc(yy, pp):
    return roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
def boot_idx(rng):
    return np.concatenate([rng.choice(g, len(g)) for g in GROUPS])
def boot_ci(stat, reps=N_BOOT, seed=SEED, sampler=None):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        i = boot_idx(rng) if sampler is None else sampler(rng)
        try: vals.append(stat(i))
        except ValueError: pass
    vals = np.asarray(vals, float)
    return np.nanpercentile(vals, [2.5, 97.5]) if np.isfinite(vals).any() else np.array([np.nan, np.nan])
def fmt(v, ci, nd=3):
    return 'n/a' if not np.isfinite(v) else (f'{v:.{nd}f}' if not np.isfinite(ci).all() else f'{v:.{nd}f} ({ci[0]:.{nd}f} to {ci[1]:.{nd}f})')
def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join('' if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r) + ' |' for r in df.values]
    return '\n'.join(lines)
REPORT = {}
L = Z.dl_logit.values; AGE = Z.age.values
F_LOGIT = logit(cv['Features']['p'])
''')

# ---------------------------------------------------------------- Part A
md(r'''
## Part A. Is the ensemble learning age?

Mutants are younger, so any IDH predictor correlates with age across all patients. The informative test is **within
each IDH class**: if the ensemble's output still tracks age among wildtype patients (or among mutants), it has learned
age-related appearance beyond the mutation itself (*pre-declared*). The handcrafted-feature score is shown alongside as
a comparator.

**A1** Spearman correlation with age. **A2** How much of the ensemble's output is explained by age, site and the
handcrafted features (cross-validated R²). **A3** AUROC within age bands. **A4** AUROC in an age-matched sample, where
age carries no information by design (*pre-declared*). **A5** What each component knows that the other two do not.
''')
code(r'''
WT, MU = y == 0, y == 1
def rho(a, b): return lambda i: spearmanr(a[i], b[i])[0]
def rho_in(a, b, mask): return lambda i: spearmanr(a[i][mask[i]], b[i][mask[i]])[0]
def pooled_rho(a, b):
    def f(i):
        zs, ws = [], []
        for g in np.unique(STRAT[i]):
            k = i[STRAT[i] == g]
            if len(k) > 8 and np.std(b[k]) > 0:
                r = spearmanr(a[k], b[k])[0]; zs.append(np.arctanh(np.clip(r, -.999, .999))); ws.append(len(k) - 3)
        return float(np.tanh(np.average(zs, weights=ws)))
    return f
rows = []
for label, mk in [('all patients', lambda a: rho(a, AGE)), ('IDH-wildtype only', lambda a: rho_in(a, AGE, WT)),
                  ('IDH-mutant only', lambda a: rho_in(a, AGE, MU)), ('within site x IDH class (pooled)', lambda a: pooled_rho(a, AGE))]:
    row = {'population': label}
    for name, a in [('ensemble logit', L), ('features score', F_LOGIT)]:
        f = mk(a); row[f'rho, {name}'] = fmt(f(ALL), boot_ci(f, N_BOOT_FAST))
    rows.append(row)
A1 = pd.DataFrame(rows); display(A1); A1.to_csv(OUTPUT_DIR / 'A1_age_correlation.csv', index=False)
print('Negative rho = higher predicted mutant probability in younger patients.')
REPORT['A1. Correlation of each component with age (Spearman, 95% CI)'] = A1
''')
code(r'''
SITE_D = pd.get_dummies(Z.site, drop_first=True).values.astype(float)
SETS = {'age': Z[['age']].values, 'site': SITE_D, 'handcrafted features': Z[CONV + MISM].values,
        'age + features': Z[['age'] + CONV + MISM].values, 'age + features + site': np.c_[Z[['age'] + CONV + MISM].values, SITE_D]}
def cv_r2(X, t, idx):
    pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), X[idx], t[idx],
                             cv=KFold(5, shuffle=True, random_state=SEED))
    return 1 - np.sum((t[idx] - pred) ** 2) / np.sum((t[idx] - t[idx].mean()) ** 2)
A2 = pd.DataFrame([{'predictors': k, **{pop: round(cv_r2(X, L, np.where(m)[0]), 3)
                    for pop, m in [('all', np.ones(len(y), bool)), ('wildtype only', WT), ('mutant only', MU)]}}
                   for k, X in SETS.items()])
display(A2); A2.to_csv(OUTPUT_DIR / 'A2_ensemble_variance_explained.csv', index=False)
print('Cross-validated R2 of the ensemble logit. Within-class values show what the ensemble encodes besides IDH status.')
REPORT['A2. Share of the ensemble output explained by age, site and features (cross-validated R²)'] = A2
''')
code(r'''
band = pd.cut(Z.age, AGE_BANDS, right=False, labels=['<40', '40-54', '>=55']).astype(str).values
A_MODELS = ['Age', 'Features', RAW, 'Features + age', 'Hybrid']
rows = []
for b in ['<40', '40-54', '>=55']:
    k = np.where(band == b)[0]; row = {'age band': b, 'n': len(k), 'mutant': int(y[k].sum())}
    for m in A_MODELS:
        if y[k].sum() < 5 or (1 - y[k]).sum() < 5: row[m] = 'too few'; continue
        sub = [g[np.isin(g, k)] for g in GROUPS]; sub = [s for s in sub if len(s)]
        sampler = lambda rng, sub=sub: np.concatenate([rng.choice(s, len(s)) for s in sub])
        f = lambda i, m=m: safe_auc(y[i], cv[m]['p'][i])
        row[m] = fmt(f(k), boot_ci(f, N_BOOT_FAST, sampler=sampler))
    rows.append(row)
A3 = pd.DataFrame(rows); display(A3); A3.to_csv(OUTPUT_DIR / 'A3_auroc_by_age_band.csv', index=False)
REPORT['A3. AUROC within age bands (95% CI)'] = A3
''')
code(r'''
rng = np.random.default_rng(SEED); used = set(); sets = []
mut = np.where(MU)[0].copy(); rng.shuffle(mut); sites = Z.site.values
for m in mut:
    cand = np.where(WT & (sites == sites[m]) & (np.abs(AGE - AGE[m]) <= MATCH_CALIPER))[0]
    cand = sorted((c for c in cand if c not in used), key=lambda c: (abs(AGE[c] - AGE[m]), c))[:MATCH_K]
    if cand: used.update(cand); sets.append(np.array([m] + list(cand)))
MATCHED = np.concatenate(sets)
sampler = lambda rng: np.concatenate([sets[j] for j in rng.integers(0, len(sets), len(sets))])
rows = []
for m in ['Age', 'Features', RAW, 'Features + age', 'Ensemble + age', 'Hybrid']:
    f = lambda i, m=m: safe_auc(y[i], cv[m]['p'][i])
    rows.append({'model': m, 'AUROC, age-matched (95% CI)': fmt(f(MATCHED), boot_ci(f, N_BOOT_FAST, sampler=sampler)),
                 'AUROC, all patients': round(safe_auc(y, cv[m]['p']), 3)})
A4 = pd.DataFrame(rows); display(A4); A4.to_csv(OUTPUT_DIR / 'A4_age_matched_auroc.csv', index=False)
dage = np.mean([np.mean(np.abs(AGE[s[1:]] - AGE[s[0]])) for s in sets])
A4_NOTE = (f'{len(sets)} of {int(MU.sum())} mutants matched to {len(MATCHED) - len(sets)} wildtype patients '
           f'(same site, |age difference| <= {MATCH_CALIPER:g} y, mean difference {dage:.2f} y). '
           'Age AUROC near 0.5 confirms that age carries no information in this sample.')
print(A4_NOTE)
REPORT['A4. AUROC in an age-matched sample (cluster bootstrap over matched sets)'] = A4
''')
code(r'''
def cross_resid(target, X, seed=SEED):
    res = np.zeros(len(target))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, STRAT):
        res[te] = target[te] - LinearRegression().fit(X[tr], target[tr]).predict(X[te])
    return res
UNIQ = {'ensemble, beyond age + features': (L, cross_resid(L, Z[['age'] + CONV + MISM].values)),
        'features, beyond ensemble + age': (F_LOGIT, cross_resid(F_LOGIT, np.c_[L, AGE])),
        'age (younger = mutant), beyond ensemble + features': (-AGE, cross_resid(-AGE, np.c_[L, F_LOGIT]))}
rows = []
for k, (raw_, res_) in UNIQ.items():
    f = lambda i, r=res_: safe_auc(y[i], r[i])
    rows.append({'component': k, 'AUROC on its own': round(safe_auc(y, raw_), 3),
                 'AUROC of what is left (95% CI)': fmt(f(ALL), boot_ci(f, N_BOOT_FAST))})
def loglik(cols):
    X = StandardScaler().fit_transform(Z[cols].values)
    p = np.clip(LogisticRegression(penalty=None, max_iter=5000).fit(X, y).predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
for (k, base, add) in [('ensemble, beyond age + features', CONV + MISM + ['age'], ['dl_logit']),
                       ('features, beyond ensemble + age', ['dl_logit', 'age'], CONV + MISM),
                       ('age (younger = mutant), beyond ensemble + features', ['dl_logit'] + CONV + MISM, ['age'])]:
    stat = 2 * (loglik(base + add) - loglik(base))
    r = next(r for r in rows if r['component'] == k)
    r['likelihood-ratio chi2 (df)'] = f'{stat:.1f} ({len(add)})'; r['p'] = f'{chi2_dist.sf(stat, len(add)):.1e}'
A5 = pd.DataFrame(rows); display(A5); A5.to_csv(OUTPUT_DIR / 'A5_unique_information.csv', index=False)
print('"What is left" = cross-fitted residual after linear regression on the other two components. AUROC > 0.5 means '
      'the component carries IDH information the others lack. Likelihood-ratio tests are in-sample and descriptive.')
REPORT['A5. Information unique to each component'] = A5
''')

# ---------------------------------------------------------------- Part B
md(r'''
## Part B. Disagreement-gated deferral

Grade is not known before surgery, so the subgroup finding "the ensemble is better in grade 2-3" cannot route
patients. Disagreement between the two components can. Each patient gets three scores, all available before surgery:

* **hybrid confidence**: distance of the hybrid's log-odds from its own nested-CV Youden threshold;
* **conflict**: -tanh(z_E/2) x tanh(z_F/2), where z is each component's log-odds distance from its own threshold.
  It is near +1 when the ensemble and features + age are both confident and on opposite sides, near 0 when either is
  unsure, and near -1 when both confidently agree;
* **combined**: mean rank of low confidence and high conflict.

Patients with the highest deferral score go to a radiologist. The **risk-coverage curve** plots the hybrid's error
rate among the patients it keeps. **AURC** is the mean error over 50-100% coverage (lower is better). The main
comparison is combined versus confidence alone: does conflict between the components add anything confidence lacks?
(Exploratory: the conflict score was defined after a test run.)
''')
code(r'''
th = lambda m: float(np.mean(cv[m]['thresholds'][:, 0]))
zH, zE, zF = (logit(cv[m]['p']) - logit(th(m)) for m in ('Hybrid', RAW, 'Features + age'))
CALL = cv['Hybrid']['pred_youden']; ERR = (CALL != y).astype(float)
CONF = np.abs(zH); DIS = -np.tanh(zE / 2) * np.tanh(zF / 2)
DIFFER = cv[RAW]['pred_youden'] != cv['Features + age']['pred_youden']
SCORES = {'low hybrid confidence': -CONF, 'component conflict': DIS,
          'combined (mean rank)': rankdata(-CONF) + rankdata(DIS)}
COV = np.round(np.arange(0.50, 1.0001, 0.01), 2)
def curve(score, idx):
    order = idx[np.argsort(score[idx], kind='stable')]
    cum = np.cumsum(ERR[order]); n = np.maximum(np.round(COV * len(idx)).astype(int), 1)
    return cum[n - 1] / n
def aurc(score, idx): return float(curve(score, idx).mean())
rows = [{'strategy': 'random deferral (expected)', 'AURC': round(ERR.mean(), 4)}]
for k, s in SCORES.items():
    rows.append({'strategy': k, 'AURC': round(aurc(s, ALL), 4),
                 'AUROC for flagging hybrid errors': round(safe_auc(ERR.astype(int), s), 3)})
B1 = pd.DataFrame(rows)
rng = np.random.default_rng(SEED + 7); diffs = {('combined (mean rank)', 'low hybrid confidence'): [],
                                                ('component conflict', 'low hybrid confidence'): []}
for _ in range(N_BOOT_FAST):
    i = boot_idx(rng)
    for (a, b) in diffs: diffs[(a, b)].append(aurc(SCORES[a], i) - aurc(SCORES[b], i))
B1b = pd.DataFrame([{'comparison': f'{a} minus {b}', 'AURC difference (95% CI)':
                     fmt(aurc(SCORES[a], ALL) - aurc(SCORES[b], ALL), np.percentile(v, [2.5, 97.5]), 4)} for (a, b), v in diffs.items()])
display(B1); display(B1b)
B1.to_csv(OUTPUT_DIR / 'B1_deferral_aurc.csv', index=False); B1b.to_csv(OUTPUT_DIR / 'B1_deferral_aurc_differences.csv', index=False)
REPORT['B1. Deferral strategies (AURC: mean hybrid error over 50-100% coverage; lower is better)'] = B1
REPORT['B1b. Paired bootstrap differences (negative = first strategy better)'] = B1b
''')
code(r'''
def at_deferral(score, frac):
    n_def = int(round(frac * len(y))); order = np.argsort(-score, kind='stable')
    dfr = np.zeros(len(y), bool); dfr[order[:n_def]] = True; keep = ~dfr
    bm = binary_metrics(y[keep], CALL[keep])
    return {'deferred': n_def, 'hybrid errors captured': f'{ERR[dfr].sum():.0f}/{ERR.sum():.0f} ({ERR[dfr].sum() / ERR.sum():.0%})',
            'mutants deferred': int(y[dfr].sum()), 'retained accuracy': round(1 - ERR[keep].mean(), 3),
            'retained sensitivity': round(bm['sensitivity'], 3), 'retained specificity': round(bm['specificity'], 3),
            'retained PPV': round(bm['ppv'], 3)}
base = binary_metrics(y, CALL)
rows = [{'strategy': 'no deferral', 'deferral': '0%', 'deferred': 0, 'hybrid errors captured': f'0/{ERR.sum():.0f}',
         'mutants deferred': 0, 'retained accuracy': round(1 - ERR.mean(), 3), 'retained sensitivity': round(base['sensitivity'], 3),
         'retained specificity': round(base['specificity'], 3), 'retained PPV': round(base['ppv'], 3)}]
for frac in (0.10, 0.20, 0.30):
    for k, s in SCORES.items(): rows.append({'strategy': k, 'deferral': f'{frac:.0%}', **at_deferral(s, frac)})
B2 = pd.DataFrame(rows); display(B2); B2.to_csv(OUTPUT_DIR / 'B2_deferral_operating_points.csv', index=False)
print('Random deferral would capture errors in proportion to the deferral rate (10%, 20%, 30%).')
REPORT['B2. Operating points (hybrid at its nested Youden threshold)'] = B2

ez = lambda v: (v - v.mean()) / v.std()
Xe = np.c_[ez(-CONF), ez(DIS)]
def or_fit(i):
    return np.exp(LogisticRegression(penalty=None, max_iter=5000).fit(Xe[i], ERR[i]).coef_[0])
pt = or_fit(ALL); rng = np.random.default_rng(SEED + 8); bs = np.array([or_fit(boot_idx(rng)) for _ in range(N_BOOT_FAST)])
B3 = pd.DataFrame([{'predictor of a hybrid error (per SD)': k, 'odds ratio (95% CI)': fmt(pt[j], np.percentile(bs[:, j], [2.5, 97.5]), 2)}
                   for j, k in enumerate(['low hybrid confidence', 'component conflict'])])
B3c = pd.crosstab(pd.Series(np.where(DIFFER, 'components disagree', 'components agree'), name='binary calls'),
                  pd.Series(np.where(ERR == 1, 'hybrid wrong', 'hybrid right'), name='hybrid'))
display(B3); display(B3c)
B3.to_csv(OUTPUT_DIR / 'B3_error_model.csv', index=False)
REPORT['B3. Joint model of hybrid errors'] = B3
REPORT['B3c. Binary agreement of the components versus hybrid correctness'] = B3c.reset_index()

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.axhline(ERR.mean() * 100, color='#9a9a96', ls='--', lw=1, label='random deferral')
for (k, s), c in zip(SCORES.items(), ['#5b5b58', '#d98a1c', '#2a6fdb']):
    ax.plot(COV * 100, curve(s, ALL) * 100, color=c, lw=1.8, label=k)
ax.set_xlabel('patients kept by the model (%)'); ax.set_ylabel('hybrid error rate among kept (%)')
ax.invert_xaxis(); ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'B_risk_coverage.png', dpi=200); plt.show()
''')

# ---------------------------------------------------------------- Part C
md(r'''
## Part C. Where does the ensemble look?

Uses the saved three-member XRAI maps (signed integrated gradients of P(mutant), averaged across members, plus XRAI
region ranking). No GPU. For each case and sequence, the head support is split into **enhancing tumour**,
**non-enhancing core** (necrosis or non-enhancing tumour), **oedema / T2 hyperintensity**, a **10 mm peritumoral ring**,
and **distant** tissue in the crop. Label conventions follow the correlates notebook (BraTS 2017/2020/2023 by labels present).

Per region: *positive share* (fraction of all mutant-supporting attribution), *enrichment* (positive share / volume
share; >1 = more attribution than its size predicts), *signed density* (mean attribution / mean |attribution|; >0
supports mutant), and *top-10% XRAI composition*.

*Pre-declared* comparisons (two-sided Mann-Whitney, Holm-corrected):
* **C1** FLAIR enrichment of the non-enhancing lesion (core + oedema) is higher in mutants than wildtype.
* **C2** T1c signed density in enhancing tumour is higher (less wildtype-supporting) in mutants, among cases with enhancement.
* **C3** The share of attribution outside the lesion (ring + distant, both sequences) is higher in errors than in correct calls.
* **C4** Among mutants, FLAIR enrichment of the non-enhancing lesion is lower in false negatives than true positives.

The sample is outcome-enriched (33 per outcome), so group comparisons are valid but frequencies are not.
''')
code(r'''
from scipy import ndimage
import json as _json
REGIONS = ['enhancing', 'nonenh_core', 'oedema', 'ring', 'distant']
SEQS = ['T1c', 'FLAIR']
def region_masks(mask, support, voxel_mm):
    labels = set(int(v) for v in np.unique(mask)) - {0}
    enh, core = ((4,), (1, 3)) if 4 in labels else ((3,), (1,))
    lesion = mask > 0
    dist = ndimage.distance_transform_edt(~lesion, sampling=voxel_mm) if lesion.any() else np.full(mask.shape, np.inf)
    R = dict(enhancing=np.isin(mask, enh) & support, nonenh_core=np.isin(mask, core) & support,
             oedema=(mask == 2) & support, ring=(~lesion) & (dist <= RING_MM) & support,
             distant=(~lesion) & (dist > RING_MM) & support)
    return R, lesion
def case_metrics(folder):
    meta = _json.loads((folder / 'metadata.json').read_text())
    with np.load(folder / 'maps.npz', allow_pickle=False) as z:
        inputs, ig, xr, mask, aff = z['inputs'], z['ig'], z['xrai'], z['mask'], z['affine']
    support = (inputs != 0).any(0)
    R, lesion = region_masks(mask, support, float(abs(np.linalg.det(aff[:3, :3])) ** (1 / 3)))
    rec = dict(patient_id=meta['patient_id'], site=meta['site'], label=meta['label'], category=meta['category'],
               probability=meta['probability'], convergence_warning=meta.get('convergence_warning', False),
               lesion_voxels=int(lesion.sum()))
    vol = {r: R[r].sum() / support.sum() for r in REGIONS}
    absall = 0.0
    for s, name in enumerate(SEQS):
        a = ig[s]; pos = np.clip(a, 0, None); tot = pos[support].sum(); mabs = np.abs(a[support]).mean()
        absall += np.abs(a[support]).sum(); rec[f'{name}_abs_total'] = float(np.abs(a[support]).sum())
        top = support & (xr[s] >= np.quantile(xr[s][support], 0.9))
        for r in REGIONS:
            share = pos[R[r]].sum() / tot if tot > 0 else np.nan
            rec[f'{name}_share_{r}'] = share
            rec[f'{name}_enrich_{r}'] = share / vol[r] if vol[r] > 0 else np.nan
            rec[f'{name}_density_{r}'] = float(a[R[r]].mean() / mabs) if R[r].any() and mabs > 0 else np.nan
            rec[f'{name}_top10_{r}'] = (top & R[r]).sum() / max(top.sum(), 1)
        nl = R['nonenh_core'] | R['oedema']
        rec[f'{name}_share_nonenh_lesion'] = pos[nl].sum() / tot if tot > 0 else np.nan
        vnl = nl.sum() / support.sum()
        rec[f'{name}_enrich_nonenh_lesion'] = rec[f'{name}_share_nonenh_lesion'] / vnl if vnl > 0 else np.nan
    for r in REGIONS: rec[f'volume_share_{r}'] = vol[r]
    rec['flair_share_of_attribution'] = rec['FLAIR_abs_total'] / absall if absall > 0 else np.nan
    w = np.array([rec['T1c_abs_total'], rec['FLAIR_abs_total']]); w = w / w.sum() if w.sum() > 0 else np.array([.5, .5])
    rec['outside_lesion_share'] = sum(w[s] * (rec[f'{n}_share_ring'] + rec[f'{n}_share_distant']) for s, n in enumerate(SEQS))
    return rec
folders = sorted(p.parent for p in XRAI_DIR.glob('*/metadata.json')) if XRAI_DIR.exists() else []
print(f'{len(folders)} completed XRAI cases in {XRAI_DIR}')
t0 = time.time(); XR = []
for j, f in enumerate(folders):
    try: XR.append(case_metrics(f))
    except Exception as e: print('  skipped', f.name, type(e).__name__, e)
    if (j + 1) % 25 == 0: print(f'  {j + 1}/{len(folders)} | {time.time() - t0:.0f}s')
XR = pd.DataFrame(XR)
if len(XR):
    XR['mutant'] = XR.label.astype(int); XR['correct'] = XR.category.isin(['TP', 'TN'])
    XR.to_csv(OUTPUT_DIR / 'C_xrai_region_metrics.csv', index=False)
    print(XR.category.value_counts().to_dict(), '| no lesion (brain-crop fallback):', int((XR.lesion_voxels == 0).sum()),
          '| convergence warnings:', int(XR.convergence_warning.sum()))
''')
code(r'''
if len(XR):
    X_ = XR[XR.lesion_voxels > 0]
    def mw(a, b, label, alt_note):
        a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3: return dict(test=label, p=np.nan)
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        return {'test': label, 'group 1 median (IQR)': f'{np.median(a):.3f} ({np.percentile(a, 25):.3f}-{np.percentile(a, 75):.3f}), n={len(a)}',
                'group 2 median (IQR)': f'{np.median(b):.3f} ({np.percentile(b, 25):.3f}-{np.percentile(b, 75):.3f}), n={len(b)}',
                'P(group 1 > group 2)': round(u / (len(a) * len(b)), 3), 'p': p, 'hypothesis': alt_note}
    enh = X_.volume_share_enhancing > 0
    C = pd.DataFrame([
        mw(X_[X_.mutant == 1].FLAIR_enrich_nonenh_lesion, X_[X_.mutant == 0].FLAIR_enrich_nonenh_lesion,
           'C1 FLAIR enrichment, non-enhancing lesion: mutant vs wildtype', 'mutant higher'),
        mw(X_[(X_.mutant == 1) & enh].T1c_density_enhancing, X_[(X_.mutant == 0) & enh].T1c_density_enhancing,
           'C2 T1c signed density, enhancing tumour: mutant vs wildtype', 'mutant higher'),
        mw(X_[~X_.correct].outside_lesion_share, X_[X_.correct].outside_lesion_share,
           'C3 attribution outside the lesion: errors vs correct', 'errors higher'),
        mw(X_[X_.category == 'FN'].FLAIR_enrich_nonenh_lesion, X_[X_.category == 'TP'].FLAIR_enrich_nonenh_lesion,
           'C4 FLAIR enrichment, non-enhancing lesion: FN vs TP', 'FN lower')])
    ps = C.p.values; order = np.argsort(ps); holm = np.empty(len(ps)); run = 0
    for r, k in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[k])); holm[k] = run
    C['Holm p'] = holm; C['p'] = C.p.map(lambda v: f'{v:.4f}'); C['Holm p'] = C['Holm p'].map(lambda v: f'{v:.4f}')
    display(C); C.to_csv(OUTPUT_DIR / 'C1_pre_declared_tests.csv', index=False)
    REPORT['C1-C4. Pre-declared attribution tests (Mann-Whitney, Holm)'] = C

    rows = []
    for cat in ['TP', 'FN', 'TN', 'FP']:
        g = X_[X_.category == cat]; row = {'outcome': cat, 'n': len(g)}
        for k, lab in [('FLAIR_share_nonenh_lesion', 'FLAIR share, non-enh. lesion'), ('T1c_share_enhancing', 'T1c share, enhancing'),
                       ('outside_lesion_share', 'share outside lesion'), ('flair_share_of_attribution', 'FLAIR share of all attribution'),
                       ('FLAIR_top10_oedema', 'FLAIR top-10% in oedema'), ('T1c_top10_enhancing', 'T1c top-10% in enhancing')]:
            v = g[k].dropna(); row[lab] = f'{v.median():.2f} ({v.quantile(.25):.2f}-{v.quantile(.75):.2f})' if len(v) else ''
        rows.append(row)
    C5 = pd.DataFrame(rows); display(C5); C5.to_csv(OUTPUT_DIR / 'C5_attribution_by_outcome.csv', index=False)
    REPORT['C5. Attribution by outcome, median (IQR) (exploratory)'] = C5

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, s in zip(axes, SEQS):
        data, pos_, cols = [], [], []
        for j, r in enumerate(REGIONS):
            for d_, (lab, c) in enumerate([(1, '#2a6fdb'), (0, '#9a9a96')]):
                v = X_[X_.mutant == lab][f'{s}_enrich_{r}'].replace([np.inf, -np.inf], np.nan).dropna()
                data.append(np.log2(v.clip(lower=1e-3))); pos_.append(j * 3 + d_); cols.append(c)
        bp = ax.boxplot(data, positions=pos_, widths=.8, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], cols): patch.set_facecolor(c); patch.set_alpha(.6)
        ax.axhline(0, color='k', lw=.6); ax.set_xticks([j * 3 + .5 for j in range(len(REGIONS))])
        ax.set_xticklabels(['enh.', 'non-enh.\ncore', 'oedema', 'ring', 'distant'], fontsize=8); ax.set_title(s, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('log2 enrichment\n(attribution share / volume share)')
    axes[1].legend([bp['boxes'][0], bp['boxes'][1]], ['IDH-mutant', 'IDH-wildtype'], frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'C_attribution_enrichment.png', dpi=200); plt.show()
''')
code(r'''
# Link to the handcrafted features (exploratory): does the network look where the radiological signs are?
if len(XR):
    J = X_.merge(d[['patient_id'] + CONV + MISM], on='patient_id', how='inner')
    rows = []
    for a, b, why in [('FLAIR_enrich_nonenh_lesion', 'mismatch_index', 'network uses non-enhancing FLAIR where T2-FLAIR mismatch is strong?'),
                      ('FLAIR_share_nonenh_lesion', 'enhancing_frac_whole', 'less non-enhancing attention when the tumour enhances?'),
                      ('T1c_share_enhancing', 'enhancing_frac_whole', 'T1c attention follows enhancement?'),
                      ('outside_lesion_share', 'log_vol_whole', 'small lesions push attribution outside?')]:
        k = J[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        r_, p_ = spearmanr(k[a], k[b]) if len(k) > 5 else (np.nan, np.nan)
        rows.append({'attribution metric': a, 'handcrafted feature': b, 'n': len(k), 'Spearman rho': round(r_, 3), 'p': f'{p_:.4f}', 'question': why})
    C6 = pd.DataFrame(rows); display(C6); C6.to_csv(OUTPUT_DIR / 'C6_attribution_vs_features.csv', index=False)
    REPORT['C6. Attribution versus handcrafted features (exploratory)'] = C6

# Link to the radiologist audit, once it exists
if AUDIT_CSV and Path(AUDIT_CSV).exists() and len(XR):
    au = pd.read_csv(AUDIT_CSV)
    if 'patient_id' not in au and AUDIT_KEY and Path(AUDIT_KEY).exists():
        key = pd.read_csv(AUDIT_KEY, dtype=str)
        rid = next(c for c in key.columns if 'review' in c.lower()); pid = next(c for c in key.columns if 'patient' in c.lower())
        rcol = next(c for c in au.columns if 'review' in c.lower())
        au = au.merge(key[[rid, pid]].rename(columns={rid: rcol, pid: 'patient_id'}), on=rcol, how='left')
    JA = X_.merge(au, on='patient_id', how='inner'); print(f'Audit linked for {len(JA)} XRAI cases')
    out = []
    for c in au.columns:
        if c in ('patient_id',) or 'review' in c.lower(): continue
        v = pd.to_numeric(JA[c], errors='coerce')
        if v.notna().sum() >= 10 and v.nunique() > 1:
            for a in ['FLAIR_enrich_nonenh_lesion', 'T1c_share_enhancing', 'outside_lesion_share']:
                k = pd.concat([v, JA[a]], axis=1).dropna()
                out.append({'audit item': c, 'attribution metric': a, 'n': len(k), 'Spearman rho': round(spearmanr(k.iloc[:, 0], k.iloc[:, 1])[0], 3)})
    if out:
        C7 = pd.DataFrame(out); display(C7); C7.to_csv(OUTPUT_DIR / 'C7_attribution_vs_audit.csv', index=False)
        REPORT['C7. Attribution versus the radiologist audit (exploratory)'] = C7
else:
    print('Radiologist audit not linked yet: set AUDIT_CSV (and AUDIT_KEY if the audit uses review IDs) and re-run this cell.')
''')

# ---------------------------------------------------------------- Part D
md(r'''
## Part D. What does the network's representation encode?

For each ensemble member, every patient is passed through **its own held-out checkpoint** (never one that trained on
it). A hook records the 256-dimensional vector that the IDH head reads. The prediction is checked against the saved
out-of-fold probability (tolerance 0.01). An **untrained network** of the same architecture is the control: a random
3D CNN already captures crude intensity and size statistics, so only the gain over it reflects learning.

Each checkpoint has its own embedding space, so probes are fitted within each fold's held-out patients
(ridge regression, inner 5-fold CV) and the out-of-fold predictions are pooled. Targets: the 8 handcrafted features
(z-scored within site), age, IDH (sanity check) and site (a shortcut check). Feature probes are repeated in
**wildtype patients only**, so a feature cannot be decoded merely because it correlates with IDH status.
**Head alignment** is the cosine between each probe's direction and the IDH head's weight vector (about 0.05 for unrelated directions in 256 dimensions).
Caution: for a target the embedding cannot decode (R² near 0), the heavily regularised probe direction collapses onto
the embedding's covariance with that target, which is dominated by IDH whenever the target correlates with IDH. High
alignment is then a correlation artefact, so read D3 only for targets with clear probe R².
''')
code(r'''
import torch, gc, sys
from collections import defaultdict
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
EMB_DIR = OUTPUT_DIR / 'embeddings'; EMB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if RUN_PROBES:
    from selected_three_xrai import RUNS, THRESHOLD
    from ensemble_oof_xrai import cohort as oof_cohort, case_dataset, assert_held_out
    from research_experiments import load_checkpoint, ExperimentModel
    print('Device:', DEVICE, '|', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU will be slow (hours)')
    CASES = oof_cohort(runs=RUNS, threshold=THRESHOLD)
    print(f'{len(CASES)} development cases | members:', [Path(r).name for r in RUNS])

def remap_row(row):
    row = dict(row)
    for k in ('t1c_path', 'flair_path', 'brain_mask_path', 'tumor_mask_path'):
        v = row.get(k) or ''
        if v and not Path(v).exists():
            for a, b in PATH_REMAP.items():
                if v.startswith(a): v = b + v[len(a):]
            row[k] = v
    return row

def embed(model, row, state):
    box = {}
    h = model.idh_head.register_forward_hook(lambda m, i, o: box.__setitem__('z', i[0].detach()))
    try:
        batch = case_dataset(state, remap_row(row), INPUT_CACHE)[0]
        t, f = batch['t1c'][None].to(DEVICE), batch['flair'][None].to(DEVICE)
        with torch.no_grad(): logits = model(t, f)
    finally:
        h.remove()
    return box['z'][0].float().cpu().numpy(), float(torch.sigmoid(logits[0, 0]))

def run_member(j, run):
    by_ckpt = defaultdict(list)
    for c in CASES: by_ckpt[c['members'][j]['checkpoint']].append(c)
    for ckpt, cs in sorted(by_ckpt.items()):
        fold = cs[0]['members'][j]['fold']; part = EMB_DIR / f'm{j}_{Path(run).name}_fold{fold}.npz'
        if part.exists(): continue
        t0 = time.time(); model, state = load_checkpoint(ckpt, DEVICE)
        try:
            ids, Zs, P, S = [], [], [], []
            for k, c in enumerate(cs):
                assert_held_out(state, c['patient_id'])
                z, p = embed(model, c['row'], state)
                ids.append(c['patient_id']); Zs.append(z); P.append(p); S.append(c['members'][j]['probability'])
                if (k + 1) % 50 == 0: print(f'    {k + 1}/{len(cs)} | {time.time() - t0:.0f}s', flush=True)
            np.savez(part, patient_id=np.array(ids), emb=np.stack(Zs), prob=np.array(P), saved_prob=np.array(S),
                     head_w=model.idh_head.weight.detach().cpu().numpy().ravel(), fold=fold)
            bad = int((np.abs(np.array(P) - np.array(S)) > 0.01).sum())
            print(f'  member {j} {Path(run).name} fold {fold}: {len(cs)} patients, {time.time() - t0:.0f}s, '
                  f'{bad} predictions differ from saved OOF by >0.01', flush=True)
        finally:
            del model, state; gc.collect()
            if DEVICE == 'cuda': torch.cuda.empty_cache()

def run_random():
    part = EMB_DIR / f'random_init_seed{RANDOM_INIT_SEED}.npz'
    if part.exists(): return
    _, state = load_checkpoint(CASES[0]['members'][0]['checkpoint'], 'cpu')
    cfg = dict(state['model_config']); cfg.pop('pretrained_path', None); cfg.pop('pretrained_sha256', None)
    torch.manual_seed(RANDOM_INIT_SEED)
    model = ExperimentModel(target=state.get('target', 'joint'), **cfg).to(DEVICE).eval()
    t0 = time.time(); ids, Zs = [], []
    for k, c in enumerate(CASES):
        z, _ = embed(model, c['row'], state); ids.append(c['patient_id']); Zs.append(z)
        if (k + 1) % 100 == 0: print(f'    random init {k + 1}/{len(CASES)} | {time.time() - t0:.0f}s', flush=True)
    fold = np.array([c['members'][0]['fold'] for c in CASES])
    np.savez(part, patient_id=np.array(ids), emb=np.stack(Zs), fold=fold, head_w=model.idh_head.weight.detach().cpu().numpy().ravel())
    del model; gc.collect()

if RUN_PROBES:
    T0 = time.time()
    for j, run in enumerate(RUNS):
        print(f'Member {j}: {Path(run).name}', flush=True); run_member(j, run)
    print('Untrained control', flush=True); run_random()
    print(f'Embeddings done in {(time.time() - T0) / 60:.1f} min -> {EMB_DIR}')
''')
code(r'''
def load_parts(pattern):
    out = []
    for p in sorted(EMB_DIR.glob(pattern)):
        with np.load(p, allow_pickle=False) as z:
            fold = z['fold'] if z['fold'].ndim else np.full(len(z['patient_id']), int(z['fold']))
            out.append(dict(ids=z['patient_id'].astype(str), emb=z['emb'], fold=fold, head_w=z['head_w'],
                            prob=z['prob'] if 'prob' in z else None, saved=z['saved_prob'] if 'saved_prob' in z else None))
    return out
if RUN_PROBES:
    SPACES = {Path(r).name: load_parts(f'm{j}_{Path(r).name}_fold*.npz') for j, r in enumerate(RUNS)}
    rnd = load_parts(f'random_init_seed{RANDOM_INIT_SEED}.npz')[0]
    # split the untrained control by member 0's folds, so every space is probed with the same sample sizes
    SPACES['untrained control'] = [dict(ids=rnd['ids'][rnd['fold'] == f], emb=rnd['emb'][rnd['fold'] == f], fold=rnd['fold'][rnd['fold'] == f],
                                        head_w=rnd['head_w'], prob=None, saved=None) for f in np.unique(rnd['fold'])]
    for k, parts in SPACES.items():
        if parts and parts[0]['prob'] is not None:
            dp = np.concatenate([np.abs(p['prob'] - p['saved']) for p in parts])
            print(f'{k}: max |recomputed - saved OOF| = {dp.max():.4f}; {(dp > .01).sum()} of {len(dp)} > 0.01')
    ZP = Z.set_index('patient_id')
    TARGETS = CONV + MISM + ['age']
    def probe_space(parts, wildtype_only=False):
        res = defaultdict(lambda: ([], [])); idh_p, idh_y, site_p, site_y, cosines = [], [], [], [], defaultdict(list)
        for part in parts:
            keep = np.isin(part['ids'], ZP.index); ids = part['ids'][keep]; E = part['emb'][keep]
            if len(ids) < 30: continue
            T = ZP.loc[ids]; yy = T.y.values
            sel = (yy == 0) if wildtype_only else np.ones(len(ids), bool)
            kf = KFold(5, shuffle=True, random_state=SEED)
            for t in TARGETS:
                v = T[t].values.astype(float); m_ = sel & np.isfinite(v)
                if m_.sum() < 25: continue
                pr = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))), E[m_], v[m_], cv=kf)
                res[t][0].append(v[m_]); res[t][1].append(pr)
                if not wildtype_only:
                    beta = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(E[m_] - E[m_].mean(0), v[m_]).coef_
                    cosines[t].append(abs(beta @ part['head_w']) / (np.linalg.norm(beta) * np.linalg.norm(part['head_w']) + 1e-12))
            if not wildtype_only and 0 < yy.sum() < len(yy) and min(yy.sum(), len(yy) - yy.sum()) >= 5:
                sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
                idh_p.append(cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), E, yy, cv=sk, method='predict_proba')[:, 1]); idh_y.append(yy)
                st = T.site.values
                if len(np.unique(st)) > 1 and pd.Series(st).value_counts().min() >= 5:
                    site_p.append(cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), E, st,
                                                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED))); site_y.append(st)
        out = {}
        for t, (vs, ps) in res.items():
            v, p = np.concatenate(vs), np.concatenate(ps); out[t] = 1 - np.sum((v - p) ** 2) / np.sum((v - v.mean()) ** 2)
        if idh_p: out['IDH (AUROC)'] = safe_auc(np.concatenate(idh_y), np.concatenate(idh_p))
        if site_p: out['site (balanced accuracy)'] = balanced_accuracy_score(np.concatenate(site_y), np.concatenate(site_p))
        return out, {t: float(np.mean(c)) for t, c in cosines.items()}
    PROBE, PROBE_WT, ALIGN = {}, {}, {}
    for k, parts in SPACES.items():
        t0 = time.time(); PROBE[k], ALIGN[k] = probe_space(parts); PROBE_WT[k], _ = probe_space(parts, wildtype_only=True)
        print(f'probed {k} in {time.time() - t0:.0f}s')
    trained = [k for k in SPACES if k != 'untrained control']
    def probe_table(P, rows):
        D = pd.DataFrame(P).reindex(rows)
        D['trained members (mean)'] = D[trained].mean(axis=1); D['gain over untrained'] = D['trained members (mean)'] - D['untrained control']
        return D.round(3).reset_index().rename(columns={'index': 'target'})
    D1 = probe_table(PROBE, TARGETS + ['IDH (AUROC)', 'site (balanced accuracy)'])
    D2 = probe_table(PROBE_WT, TARGETS)
    D3 = pd.DataFrame(ALIGN).reindex(TARGETS).round(3).reset_index().rename(columns={'index': 'target'})
    print('D1. Linear decodability from the embedding (cross-validated R2 unless stated)'); display(D1)
    print('D2. Same, IDH-wildtype patients only'); display(D2)
    print('D3. |cosine| between each probe direction and the IDH head (unrelated directions: about 0.05)'); display(D3)
    D1.to_csv(OUTPUT_DIR / 'D1_probe_all.csv', index=False); D2.to_csv(OUTPUT_DIR / 'D2_probe_wildtype.csv', index=False)
    D3.to_csv(OUTPUT_DIR / 'D3_head_alignment.csv', index=False)
    REPORT['D1. What the representation encodes (cross-validated R²; IDH as AUROC, site as balanced accuracy)'] = D1
    REPORT['D2. Feature probes in IDH-wildtype patients only (cross-validated R²)'] = D2
    REPORT['D3. Alignment of probe directions with the IDH head (|cosine|)'] = D3

    fig, ax = plt.subplots(figsize=(7.5, 3.4)); xs = np.arange(len(TARGETS))
    ax.bar(xs - .2, D1.set_index('target').loc[TARGETS, 'trained members (mean)'], .4, color='#2a6fdb', label='trained (mean of 3)')
    ax.bar(xs + .2, D1.set_index('target').loc[TARGETS, 'untrained control'], .4, color='#9a9a96', label='untrained control')
    ax.set_xticks(xs); ax.set_xticklabels(TARGETS, rotation=35, ha='right', fontsize=8); ax.axhline(0, color='k', lw=.6)
    ax.set_ylabel('cross-validated R²'); ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'D_probes.png', dpi=200); plt.show()
else:
    print('RUN_PROBES is False: Part D skipped.')
''')

md('## Report')
code(r'''
lines = ['# What the ensemble adds beyond age and radiology', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {len(Z)} patients, {int(y.sum())} IDH-mutant | internal pooled-site cross-validation.', '',
         'Cross-validated AUROC (reproduced from the hybrid notebook): ' + ', '.join(f'{k} {v:.3f}' for k, v in AUC0.items()), '']
for title, df in REPORT.items():
    lines += [f'## {title}', '', md_table(df), '']
    if title.startswith('A4'): lines += [A4_NOTE, '']
lines += ['## Notes', '',
          '- Pre-declared: A1 within-class correlation, A4 age-matched AUROC, C1-C4. Part B and everything else is exploratory.',
          '- Deferral scores use only pre-operative inputs (images, segmentation, age).',
          '- The XRAI sample is outcome-enriched (33 per outcome): group comparisons are valid, frequencies are not.',
          '- Probe R² shows linear decodability, not use by the IDH head. D3 is interpretable only for targets with clear probe R²; for undecodable targets it reflects the target-IDH correlation.',
          '- Likelihood-ratio tests in A5 are in-sample and descriptive.']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('Wrote', OUTPUT_DIR / 'REPORT.md')
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
nbf.validate(nb)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'IDH_Hybrid_Novelty.ipynb'
nbf.write(nb, out); print('Wrote', out)
