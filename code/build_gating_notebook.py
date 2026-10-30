"""Build IDH_Confidence_Gating.ipynb: the RSNA 2026 confidence gate applied to the IDH hybrid.

Two arms, as in the paediatric HGG work:
  DL arm  = the three-member ensemble (logistic recalibration of its out-of-fold logit)
  ML arm  = handcrafted features + age (L2 logistic)
Strategies, all evaluated with repeated NESTED cross-validation (everything fitted, including thresholds,
uses only training folds):
  A  ML arm alone            B  DL arm alone
  C  fixed blend  a*p_ML + (1-a)*p_DL, a chosen on inner-CV predictions of the training folds
  D1 confidence gate (RSNA, parameter-free): w_ML = |p_ML-0.5| / (|p_ML-0.5| + |p_DL-0.5|)
  D2 confidence gate, threshold-centred: confidence = |logit p - logit t| with t each arm's training-fold
     Youden threshold (adapts the gate to 13% prevalence, where 0.5 is far from the decision boundary)
  E  stacked meta-learner: logistic regression on [logit p_ML, logit p_DL]
  H  full hybrid (ensemble logit + features + age), as in the paper
Run: python build_gating_notebook.py [output.ipynb]
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

CONFIG = after('## 1.').replace("OUTPUT_DIR = PROJECT / 'outputs' / 'idh_hybrid'", "OUTPUT_DIR = PROJECT / 'outputs' / 'idh_gating'")
assert 'idh_gating' in CONFIG
SETUP, HELPERS = after('## 2.'), after('## 3.')

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# Confidence gating for the IDH hybrid

The parameter-free confidence gate from the paediatric HGG work (RSNA 2026), applied to adult IDH prediction.
Two arms are combined per patient: the **DL arm** (three-member ensemble) and the **ML arm** (handcrafted features + age).

| Strategy | Rule | Fitted on training folds |
|---|---|---|
| A · ML arm | features + age, logistic | coefficients |
| B · DL arm | ensemble logit, recalibrated | intercept and slope |
| C · Fixed blend | a·p_ML + (1-a)·p_DL | a, on inner-CV predictions |
| **D1 · Confidence gate (RSNA)** | w_ML = \|p_ML - 0.5\| / (\|p_ML - 0.5\| + \|p_DL - 0.5\|) | nothing |
| D2 · Gate, threshold-centred | same, with confidence = \|logit p - logit t\| and t each arm's Youden threshold | the two thresholds |
| E · Stacked | logistic on [logit p_ML, logit p_DL] | coefficients |
| H · Full hybrid (paper) | logistic on ensemble logit + 8 features + age | coefficients |

Everything, including every operating threshold, is chosen inside the training folds of a site- and class-stratified
5-fold split repeated 5 times (the same splits as the hybrid notebook). D2 exists because IDH-mutant prevalence is 13%:
the decision boundary sits near 0.15, not 0.5, so "distance from 0.5" mostly measures how wildtype-looking a case is.

**Pre-declared comparisons:** D1 vs the better single arm, D1 vs H, D2 vs D1 (paired bootstrap). Everything else is
exploratory. Runtime: a few minutes on CPU.
''')
md('## 1. Configuration')
code(CONFIG + "\nN_BOOT_FAST = 1000\nAGE_MATCH_CALIPER, AGE_MATCH_K = 2.0, 3\n")
md('## 2. Cohort (identical to the hybrid notebook)')
code(SETUP)
code(HELPERS)

md(r'''
## 3. Nested evaluation of all strategies

For each outer training fold: inner 5-fold CV gives out-of-fold arm probabilities for the training patients; on those
we choose the blend weight a, fit the stacked model and set every strategy's Youden and 90%-sensitivity thresholds.
The arms are then refitted on the whole training fold and applied to the test fold.
''')
code(r'''
from scipy.stats import rankdata
warnings.filterwarnings('ignore', category=FutureWarning)
ML_COLS, DL_COLS = CONV + MISM + ['age'], ['dl_logit']
STRATS = ['A · ML arm (features + age)', 'B · DL arm (ensemble)', 'C · Fixed blend', 'D1 · Confidence gate (RSNA)',
          'D2 · Gate, threshold-centred', 'E · Stacked', 'H · Full hybrid']
A_, B_, C_, D1_, D2_, E_, H_ = STRATS
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def gate(pm, pd_, centre_m=0.5, centre_d=0.5, logodds=False):
    if logodds: cm, cd = np.abs(lg(pm) - lg(centre_m)), np.abs(lg(pd_) - lg(centre_d))
    else:       cm, cd = np.abs(pm - centre_m), np.abs(pd_ - centre_d)
    w = np.where(cm + cd > 0, cm / np.maximum(cm + cd, 1e-12), 0.5)
    return w * pm + (1 - w) * pd_, w
def youden_t(y, p): return youden_threshold(y, p)
def sens_t(y, p):   return sens_threshold(y, p)
ALPHAS = np.round(np.linspace(0, 1, 21), 2)

def fit_combiners(y, pm, pd_):
    """Everything a strategy needs, from out-of-fold arm probabilities of TRAINING patients only."""
    a = ALPHAS[int(np.argmax([roc_auc_score(y, al * pm + (1 - al) * pd_) for al in ALPHAS]))]
    stk = LogisticRegression(C=C_REG, max_iter=5000).fit(np.c_[lg(pm), lg(pd_)], y)
    tm, td = youden_t(y, pm), youden_t(y, pd_)
    return dict(a=a, stk=stk, tm=tm, td=td)

def combine(pm, pd_, f):
    out = {A_: pm, B_: pd_, C_: f['a'] * pm + (1 - f['a']) * pd_}
    out[D1_], w1 = gate(pm, pd_)
    out[D2_], w2 = gate(pm, pd_, f['tm'], f['td'], logodds=True)
    out[E_] = f['stk'].predict_proba(np.c_[lg(pm), lg(pd_)])[:, 1]
    return out, w1, w2

y = Z.y.values; strat = (Z.site + '_' + Z.y.astype(str)).values
XM, XD, XH = Z[ML_COLS].values, Z[DL_COLS].values, Z[MODELS['Hybrid']].values
n = len(Z); P = {s: np.zeros((REPEATS, n)) for s in STRATS}
DY = {s: np.zeros((REPEATS, n), int) for s in STRATS}; DS = {s: np.zeros((REPEATS, n), int) for s in STRATS}
W1 = np.zeros((REPEATS, n)); W2 = np.zeros((REPEATS, n)); ALPHA_USED = []
t0 = time.time()
for r in range(REPEATS):
    for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r).split(Z, strat)):
        # inner out-of-fold arm probabilities for the training patients
        pm_in, pd_in, ph_in = np.zeros(len(tr)), np.zeros(len(tr)), np.zeros(len(tr))
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=SEED + 100 * r + k).split(tr, strat[tr]):
            a_, b_ = tr[itr], tr[ite]
            pm_in[ite] = fit(XM[a_], y[a_]).predict_proba(XM[b_])[:, 1]
            pd_in[ite] = fit(XD[a_], y[a_]).predict_proba(XD[b_])[:, 1]
            ph_in[ite] = fit(XH[a_], y[a_]).predict_proba(XH[b_])[:, 1]
        f = fit_combiners(y[tr], pm_in, pd_in); ALPHA_USED.append(f['a'])
        train_p, _, _ = combine(pm_in, pd_in, f); train_p[H_] = ph_in
        # final arms on the whole training fold, applied to the test fold
        pm, pd_ = fit(XM[tr], y[tr]).predict_proba(XM[te])[:, 1], fit(XD[tr], y[tr]).predict_proba(XD[te])[:, 1]
        test_p, w1, w2 = combine(pm, pd_, f); test_p[H_] = fit(XH[tr], y[tr]).predict_proba(XH[te])[:, 1]
        W1[r, te], W2[r, te] = w1, w2
        for s in STRATS:
            ty, ts = youden_t(y[tr], train_p[s]), sens_t(y[tr], train_p[s])
            P[s][r, te] = test_p[s]; DY[s][r, te] = test_p[s] >= ty; DS[s][r, te] = test_p[s] >= ts
cvg = {s: dict(p=P[s].mean(0), pred_youden=(DY[s].mean(0) >= .5).astype(int), pred_sens90=(DS[s].mean(0) >= .5).astype(int))
       for s in STRATS}
w_ml1, w_ml2 = W1.mean(0), W2.mean(0)
print(f'{REPEATS}x{N_FOLDS} nested CV in {time.time() - t0:.0f}s | blend weight a on ML arm: median {np.median(ALPHA_USED):.2f} '
      f'(range {min(ALPHA_USED):.2f}-{max(ALPHA_USED):.2f})')
pd.DataFrame({'patient_id': Z.patient_id, 'site': Z.site, 'y': y, 'age': Z.age, 'grade_group': Z.grade_group,
              'w_ML_gate': w_ml1, 'w_ML_gate_threshold_centred': w_ml2,
              **{f'p[{s}]': cvg[s]['p'] for s in STRATS}, **{f'call[{s}]': cvg[s]['pred_youden'] for s in STRATS}}
             ).to_csv(OUTPUT_DIR / 'gating_cv_predictions.csv', index=False)
''')

md('## 4. Performance')
code(r'''
point = {s: all_metrics(y, cvg[s]) for s in STRATS}
best_arm = A_ if point[A_]['auroc'] >= point[B_]['auroc'] else B_
COMPARE = [(D1_, best_arm), (D1_, H_), (D2_, D1_), (D1_, B_), (D2_, H_), (E_, H_), (C_, D1_)]
ci, dci = bootstrap(Z, cvg, STRATS, compare=COMPARE)
SHOW = [('auroc', 'AUROC'), ('average_precision', 'AP'), ('brier', 'Brier'), ('ece', 'ECE'), ('calibration_slope', 'Slope'),
        ('sensitivity@youden', 'Sens.'), ('specificity@youden', 'Spec.'), ('ppv@youden', 'PPV'), ('f1@youden', 'F1'),
        ('mcc@youden', 'MCC'), ('specificity@sens90', 'Spec. @90% sens.')]
G1 = pd.DataFrame([{'strategy': s, **{lab: fmt_ci(point[s][k], ci[s][k]) for k, lab in SHOW}} for s in STRATS])
display(G1); G1.to_csv(OUTPUT_DIR / 'gating_metrics.csv', index=False)
pd.DataFrame([{'strategy': s, **point[s]} for s in STRATS]).to_csv(OUTPUT_DIR / 'gating_metrics_raw.csv', index=False)
rows = []
for (a, b) in COMPARE:
    d_ = dci[(a, b)]
    rows.append({'comparison': f'{a}  minus  {b}', 'pre-declared': (a, b) in [(D1_, best_arm), (D1_, H_), (D2_, D1_)],
                 **{lab: f'{d_[k][0]:+.3f} ({d_[k][1]:+.3f} to {d_[k][2]:+.3f})' for k, lab in
                    [('auroc', 'AUROC'), ('average_precision', 'AP'), ('brier', 'Brier'), ('specificity@youden', 'Spec.'), ('f1@youden', 'F1')]}})
G2 = pd.DataFrame(rows); display(G2); G2.to_csv(OUTPUT_DIR / 'gating_paired_differences.csv', index=False)
print('Better single arm:', best_arm)
''')

md(r'''
## 5. How the gate behaves

Left: each patient's arm confidences, with the ML arm's errors marked (the RSNA figure). Right: the per-patient weight
the gate gives the ML arm. Then: who the gate hands to which arm, by grade, age and enhancement.
''')
code(r'''
pA, pB = cvg[A_]['p'], cvg[B_]['p']
errA = cvg[A_]['pred_youden'] != y; errB = cvg[B_]['pred_youden'] != y
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
cA, cB = np.abs(pA - .5), np.abs(pB - .5)
ax.scatter(cA[~errA], cB[~errA], s=6, color='#9a9a96', alpha=.5, label='ML arm correct')
ax.scatter(cA[errA], cB[errA], s=22, marker='x', color='#c0392b', label='ML arm wrong')
ax.plot([0, .5], [0, .5], color='k', lw=.7); ax.set_xlabel('ML-arm confidence |p - 0.5|'); ax.set_ylabel('DL-arm confidence |p - 0.5|')
ax.set_title('D1 gate: arm confidences'); ax.legend(frameon=False, fontsize=8)
ax = axes[1]
tmA, tmB = youden_t(y, pA), youden_t(y, pB)
lcA, lcB = np.abs(lg(pA) - lg(tmA)), np.abs(lg(pB) - lg(tmB))
ax.scatter(lcA[~errA], lcB[~errA], s=6, color='#9a9a96', alpha=.5, label='ML arm correct')
ax.scatter(lcA[errA], lcB[errA], s=22, marker='x', color='#c0392b', label='ML arm wrong')
m_ = max(lcA.max(), lcB.max()); ax.plot([0, m_], [0, m_], color='k', lw=.7)
ax.set_xlabel('ML-arm confidence |logit p - logit t|'); ax.set_ylabel('DL-arm confidence'); ax.set_title('D2 gate: threshold-centred')
ax = axes[2]
ax.hist(w_ml1, bins=30, alpha=.6, color='#3b8e3f', label='D1 (centre 0.5)')
ax.hist(w_ml2, bins=30, alpha=.6, color='#2a6fdb', label='D2 (threshold-centred)')
ax.set_xlabel('weight on ML arm (features + age)'); ax.set_ylabel('patients'); ax.legend(frameon=False, fontsize=8)
ax.set_title('Per-patient gate weights')
for a in axes: a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'gate_behaviour.png', dpi=200); plt.show()

band = pd.cut(Z.age, [0, 40, 55, 200], right=False, labels=['<40', '40-54', '>=55']).astype(str)
enh = np.where(Z.no_enhancement.values > .5, 'non-enhancing', 'enhancing')
rows = []
for var, vals in [('all', np.array(['all'] * n)), ('grade', Z.grade_group.values), ('age', band.values), ('enhancement', enh)]:
    for lev in pd.unique(vals):
        k = vals == lev
        rows.append({'group': f'{var}: {lev}', 'n': int(k.sum()), 'mutant': int(y[k].sum()),
                     'D1 mean w_ML': round(w_ml1[k].mean(), 2), 'D1 % patients led by DL (w_ML<0.5)': round(100 * (w_ml1[k] < .5).mean(), 1),
                     'D2 mean w_ML': round(w_ml2[k].mean(), 2), 'D2 % led by DL': round(100 * (w_ml2[k] < .5).mean(), 1),
                     'ML-arm errors / DL-arm errors': f'{int(errA[k].sum())} / {int(errB[k].sum())}'})
G3 = pd.DataFrame(rows); display(G3); G3.to_csv(OUTPUT_DIR / 'gate_weights_by_group.csv', index=False)
both = errA & errB
print(f'Errors: ML arm {errA.sum()}, DL arm {errB.sum()}, both {both.sum()} (a gate cannot fix cases where both arms are wrong).')
print(f'ML-arm errors where the DL arm was more confident (D2 scale): {int((errA & (lcB > lcA)).sum())} of {int(errA.sum())}.')
''')

md(r'''
## 6. Subgroups, age matching and deferral

Does the gate keep the ensemble's advantages (grade 2-3, under 40) and features+age's (grade 4, non-enhancing)?
Does it hold up when age is matched out? And does deferral by the gate's own confidence work as well as the hybrid's?
''')
code(r'''
SG_STRATS = [A_, B_, D1_, D2_, E_, H_]
rows = []
for var, vals in [('grade', Z.grade_group.values), ('age', band.values), ('enhancement', enh), ('site', Z.site.values)]:
    for lev in sorted(pd.unique(vals)):
        k = vals == lev
        if y[k].sum() < 5 or (1 - y[k]).sum() < 5: continue
        rows.append({'group': f'{var}: {lev}', 'n': int(k.sum()), 'mutant': int(y[k].sum()),
                     **{s: round(roc_auc_score(y[k], cvg[s]['p'][k]), 3) for s in SG_STRATS}})
G4 = pd.DataFrame(rows); display(G4); G4.to_csv(OUTPUT_DIR / 'gating_subgroups.csv', index=False)

# age-matched sample: same procedure as the novelty notebook
rng = np.random.default_rng(SEED); used = set(); sets = []; AGE = Z.age.values; sites = Z.site.values
mut = np.where(y == 1)[0].copy(); rng.shuffle(mut)
for m in mut:
    cand = np.where((y == 0) & (sites == sites[m]) & (np.abs(AGE - AGE[m]) <= AGE_MATCH_CALIPER))[0]
    cand = sorted((c for c in cand if c not in used), key=lambda c: (abs(AGE[c] - AGE[m]), c))[:AGE_MATCH_K]
    if cand: used.update(cand); sets.append(np.array([m] + list(cand)))
MATCHED = np.concatenate(sets)
def clus_ci(s):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(N_BOOT_FAST):
        i = np.concatenate([sets[j] for j in rng.integers(0, len(sets), len(sets))]); v.append(roc_auc_score(y[i], cvg[s]['p'][i]))
    return np.percentile(v, [2.5, 97.5])
G5 = pd.DataFrame([{'strategy': s, 'AUROC, all': round(point[s]['auroc'], 3),
                    'AUROC, age-matched': fmt_ci(roc_auc_score(y[MATCHED], cvg[s]['p'][MATCHED]), clus_ci(s))} for s in STRATS])
print(f'Age-matched sample: {len(sets)} mutants, {len(MATCHED) - len(sets)} wildtype')
display(G5); G5.to_csv(OUTPUT_DIR / 'gating_age_matched.csv', index=False)

# deferral by each strategy's own confidence: distance of its probability from its (median nested) Youden threshold
COV = np.round(np.arange(.5, 1.0001, .01), 2)
def curve(p, call, thr):
    err = (call != y).astype(float); order = np.argsort(np.abs(lg(p) - lg(thr)))[::-1]   # most confident first
    cum = np.cumsum(err[order]); nk = np.maximum(np.round(COV * n).astype(int), 1); return cum[nk - 1] / nk
rows = []; fig, ax = plt.subplots(figsize=(5.2, 3.8))
for s, c in [(H_, '#2a6fdb'), (D1_, '#7b1fa2'), (D2_, '#d98a1c'), (B_, '#5b5b58')]:
    thr = youden_t(y, cvg[s]['p'])   # display only: the ranking uses distance from a single threshold
    cu = curve(cvg[s]['p'], cvg[s]['pred_youden'], thr); ax.plot(COV * 100, cu * 100, color=c, lw=1.6, label=s)
    k80 = int(np.where(COV == .8)[0][0])
    rows.append({'strategy': s, 'error rate, no deferral': round(1 - (cvg[s]['pred_youden'] == y).mean(), 3),
                 'error rate, 20% deferred': round(cu[k80], 3), 'AURC (50-100% coverage)': round(cu.mean(), 4)})
ax.invert_xaxis(); ax.set_xlabel('patients kept (%)'); ax.set_ylabel('error rate among kept (%)'); ax.legend(frameon=False, fontsize=7)
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'gating_risk_coverage.png', dpi=200); plt.show()
G6 = pd.DataFrame(rows); display(G6); G6.to_csv(OUTPUT_DIR / 'gating_deferral.csv', index=False)
print('Deferral ranks patients by distance from a single full-data Youden threshold for display; the error calls are the nested ones.')
''')

md('## Report')
code(r'''
def mdt(df):
    cols = list(df.columns)
    return '\n'.join(['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.values])
lines = ['# Confidence gating for the IDH hybrid', '', f'Run {time.strftime("%Y-%m-%d %H:%M")} | {n} patients, {int(y.sum())} IDH-mutant | '
         f'{REPEATS}x{N_FOLDS} nested CV. DL arm = ensemble; ML arm = features + age. Better single arm: {best_arm}.', '',
         '## Performance (95% CI)', '', mdt(G1), '', '## Paired differences (95% CI)', '', mdt(G2), '',
         '## Gate weights by group', '', mdt(G3), '', '## Subgroup AUROC', '', mdt(G4), '',
         '## Age-matched AUROC', '', mdt(G5), '', '## Deferral by each strategy\'s own confidence', '', mdt(G6), '',
         '## Notes', '', '- Pre-declared: D1 vs the better single arm, D1 vs H, D2 vs D1. Everything else is exploratory.',
         '- D1 is exactly the RSNA rule (distance from 0.5). D2 measures distance from each arm\'s training-fold Youden threshold on the log-odds scale.',
         '- The fixed blend weight and the stacked model are fitted on inner-CV predictions of the training folds only.',
         '- ECE intervals are not reported (biased under resampling).']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8'); print('Wrote', OUTPUT_DIR / 'REPORT.md')
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
nbf.validate(nb)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'IDH_Confidence_Gating.ipynb'
nbf.write(nb, out); print('Wrote', out)
