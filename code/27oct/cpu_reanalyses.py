"""CPU re-analyses for the RadGenome ISBI paper, 27 Oct 2026 (responds to the v6 pre-submission review and the
strengthening plan). Everything runs from files in the v6 package plus run-level OOF files; no images, no GPU.

Each section reproduces a published number first, then adds the new analysis, and writes its own CSV. The paper's
own code is executed verbatim where it exists (hybrid helpers, correlates hypothesis tests, gating loop).

  R0  reproduction check of the paper's cross-validated predictions
  R1  imaging correlates rerun on development patients only (review M1)
  R2  reserved set: spectrum, post-hoc paired differences, grade-4-only (review M2, plan Part 1)
  R3  deferral, hybrid vs gate under the same in-fold threshold definition (review B8)
  R4  E3 nested member-selection simulation: how optimistic is the selected ensemble?
  R5  age-matched sample: all models incl. ensemble + features and the gate; who was matched (review B13, H)
  R6  traceable sensitivity checks: ensemble-aligned folds; training-fold z-scoring (review C, 'untraceable checks')
  R7  joint vs IDH-only head on v3 without the reserved patients (review M1)
  R8  MU-Glioma-Post ensemble 0.627 vs 0.602 (review C)
  R9  attribution: analysed n per test, region shares, blinded copy of C_xrai_region_metrics.csv (review B1, B9, F)
  R10 subgroups with harmonised age bands (<40 / 40-54 / >=55) and gate weights (review B3, C)
  R11 counts quoted in the review (mask provenance, grade unknown, ADC/perfusion subset, youngest reserved patient)
  R12 combined Table 1: internal, age-matched, reserved

Run (from this folder): python cpu_reanalyses.py [output_dir]   (paths: see common.py)
"""
import sys, json, time, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
import nbformat
from scipy.stats import spearmanr, mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from common import build, built, logit, md_table, RES, DL_OUT, RG_OUT, PKG, display, _cell_after

warnings.filterwarnings('ignore')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'out/12_cpu_reanalyses_27oct'); OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).parent
T0 = time.time()
REPORT = []
def section(title, *blocks):
    REPORT.append(f'## {title}\n')
    for b in blocks: REPORT.append(b if isinstance(b, str) else md_table(b)); REPORT.append('')
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

G = build(); Z = G['Z']; d = G['d']; y = Z.y.values
MODELS, ORDER, RAW, SEED = G['MODELS'], G['ORDER'], G['RAW'], G['SEED']
CONV, MISM = G['CONV'], G['MISM']
run_cv, fit, all_metrics, bootstrap = G['run_cv'], G['fit'], G['all_metrics'], G['bootstrap']
youden_threshold, sens_threshold, binary_metrics = G['youden_threshold'], G['sens_threshold'], G['binary_metrics']
STRAT = (Z.site + '_' + Z.y.astype(str)).values
GROUPS = [np.where(STRAT == g)[0] for g in np.unique(STRAT)]
def fmt(v, ci, nd=3): return f'{v:.{nd}f} ({ci[0]:.{nd}f} to {ci[1]:.{nd}f})'
def sfmt(v, ci, nd=3): return f'{v:+.{nd}f} ({ci[0]:+.{nd}f} to {ci[1]:+.{nd}f})'

# ------------------------------------------------------------------ R0
log('R0 reproduce CV')
cv = run_cv(Z, MODELS)
cvp = G['CVP']
r0 = pd.DataFrame([dict(model=m, auroc=roc_auc_score(y, cv[m]['p']),
                        max_abs_diff_vs_package=float(np.abs(cv[m]['p'] - cvp[f'p[{m}]'].values).max()),
                        youden_calls_identical=bool((cv[m]['pred_youden'] == cvp[f'call[{m}]'].values).all())) for m in ORDER])
r0.to_csv(OUT / 'R0_reproduction.csv', index=False)
assert r0.max_abs_diff_vs_package.max() < 1e-9 and r0.youden_calls_identical.all()
section('R0. Reproduction of the paper\'s cross-validated predictions (1,393 patients)',
        'All eight models reproduce the package\'s `cv_predictions.csv` to machine precision, with identical Youden calls, so every analysis below runs on exactly the paper\'s predictions.', r0)

# ------------------------------------------------------------------ R1
log('R1 correlates, development only')
NBC = nbformat.read(built('idh_correlates'), 4)
hyp_code = _cell_after(NBC, '## 7.')
hyp_code = hyp_code[:hyp_code.index("STRATUM = 'site'\nres = run_hypotheses()")]
feats_all = pd.read_csv(RES / '01_IDH_imaging_correlates' / 'features.csv')
feats_all = feats_all[feats_all.error.isna() | (feats_all.error.astype(str).str.len() == 0)]
man = pd.read_csv(DL_OUT / 'clinical_experiments' / 'v2_reference_seed42' / 'fold_manifest.csv', dtype=str, keep_default_na=False)
reserved_ids = set(json.loads((DL_OUT / 'balanced_holdout_ids.json').read_text())['test_patient_ids'])
lab = man[man.idh.isin(['0', '1'])][['patient_id', 'idh']].rename(columns={'idh': 'y'}); lab['y'] = lab.y.astype(int)
grade = pd.read_csv(RES / '09_confidence_gating' / 'gating_cv_predictions.csv')[['patient_id', 'grade_group', 'age']]
def corr_data(ids):
    x = feats_all[feats_all.patient_id.isin(ids)].merge(lab, on='patient_id').merge(grade, on='patient_id', how='left')
    x.loc[x.grade_group.isna() & (x.site == 'UPENN'), 'grade_group'] = 'grade 4'
    x['grade_group'] = x.grade_group.fillna('unknown')
    return x
r1_rows = []
for label, ids in [('development, all with the feature', set(man.patient_id) - reserved_ids),
                   ('development, the 1,393 analysis patients', set(Z.patient_id))]:
    gg = {'display': display, 'N_PERM': 10000, 'N_BOOT': 2000, 'SEED': SEED, 'pd': pd, 'np': np, 'rankdata': rankdata,
          'mannwhitneyu': mannwhitneyu, 'data': corr_data(ids)}
    assert not (set(gg['data'].patient_id) & reserved_ids)
    exec(hyp_code, gg)
    for stratum, tag in [('site', 'within site'), ('site+grade', 'within site x grade group')]:
        if stratum == 'site+grade':
            gg['data'] = gg['data'][gg['data'].grade_group != 'unknown']
        gg['STRATUM'] = stratum; res = gg['run_hypotheses']()
        res['cohort_definition'] = label; res['analysis'] = tag; r1_rows.append(res)
        gg['data'] = corr_data(ids)
R1 = pd.concat(r1_rows); R1.to_csv(OUT / 'R1_correlates_development_only.csv', index=False)
pub = pd.read_csv(RES / '01_IDH_imaging_correlates' / 'hypothesis_results.csv')
pooled = R1[R1.cohort.str.startswith('Pooled') & R1.id.str.startswith('H')]
rows = []
for (cd, an), g in pooled.groupby(['cohort_definition', 'analysis'], sort=False):
    for r in g.itertuples():
        pp = pub[(pub.id == r.id) & pub.cohort.str.startswith('Pooled') & pub.analysis.str.contains('grade' if 'grade' in an else 'primary')].iloc[0]
        rows.append({'cohort': cd, 'analysis': an, 'hypothesis': f'{r.id} {r.label}', 'n (mutant)': f'{r.n} ({r.idh_mutant})',
                     'oriented AUROC (95% CI)': f'{r.oriented_auroc:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})', 'Holm p': f'{r.p_holm:.4f}',
                     'verdict': r.verdict, 'published (with reserved)': f'{pp.oriented_auroc:.3f}, n={pp.n}'})
R1s = pd.DataFrame(rows); R1s.to_csv(OUT / 'R1_correlates_summary.csv', index=False)
site_rows = R1[(~R1.cohort.str.startswith('Pooled')) & R1.id.str.startswith('H') & (R1.analysis == 'within site')]
all_above = site_rows.groupby('cohort_definition').apply(lambda g: bool((g.oriented_auroc > .5).all())).to_dict()
section('R1. Imaging correlates rerun without the reserved patients (review M1)',
        'The published correlates analysis (n = 1,587) included 191 reserved patients. Re-running the identical pre-specified '
        'tests (same code, 10,000 within-site permutations, Holm across H1-H4) on development patients only:', R1s,
        f'Direction agrees at every site for H1-H4: {all_above}. The minimum attainable permutation p is 1/10,001, so Holm p = 0.0004 is the floor for four tests.')

# ------------------------------------------------------------------ R2
log('R2 reserved set')
rp = pd.read_csv(RES / '11_reserved_test' / 'reserved_predictions.csv')
yr = rp.y.values
gate_name = 'Confidence gate (RSNA)'
strata = (rp.site + '_' + rp.y.astype(str)).values
rng = np.random.default_rng(SEED)
BOOT = [np.concatenate([rng.choice(g, len(g)) for g in [np.where(strata == s)[0] for s in np.unique(strata)]]) for _ in range(2000)]
pub_diff = pd.read_csv(RES / '11_reserved_test' / 'reserved_paired_differences.csv')
def rdiff(a, b, idx_sets, mask=None):
    m_ = np.ones(len(yr), bool) if mask is None else mask
    yy = yr; pa, pb = rp[a].values, rp[b].values
    v = [roc_auc_score(yy[i], pa[i]) - roc_auc_score(yy[i], pb[i]) for i in idx_sets]
    pt = roc_auc_score(yy[m_], pa[m_]) - roc_auc_score(yy[m_], pb[m_])
    return pt, np.percentile(v, [2.5, 97.5]), float(np.mean(np.array(v) <= 0))
# check: reproduce one published difference with the same bootstrap
pt, ci, _ = rdiff('Hybrid', 'Ensemble (raw)', BOOT)
pubrow = pub_diff[pub_diff.comparison == 'Hybrid minus Ensemble (raw)'].iloc[0]
repro_ok = abs(pt - pubrow.delta_auroc) < 1e-9 and pubrow.ci == f'{ci[0]:.3f}-{ci[1]:.3f}'
g4 = (rp.grade_group == 'grade 4').values
rng4 = np.random.default_rng(SEED + 4)
B4 = [np.concatenate([rng4.choice(np.where(g4 & (yr == v))[0], (g4 & (yr == v)).sum()) for v in (0, 1)]) for _ in range(2000)]
# internal equivalents, same bootstrap as the paper (paired, site x class)
ci_int, dci_int = bootstrap(Z, cv, ['Hybrid'], compare=[('Features + age', RAW), ('Ensemble + features', 'Features'),
                                                   ('Hybrid', 'Features + age'), ('Features + age', 'Ensemble + features')])
r2 = []
for a, b, why in [('Features + age', 'Ensemble (raw)', 'post hoc: does the internal parity of features + age and the ensemble hold?'),
                  ('Ensemble + features', 'Features', 'post hoc: image information of the network beyond the radiological features, no age'),
                  ('Features + age', 'Ensemble + features', 'post hoc: age versus the network, both added to the features'),
                  ('Hybrid', 'Features + age', 'pre-declared (as published)')]:
    pt, ci, p0 = rdiff(a, b, BOOT); pt4, ci4, _ = rdiff(a, b, B4, g4)
    key = (a if a != 'Ensemble (raw)' else RAW, b if b != 'Ensemble (raw)' else RAW)
    di = dci_int.get(key)
    r2.append({'comparison': f'{a} minus {b}', 'status': why,
               'internal (1,393)': sfmt(di['auroc'][0], di['auroc'][1:]) if di else '',
               'reserved (190)': sfmt(pt, ci), 'reserved P(diff<=0)': round(p0, 3),
               'reserved grade 4 (127)': sfmt(pt4, ci4)})
R2 = pd.DataFrame(r2); R2.to_csv(OUT / 'R2_reserved_paired_differences_post_hoc.csv', index=False)
# spectrum
spec = pd.crosstab([rp.site, rp.grade_group], rp.y.map({1: 'IDH-mutant', 0: 'IDH-wildtype'}), margins=True)
spec.to_csv(OUT / 'R2_reserved_spectrum.csv')
spec_dev = pd.crosstab(Z.grade_group, Z.y.map({1: 'IDH-mutant', 0: 'IDH-wildtype'}), margins=True)
sub_pub = pd.read_csv(RES / '11_reserved_test' / 'reserved_subgroups.csv')
g4tab = sub_pub[sub_pub.subgroup == 'grade 4'][['model', 'n', 'idh_mutant', 'auroc', 'auroc_ci']]
# all-model grade 4 AUROCs (adds Features, Ensemble + features)
g4all = pd.DataFrame([dict(model=m, auroc=roc_auc_score(yr[g4], rp[m].values[g4]),
                           ci='{:.3f}-{:.3f}'.format(*np.percentile([roc_auc_score(yr[i], rp[m].values[i]) for i in B4], [2.5, 97.5])))
                      for m in ['Age', 'Features', 'Features + age', 'Ensemble (raw)', 'Ensemble + age', 'Ensemble + features', 'Hybrid', gate_name]])
g4all.to_csv(OUT / 'R2_reserved_grade4_all_models.csv', index=False)
# development grade-4 AUROCs for comparison
dg4 = (Z.grade_group == 'grade 4').values
dev_g4 = pd.DataFrame([dict(model=m, development_grade4=roc_auc_score(y[dg4], cv[m]['p'][dg4]),
                            development_grade23=roc_auc_score(y[Z.grade_group.values == 'grade 2-3'], cv[m]['p'][Z.grade_group.values == 'grade 2-3']))
                       for m in ORDER])
section('R2. Reserved set: spectrum and post-hoc comparisons (review M2; plan Part 1)',
        f'Bootstrap reproduces the package\'s reserved_paired_differences.csv exactly: {repro_ok} (same seed, 2,000 site x IDH resamples). Grade-4 subgroup intervals use a class-stratified bootstrap within grade 4.',
        '**Spectrum (analysed reserved patients):**', spec.reset_index(),
        '**Development spectrum (1,393):**', spec_dev.reset_index(),
        '**Paired AUROC differences** (only "Hybrid minus ..." was pre-declared for the reserved set):', R2,
        '**Reserved grade 4 only (n = 127, 31 mutant), all models:**', g4all,
        '**Development AUROC by grade group, for comparison:**', dev_g4)

# ------------------------------------------------------------------ R3
log('R3 deferral, consistent definitions')
NBG = nbformat.read(built('gating'), 4)
loop = _cell_after(NBG, '## 3. Nested')
loop = loop.replace("            P[s][r, te] = test_p[s]; DY[s][r, te] = test_p[s] >= ty; DS[s][r, te] = test_p[s] >= ts",
                    "            P[s][r, te] = test_p[s]; DY[s][r, te] = test_p[s] >= ty; DS[s][r, te] = test_p[s] >= ts\n"
                    "            TH_FOLD[s].append(ty)")
loop = loop.replace("t0 = time.time()\n", "t0 = time.time()\nTH_FOLD = {s: [] for s in STRATS}\n", 1)
assert 'TH_FOLD[s].append(ty)' in loop
loop = loop.replace(").to_csv(OUTPUT_DIR / 'gating_cv_predictions.csv', index=False)", ")")
gg = dict(G); gg['display'] = display
exec(loop, gg)
gcv = pd.read_csv(RES / '09_confidence_gating' / 'gating_cv_predictions.csv')
for s in gg['STRATS']:
    assert np.abs(gg['cvg'][s]['p'] - gcv[f'p[{s}]'].values).max() < 1e-9 and (gg['cvg'][s]['pred_youden'] == gcv[f'call[{s}]'].values).all(), s
H_, D1_, D2_, B_, A_ = 'H · Full hybrid', 'D1 · Confidence gate (RSNA)', 'D2 · Gate, threshold-centred', 'B · DL arm (ensemble)', 'A · ML arm (features + age)'
def deferral_table(p, call, thr_fold, name):
    out = []
    err = (call != y).astype(float)
    t_in = float(np.mean(thr_fold)); t_full = youden_threshold(y, p)
    for rule, centre in [('in-fold threshold (mean of 25 training-fold Youden thresholds)', t_in),
                         ('full-data Youden threshold (as in gating_deferral.csv)', t_full), ('distance from 0.5 (RSNA rule)', 0.5)]:
        conf = np.abs(logit(p) - logit(centre))
        order = np.argsort(-conf, kind='stable')
        for frac in (0.10, 0.20, 0.30):
            n_def = int(round(frac * len(y))); keep = np.zeros(len(y), bool); keep[order[:len(y) - n_def]] = True
            bm = binary_metrics(y[keep], call[keep])
            out.append({'model': name, 'ranking': rule, 'centre': round(centre, 4), 'deferred': f'{frac:.0%}',
                        'errors captured': f'{int(err[~keep].sum())}/{int(err.sum())} ({err[~keep].sum() / err.sum():.0%})',
                        'error among kept': round(err[keep].mean(), 3), 'retained accuracy': round(1 - err[keep].mean(), 3),
                        'mutants deferred': int(y[~keep].sum()), 'retained sensitivity': round(bm['sensitivity'], 3),
                        'retained specificity': round(bm['specificity'], 3), 'retained PPV': round(bm['ppv'], 3)})
    return out
r3 = []
r3 += deferral_table(cv['Hybrid']['p'], cv['Hybrid']['pred_youden'], cv['Hybrid']['thresholds'][:, 0], 'Hybrid (hybrid-notebook CV)')
for s in (H_, D1_, D2_, B_):
    r3 += deferral_table(gg['cvg'][s]['p'], gg['cvg'][s]['pred_youden'], gg['TH_FOLD'][s], s)
R3 = pd.DataFrame(r3); R3.to_csv(OUT / 'R3_deferral_consistent_definitions.csv', index=False)
# AURC under the in-fold definition, with paired bootstrap hybrid vs gate
COV = np.round(np.arange(0.50, 1.0001, 0.01), 2)
def aurc(p, call, centre, idx):
    err = (call != y).astype(float); conf = np.abs(logit(p) - logit(centre))
    o = idx[np.argsort(-conf[idx], kind='stable')]; cum = np.cumsum(err[o]); n = np.maximum(np.round(COV * len(idx)).astype(int), 1)
    return float((cum[n - 1] / n).mean())
cH = float(np.mean(cv['Hybrid']['thresholds'][:, 0])); cD = float(np.mean(gg['TH_FOLD'][D1_]))
ALL = np.arange(len(y)); rngb = np.random.default_rng(SEED + 7); dd_ = []
for _ in range(1000):
    i = np.concatenate([rngb.choice(g, len(g)) for g in GROUPS])
    dd_.append(aurc(cv['Hybrid']['p'], cv['Hybrid']['pred_youden'], cH, i) - aurc(gg['cvg'][D1_]['p'], gg['cvg'][D1_]['pred_youden'], cD, i))
aurc_txt = (f"AURC (50-100% coverage), in-fold definition: hybrid {aurc(cv['Hybrid']['p'], cv['Hybrid']['pred_youden'], cH, ALL):.4f}, "
            f"gate D1 {aurc(gg['cvg'][D1_]['p'], gg['cvg'][D1_]['pred_youden'], cD, ALL):.4f}; hybrid minus gate "
            f"{np.mean(dd_):+.4f} (95% CI {np.percentile(dd_, 2.5):+.4f} to {np.percentile(dd_, 97.5):+.4f}).")
r3s = R3[(R3.deferred == '20%') & R3.model.isin(['Hybrid (hybrid-notebook CV)', D1_, D2_])][
    ['model', 'ranking', 'centre', 'errors captured', 'error among kept', 'retained accuracy', 'mutants deferred', 'retained sensitivity', 'retained PPV']]
section('R3. Deferral: hybrid vs confidence gate under one definition (review B8)',
        'The gating loop was re-executed verbatim (its probabilities and calls reproduce `gating_cv_predictions.csv` exactly) while recording each training fold\'s Youden threshold. '
        'Ranking = distance of the log-odds from the stated centre; calls are always the in-fold majority calls. 20% deferral shown; 10% and 30% are in the CSV.', r3s, aurc_txt)

# ------------------------------------------------------------------ R4
log('R4 nested member selection')
CE = DL_OUT / 'clinical_experiments'
RUNS = {'CA42': CE / 'v2_reference_seed42', 'CA43': CE / 'v2_reference_seed43', 'CA44': CE / 'v2_reference_seed44',
        'MC42': CE / 'v2_mean_concat_seed42', 'MC43': RG_OUT / 'clinical_experiments' / 'v2_mean_concat_seed43',
        'MC44': RG_OUT / 'clinical_experiments' / 'v2_mean_concat_seed44', 'T1c42': CE / 'v2_t1c_only_seed42'}
PM = None
for k, v in RUNS.items():
    o = pd.read_csv(v / 'oof_predictions.csv')[['patient_id', 'site', 'idh_label', 'idh_probability']].rename(columns={'idh_probability': k})
    PM = o if PM is None else PM.merge(o[['patient_id', k]], on='patient_id')
PM = PM[PM.idh_label.notna()].reset_index(drop=True); ym = PM.idh_label.astype(int).values
names = list(RUNS); M = PM[names].values
subsets = [c for r in range(1, len(names) + 1) for c in itertools.combinations(range(len(names)), r)]
ENS = np.stack([M[:, list(s)].mean(1) for s in subsets], 1)          # patients x subsets
def fast_auc(yy, P):                                                 # AUROC for each column of P
    R = np.apply_along_axis(rankdata, 0, P); n1 = yy.sum(); n0 = len(yy) - n1
    return (R[yy == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
app = fast_auc(ym, ENS)
sel3 = subsets.index(tuple(names.index(k) for k in ('CA42', 'CA44', 'MC42')))
strat_m = (PM.site + '_' + PM.idh_label.astype(int).astype(str)).values
res4 = []
for restrict, allowed in [('any subset of the 7 candidates (127)', np.ones(len(subsets), bool)),
                          ('three-member subsets only (35)', np.array([len(s) == 3 for s in subsets]))]:
    reps = []; chosen = []
    for r in range(5):
        pred = np.zeros(len(ym))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(PM, strat_m):
            a_tr = fast_auc(ym[tr], ENS[tr]); a_tr[~allowed] = -1
            j = int(np.argmax(a_tr)); chosen.append('+'.join(names[i] for i in subsets[j])); pred[te] = ENS[te, j]
        reps.append(roc_auc_score(ym, pred))
    best_app = float(app[allowed].max())
    res4.append({'selection space': restrict, 'apparent best (selected and scored on all 1,432)': round(best_app, 4),
                 'nested: selected in training folds, scored on held-out (mean over 5 repeats)': round(float(np.mean(reps)), 4),
                 'nested range over repeats': f'{min(reps):.4f}-{max(reps):.4f}', 'optimism (apparent - nested)': round(best_app - float(np.mean(reps)), 4),
                 'paper\'s three (CA42+CA44+MC42), apparent': round(float(app[sel3]), 4),
                 'most often chosen in folds': pd.Series(chosen).value_counts().head(3).to_dict()})
R4 = pd.DataFrame(res4); R4.to_csv(OUT / 'R4_nested_member_selection.csv', index=False)
singles = pd.DataFrame([{'member': k, 'OOF AUROC (1,432)': round(roc_auc_score(ym, PM[k]), 4)} for k in names])
singles.to_csv(OUT / 'R4_single_members.csv', index=False)
q = np.sort(app)[::-1]
section('R4. E3 nested member-selection simulation',
        'Candidate pool = the seven T1c-reading runs available when the ensemble was chosen (CA42-44, MC42-44, T1c-only 42); '
        'the two T2+FLAIR runs are excluded because they were trained after the ensemble was chosen (alone they score 0.73-0.78). '
        'Selection criterion = OOF AUROC of the mean-probability ensemble on the training patients of each outer fold (5x5, site x IDH stratified).',
        R4, singles,
        f'Spread of apparent AUROC across all 127 subsets: max {q[0]:.4f}, 90th percentile {np.percentile(app, 90):.4f}, '
        f'median {np.median(app):.4f}; across subsets of size >= 2: min {app[[len(s) >= 2 for s in subsets]].min():.4f}.')

# ------------------------------------------------------------------ R5
log('R5 age matched')
AGE = Z.age.values; MU_ = y == 1; WT = y == 0; sites = Z.site.values
rngm = np.random.default_rng(SEED); used = set(); sets = []
mut = np.where(MU_)[0].copy(); rngm.shuffle(mut)
for m in mut:
    cand = np.where(WT & (sites == sites[m]) & (np.abs(AGE - AGE[m]) <= 2.0))[0]
    cand = sorted((c for c in cand if c not in used), key=lambda c: (abs(AGE[c] - AGE[m]), c))[:3]
    if cand: used.update(cand); sets.append(np.array([m] + list(cand)))
MATCHED = np.concatenate(sets)
def clus_ci(p):
    r_ = np.random.default_rng(SEED); v = []
    for _ in range(1000):
        i = np.concatenate([sets[j] for j in r_.integers(0, len(sets), len(sets))])
        v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])
pub_a4 = pd.read_csv(RES / '03_what_the_ensemble_adds' / 'A4_age_matched_auroc.csv').set_index('model')
preds5 = {m: cv[m]['p'] for m in ORDER}
preds5['Confidence gate (D1)'] = gg['cvg'][D1_]['p']; preds5['Gate, threshold-centred (D2)'] = gg['cvg'][D2_]['p']
r5 = []
for m, p in preds5.items():
    a = roc_auc_score(y[MATCHED], p[MATCHED]); ci = clus_ci(p)
    r5.append({'model': m, 'contains age': m in ('Age', 'Features + age', 'Ensemble + age', 'Hybrid', 'Confidence gate (D1)', 'Gate, threshold-centred (D2)'),
               'AUROC, age-matched (95% CI)': fmt(a, ci), 'AUROC, all patients': round(roc_auc_score(y, p), 3),
               'published': pub_a4.loc[m, 'AUROC, age-matched (95% CI)'] if m in pub_a4.index else ''})
R5 = pd.DataFrame(r5); R5.to_csv(OUT / 'R5_age_matched_all_models.csv', index=False)
mm = MATCHED[y[MATCHED] == 1]; um = np.setdiff1d(np.where(MU_)[0], mm)
r5b = pd.DataFrame([{'group': 'all IDH-mutant', 'n': int(MU_.sum()), 'median age': np.median(AGE[MU_]), 'IQR': f'{np.percentile(AGE[MU_], 25):.0f}-{np.percentile(AGE[MU_], 75):.0f}'},
                    {'group': 'matched IDH-mutant', 'n': len(mm), 'median age': np.median(AGE[mm]), 'IQR': f'{np.percentile(AGE[mm], 25):.0f}-{np.percentile(AGE[mm], 75):.0f}'},
                    {'group': 'unmatched IDH-mutant', 'n': len(um), 'median age': np.median(AGE[um]), 'IQR': f'{np.percentile(AGE[um], 25):.0f}-{np.percentile(AGE[um], 75):.0f}'},
                    {'group': 'matched IDH-wildtype', 'n': int((y[MATCHED] == 0).sum()), 'median age': np.median(AGE[MATCHED][y[MATCHED] == 0]), 'IQR': ''},
                    {'group': 'matched IDH-mutant that are grade 2-3', 'n': int((Z.grade_group.values[mm] == 'grade 2-3').sum()), 'median age': np.median(AGE[mm][Z.grade_group.values[mm] == 'grade 2-3']), 'IQR': ''},
                    {'group': 'unmatched IDH-mutant that are grade 2-3', 'n': int((Z.grade_group.values[um] == 'grade 2-3').sum()), 'median age': np.median(AGE[um][Z.grade_group.values[um] == 'grade 2-3']), 'IQR': ''}])
r5b.to_csv(OUT / 'R5_age_matched_who_was_matched.csv', index=False)
contain = R5[R5['contains age'] & (R5.model != 'Age')]
lo_hi = [float(s.split(' ')[0]) for s in contain['AUROC, age-matched (95% CI)']]
section('R5. Age-matched sample: every model, and who could be matched (review M3, B13; plan)',
        f'Matching re-run with the published code and seed: {len(sets)} of {int(MU_.sum())} mutants, {len(MATCHED) - len(sets)} wildtype (published 96/262).',
        R5, r5b, f'Fitted models that contain age (age alone excluded) span {min(lo_hi):.3f}-{max(lo_hi):.3f} in the age-matched sample; image-only models {", ".join(R5[~R5["contains age"]]["AUROC, age-matched (95% CI)"].str.split(" ").str[0])}.')

# ------------------------------------------------------------------ R6
log('R6 sensitivity checks')
def run_cv_custom(frame, cols, splits_fn, zscore_in_fold=False, repeats=5):
    yy = frame.y.values; strat = (frame.site + '_' + frame.y.astype(str)).values
    P = np.zeros((repeats, len(frame)))
    raw = d.set_index('patient_id').loc[frame.patient_id]
    for r in range(repeats):
        for k, (tr, te) in enumerate(splits_fn(r, frame, strat)):
            X = frame[cols].copy()
            if zscore_in_fold:
                for c in cols:
                    if c in ('dl_logit', 'age', 'no_enhancement'): continue
                    v = raw[c].values.astype(float); out = np.empty(len(v))
                    for s in np.unique(frame.site):
                        ks = frame.site.values == s; trs = np.intersect1d(tr, np.where(ks)[0])
                        mu_, sd_ = v[trs].mean(), v[trs].std(ddof=0) + 1e-9; out[ks] = (v[ks] - mu_) / sd_
                    X[c] = out
            X = X.values
            P[r, te] = fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return P.mean(0)
std_split = lambda r, f, s: StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(f, s)
ens_fold = Z.fold.values
ens_split = lambda r, f, s: [(np.where(ens_fold != k)[0], np.where(ens_fold == k)[0]) for k in sorted(np.unique(ens_fold))]
r6 = []
for m in ('Hybrid', 'Features + age', 'Ensemble + age'):
    base = roc_auc_score(y, cv[m]['p'])
    a1 = roc_auc_score(y, run_cv_custom(Z, MODELS[m], ens_split, repeats=1))
    a2 = roc_auc_score(y, run_cv_custom(Z, MODELS[m], std_split, zscore_in_fold=True))
    r6.append({'model': m, 'paper (5x5 stratified CV, site z-scoring on all patients)': round(base, 4),
               'folds = the ensemble members\' own 5 folds': round(a1, 4), 'site z-scoring fitted in training folds only': round(a2, 4),
               'change, in-fold z-scoring': round(a2 - base, 4)})
R6 = pd.DataFrame(r6); R6.to_csv(OUT / 'R6_sensitivity_checks.csv', index=False)
section('R6. Traceable versions of the two sensitivity checks quoted in Methods', R6,
        'Aligned folds: the combiner\'s single 5-fold split is the ensemble members\' own CV split (fold column of the OOF file), so each test fold\'s ensemble logits all come from the one checkpoint per member that excluded that fold. In-fold z-scoring: site means and SDs of the eight features are computed on the training patients of each fold only.')

# ------------------------------------------------------------------ R7
log('R7 v3 heads')
V3 = DL_OUT / 'v3_experiments' / 'v3_mask_only_01'
j3 = pd.read_csv(V3 / 'joint_crop' / 'oof_predictions.csv'); i3 = pd.read_csv(V3 / 'idh_only_crop' / 'oof_predictions.csv')
h3 = j3[['patient_id', 'site', 'idh_label', 'idh_probability']].rename(columns={'idh_probability': 'joint'}).merge(
    i3[['patient_id', 'idh_probability']].rename(columns={'idh_probability': 'idh_only'}), on='patient_id')
h3 = h3[h3.idh_label.notna()].reset_index(drop=True)
r7 = []
for label, keep in [('all v3 patients (as in README)', np.ones(len(h3), bool)), ('reserved patients removed', ~h3.patient_id.isin(reserved_ids).values),
                    ('the 1,393 analysis patients', h3.patient_id.isin(set(Z.patient_id)).values)]:
    s = h3[keep].reset_index(drop=True); yy = s.idh_label.astype(int).values
    st = (s.site + '_' + s.idh_label.astype(int).astype(str)).values
    gr = [np.where(st == g)[0] for g in np.unique(st)]; r_ = np.random.default_rng(SEED); v = []
    for _ in range(2000):
        i = np.concatenate([r_.choice(g, len(g)) for g in gr]); v.append(roc_auc_score(yy[i], s.idh_only.values[i]) - roc_auc_score(yy[i], s.joint.values[i]))
    r7.append({'patients': label, 'n': len(s), 'mutant': int(yy.sum()), 'reserved included': int(s.patient_id.isin(reserved_ids).sum()),
               'joint head IDH AUROC': round(roc_auc_score(yy, s.joint), 3), 'IDH-only head IDH AUROC': round(roc_auc_score(yy, s.idh_only), 3),
               'IDH-only minus joint (95% CI)': sfmt(roc_auc_score(yy, s.idh_only) - roc_auc_score(yy, s.joint), np.percentile(v, [2.5, 97.5]))})
R7 = pd.DataFrame(r7); R7.to_csv(OUT / 'R7_v3_head_comparison_without_reserved.csv', index=False)
section('R7. Joint vs IDH-only head (v3, matched runs) without the reserved patients (review M1)', R7)

# ------------------------------------------------------------------ R8
log('R8 MU')
mu4 = pd.read_csv(RES / '04_external_MU_RHUH' / 'external_metrics.csv')
mu7 = pd.read_csv(RES / '07_ensemble_only_external_earlier' / '02_external_MU_Glioma_Post_three_member' / 'mu_three_member_scan_predictions.csv')
mu7j = json.loads((RES / '07_ensemble_only_external_earlier' / '02_external_MU_Glioma_Post_three_member' / 'mu_three_member_results.json').read_text())
fm = pd.read_csv(RES / '04_external_MU_RHUH' / 'features_MU_ext-v3.csv')
mu7['idh'] = pd.to_numeric(mu7.idh, errors='coerce')
lab7 = mu7[mu7.idh.isin([0, 1])].copy()
first7 = lab7.sort_values(['subject_id', 'timepoint']).groupby('subject_id').head(1)
fm_ok = fm.error.isna() | (fm.error.astype(str).str.len() == 0)
in04 = lab7[lab7.patient_id.isin(set(fm.scan_id))]                       # folder 04: scans with a tumour mask and IDH label
first04 = in04.sort_values(['subject_id', 'timepoint']).groupby('subject_id').head(1)
first04 = first04[first04.patient_id.isin(set(fm.loc[fm_ok, 'scan_id']))]  # then scans whose feature extraction failed are dropped
first_ok = first04
r8 = [{'analysis': 'folder 07: earliest labelled scan per patient (any scan)', 'n': len(first7), 'mutant': int(first7.idh.sum()),
       'ensemble AUROC': round(roc_auc_score(first7.idh, first7.idh_probability_three), 3)},
      {'analysis': 'folder 04: earliest labelled scan that has a tumour mask, extraction errors dropped', 'n': len(first04), 'mutant': int(first04.idh.sum()),
       'ensemble AUROC': round(roc_auc_score(first04.idh, first04.idh_probability_three), 3)}]
for r in mu4[(mu4.cohort == 'MU-Glioma-Post') & mu4.model.isin(['Ensemble (raw)', 'Hybrid', 'Age'])].itertuples():
    r8.append({'analysis': f'folder 04 published: {r.analysis}, {r.model}', 'n': r.n, 'mutant': r.idh_mutant, 'ensemble AUROC': round(r.auroc, 3)})
R8 = pd.DataFrame(r8); R8.to_csv(OUT / 'R8_mu_ensemble_discrepancy.csv', index=False)
diff_pat = sorted(set(first7.subject_id) ^ set(first_ok.subject_id)); chg = first7.merge(first_ok, on='subject_id', suffixes=('_07', '_04'))
n_changed = int((chg.patient_id_07 != chg.patient_id_04).sum())
section('R8. MU-Glioma-Post ensemble AUROC 0.627 (folder 04) vs 0.602 (folder 07)', R8,
        f'Same 15-checkpoint probabilities in both folders. {n_changed} patients have a different "earliest scan": folder 07 takes the earliest labelled scan, folder 04 the earliest labelled scan that has a tumour mask. {len(diff_pat)} patients are in only one of the two analyses. '
        f'Labelled scans without a tumour mask (absent from folder 04): {len(lab7) - len(in04)}. The 0.025 difference therefore comes from scan selection, not from different model outputs.')

# ------------------------------------------------------------------ R9
log('R9 attribution')
xr = pd.read_csv(RES / '03_what_the_ensemble_adds' / 'C_xrai_region_metrics.csv')
sel = json.loads((RES / '05_feature_audit_sample' / 'xrai_review_sample_selection.json').read_text())
in_cohort = xr.patient_id.isin(set(Z.patient_id))
X_ = xr[xr.lesion_voxels > 0]
counts = pd.DataFrame([
    {'item': 'review sample', 'n': len(xr), 'detail': xr.category.value_counts().to_dict()},
    {'item': 'not in the 1,393 analysis cohort', 'n': int((~in_cohort).sum()), 'detail': xr.loc[~in_cohort, 'category'].value_counts().to_dict()},
    {'item': 'no lesion voxels (excluded from region tests)', 'n': int((xr.lesion_voxels == 0).sum()), 'detail': xr.loc[xr.lesion_voxels == 0, ['site', 'category']].astype(str).agg(' '.join, axis=1).tolist()},
    {'item': 'C1 FLAIR enrichment, mutant vs wildtype', 'n': int(X_.FLAIR_enrich_nonenh_lesion.replace([np.inf, -np.inf], np.nan).notna().sum()),
     'detail': X_[np.isfinite(X_.FLAIR_enrich_nonenh_lesion)].label.map({1: 'mutant', 0: 'wildtype'}).value_counts().to_dict()},
    {'item': 'C2 T1c density, enhancing cases', 'n': int(X_[X_.volume_share_enhancing > 0].T1c_density_enhancing.notna().sum()), 'detail': ''},
    {'item': 'C3 outside-lesion share, errors vs correct', 'n': int(X_.outside_lesion_share.notna().sum()),
     'detail': f"errors {int((~X_.category.isin(['TP', 'TN'])).sum())}, correct {int(X_.category.isin(['TP', 'TN']).sum())}"},
    {'item': 'C4 FN vs TP (finite FLAIR enrichment)', 'n': int((X_.category.isin(['FN', 'TP']) & np.isfinite(X_.FLAIR_enrich_nonenh_lesion)).sum()),
     'detail': X_[X_.category.isin(['FN', 'TP']) & np.isfinite(X_.FLAIR_enrich_nonenh_lesion)].category.value_counts().to_dict()},
    {'item': 'sampling', 'n': sel.get('sampling', {}).get('eligible', ''), 'detail': f"threshold {sel.get('threshold')}, outcome categories from the ensemble at that threshold"}])
counts.to_csv(OUT / 'R9_attribution_counts.csv', index=False)
shares = []
for s in ('T1c', 'FLAIR'):
    row = {'sequence': s}
    for r in ('enhancing', 'nonenh_core', 'oedema', 'ring', 'distant'):
        row[f'share {r} (median)'] = round(X_[f'{s}_share_{r}'].median(), 3)
        row[f'enrichment {r} (median)'] = round(X_[f'{s}_enrich_{r}'].replace([np.inf, -np.inf], np.nan).median(), 2)
    shares.append(row)
shares.append({'sequence': 'volume share (median)', **{f'share {r} (median)': round(X_[f'volume_share_{r}'].median(), 3) for r in ('enhancing', 'nonenh_core', 'oedema', 'ring', 'distant')}})
R9 = pd.DataFrame(shares); R9.to_csv(OUT / 'R9_attribution_region_shares.csv', index=False)
blind = xr.drop(columns=[c for c in ('patient_id', 'label', 'category', 'probability', 'mutant', 'correct') if c in xr.columns])
blind = blind.sample(frac=1, random_state=1).reset_index(drop=True)
blind.to_csv(OUT / 'C_xrai_region_metrics_BLINDED.csv', index=False)
section('R9. Attribution analysis: analysed n and region shares (review B1, B9, F)',
        'Shares and enrichments are computed from positive Integrated Gradients (absolute IG for the FLAIR share); XRAI only defines the top-10% masks.',
        counts, R9,
        '`C_xrai_region_metrics_BLINDED.csv` drops patient_id, label, category and probability and shuffles rows; use it in any copy of the package that may reach the reader.')

# ------------------------------------------------------------------ R10
log('R10 subgroups, harmonised age bands')
band = pd.cut(Z.age, [0, 40, 55, 200], right=False, labels=['<40', '40-54', '>=55']).astype(str).values
r10 = []
for lev in ('<40', '40-54', '>=55'):
    k = band == lev; row = {'age band': lev, 'n': int(k.sum()), 'mutant': int(y[k].sum())}
    for m in ['Age', 'Features + age', RAW, 'Ensemble + age', 'Hybrid']: row[m] = round(roc_auc_score(y[k], cv[m]['p'][k]), 3)
    row['gate D1 mean w_ML'] = round(gg['w_ml1'][k].mean(), 2); row['gate D1 % led by ensemble'] = round(100 * (gg['w_ml1'][k] < .5).mean(), 1)
    r10.append(row)
enh = np.where(Z.no_enhancement.values > .5, 'non-enhancing', 'enhancing')
for lev in ('enhancing', 'non-enhancing'):
    k = enh == lev; row = {'age band': f'enhancement: {lev}', 'n': int(k.sum()), 'mutant': int(y[k].sum())}
    for m in ['Age', 'Features + age', RAW, 'Ensemble + age', 'Hybrid']: row[m] = round(roc_auc_score(y[k], cv[m]['p'][k]), 3)
    r10.append(row)
R10 = pd.DataFrame(r10); R10.to_csv(OUT / 'R10_subgroups_harmonised_age_bands.csv', index=False)
section('R10. Subgroups with the gate\'s age bands (<40 / 40-54 / >=55) and enhancement (review B2, B3, C)', R10)

# ------------------------------------------------------------------ R11
log('R11 counts')
adv = Z.dropna(subset=[c for c in G['ADV'] if c in Z])
res_full = pd.read_csv(RES / '11_reserved_test' / 'reserved_predictions.csv')
r11 = pd.DataFrame([
    {'item': 'mask provenance (1,393)', 'value': Z.mask_source.fillna('unknown').value_counts().to_dict()},
    {'item': 'grade groups (1,393)', 'value': Z.grade_group.value_counts().to_dict()},
    {'item': 'unknown grade, by site', 'value': Z[Z.grade_group == 'unknown'].site.value_counts().to_dict()},
    {'item': 'ADC + perfusion subset', 'value': f'n = {len(adv)}, {int(adv.y.sum())} mutant, sites {sorted(adv.site.unique())}'},
    {'item': 'youngest patients (1,393)', 'value': Z.nsmallest(3, 'age')[['site', 'age', 'y']].to_dict('records')},
    {'item': 'youngest reserved patients (190)', 'value': res_full.nsmallest(3, 'age')[['site', 'age', 'y', 'grade_group']].to_dict('records')},
    {'item': 'reserved: wildtype grade 2-3 / mutant grade 2-3', 'value': f"{int(((res_full.grade_group == 'grade 2-3') & (res_full.y == 0)).sum())} / {int(((res_full.grade_group == 'grade 2-3') & (res_full.y == 1)).sum())}"},
])
r11.to_csv(OUT / 'R11_counts.csv', index=False)
section('R11. Counts quoted in the review', r11)

# ------------------------------------------------------------------ R12
log('R12 combined table')
rm = pd.read_csv(RES / '11_reserved_test' / 'reserved_metrics.csv').set_index('model')
pm_ = pd.read_csv(RES / '02_hybrid_model' / 'primary_metrics.csv').set_index('model')
gm = pd.read_csv(RES / '09_confidence_gating' / 'gating_metrics.csv').set_index('strategy')
r12 = []
for m in ['Age', 'Features', 'Features + age', 'Ensemble (raw)', 'Ensemble + age', 'Ensemble + features', 'Hybrid', 'Confidence gate (D1)']:
    internal = gm.loc[D1_, 'AUROC'] if m.startswith('Confidence') else pm_.loc[m, 'auroc']
    am = R5.set_index('model').loc[m, 'AUROC, age-matched (95% CI)']
    rk = gate_name if m.startswith('Confidence') else m
    r12.append({'model': m, 'internal AUROC (1,393)': internal, 'age-matched AUROC (358)': am,
                'reserved AUROC (190)': f"{rm.loc[rk, 'auroc']:.3f} ({rm.loc[rk, 'auroc_ci'].replace('-', '-')})",
                'reserved grade 4 (127)': f"{g4all.set_index('model').loc[rk, 'auroc']:.3f}"})
R12 = pd.DataFrame(r12); R12.to_csv(OUT / 'R12_table1_internal_agematched_reserved.csv', index=False)
section('R12. Combined Table 1 candidate: internal, age-matched and reserved AUROC', R12)

# ------------------------------------------------------------------ report
head = ['# CPU re-analyses for the ISBI paper (27 Oct 2026)', '',
        f'Generated by `cpu_reanalyses.py` from package v6 files and run-level OOF predictions. Runtime {time.time() - T0:.0f} s. '
        'All analyses are internal and use no images. "Post hoc" marks comparisons not in any pre-declared protocol.', '']
(OUT / 'REPORT.md').write_text('\n'.join(head + REPORT), encoding='utf-8')
log('done ->', OUT)
