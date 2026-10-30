import sys, numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest, spearmanr
R=sys.argv[1].rstrip('/')+'/'; OUT=sys.argv[2].rstrip('/')+'/'; SEED=20261021
cv=pd.read_csv(R+'03_what_the_ensemble_adds/cv_predictions.csv')
g=pd.read_csv(R+'09_confidence_gating/gating_cv_predictions.csv')
f=pd.read_csv(R+'01_IDH_imaging_correlates/features.csv')[['patient_id','no_enhancement']].drop_duplicates('patient_id')
Z=cv.merge(g[['patient_id','grade_group']],on='patient_id',how='left').merge(f,on='patient_id',how='left')
assert (Z.patient_id.values==cv.patient_id.values).all() and (g.patient_id.values==cv.patient_id.values).all()
Z.loc[Z.grade_group.isna()&(Z.site=='UPENN'),'grade_group']='grade 4'
y=Z.y.values; age=Z.age.values; site=Z.site.values
M=['Age','Features','Features + age','Ensemble (raw)','Ensemble + age','Ensemble + features','Hybrid']
P={m:Z[f'p[{m}]'].values for m in M}; P['Confidence gate (D1)']=g['p[D1 · Confidence gate (RSNA)]'].values
# S1 within-set age-matched AUROC
rng=np.random.default_rng(SEED); used=set(); sets=[]
mut=np.where(y==1)[0].copy(); rng.shuffle(mut)
for m in mut:
    cand=np.where((y==0)&(site==site[m])&(np.abs(age-age[m])<=2.0))[0]
    cand=sorted((c for c in cand if c not in used),key=lambda c:(abs(age[c]-age[m]),c))[:3]
    if cand: used.update(cand); sets.append(np.array([m]+list(cand)))
assert len(sets)==96 and sum(len(s) for s in sets)==358
def within(p,ss):
    num=den=0.
    for s in ss: pm=p[s[0]]; pw=p[s[1:]]; num+=(pm>pw).sum()+.5*(pm==pw).sum(); den+=len(pw)
    return num/den
r=np.random.default_rng(SEED); B=[r.integers(0,len(sets),len(sets)) for _ in range(2000)]
bs={m:np.array([within(P[m],[sets[j] for j in b]) for b in B]) for m in P}
s1=pd.DataFrame([{'model':m,'within-set AUROC':round(within(P[m],sets),3),'ci':f'{np.percentile(bs[m],2.5):.3f}-{np.percentile(bs[m],97.5):.3f}','pooled AUROC in matched sample (v7)':round(roc_auc_score(y[np.concatenate(sets)],P[m][np.concatenate(sets)]),3)} for m in P])
rows=[]
for a,b in [('Features + age','Ensemble (raw)'),('Ensemble (raw)','Features'),('Ensemble + features','Features'),('Hybrid','Features + age'),('Hybrid','Ensemble + age')]:
    d=bs[a]-bs[b]; rows.append({'comparison':f'{a} minus {b}','delta':round(within(P[a],sets)-within(P[b],sets),3),'ci':f'{np.percentile(d,2.5):+.3f} to {np.percentile(d,97.5):+.3f}'})
s1b=pd.DataFrame(rows)
# S2 subgroups
def boot(mask,a,b):
    idx=np.where(mask)[0]; st=site[idx]+'_'+y[idx].astype(str); G=[idx[st==s] for s in np.unique(st)]
    rr=np.random.default_rng(SEED); v=[]
    for _ in range(2000):
        i=np.concatenate([rr.choice(gg,len(gg)) for gg in G]); v.append(roc_auc_score(y[i],P[a][i])-roc_auc_score(y[i],P[b][i]))
    return np.percentile(v,[2.5,97.5])
subs={'age <40':age<40,'grade 2-3':Z.grade_group.values=='grade 2-3','grade 4':Z.grade_group.values=='grade 4','non-enhancing':Z.no_enhancement.values>.5}
s2=[]
for k,msk in subs.items():
    ci=boot(msk,'Ensemble (raw)','Features + age')
    s2.append({'subgroup':k,'n':int(msk.sum()),'mutant':int(y[msk].sum()),**{m:round(roc_auc_score(y[msk],P[m][msk]),3) for m in ['Features + age','Ensemble (raw)','Ensemble + age','Hybrid']},
               'Ensemble minus Features + age':f"{roc_auc_score(y[msk],P['Ensemble (raw)'][msk])-roc_auc_score(y[msk],P['Features + age'][msk]):+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})"})
s2=pd.DataFrame(s2)
# S3 external McNemar
d=json.load(open(R+'02_hybrid_model/frozen_models.json'))
E=pd.read_csv(R+'13_held_out_and_external_27oct/e2_ds007045/frozen_model_predictions.csv')
s3=[];s3b=[]
for op in ['threshold_youden','threshold_sens90']:
    C={m:(E[m].values>=d['models'][m][op]) for m in ['Age','Features + age','Ensemble (raw)','Hybrid']}
    for m,c in C.items(): s3b.append({'operating point':op,'model':m,'false positives':int(c.sum()),'specificity':round(1-c.mean(),3)})
    for a,b in [('Hybrid','Features + age'),('Ensemble (raw)','Features + age'),('Hybrid','Ensemble (raw)')]:
        n01=int((C[a]&~C[b]).sum()); n10=int((~C[a]&C[b]).sum())
        s3.append({'operating point':op,'comparison':f'{a} vs {b}','FP only by first':n01,'FP only by second':n10,'exact McNemar p':f'{binomtest(min(n01,n10),n01+n10,0.5).pvalue:.2g}'})
u=E[E.age<40]
s3c=pd.DataFrame([{'group':'ds007045 age <40','n':len(u),'Age FP':int((u.Age>=d['models']['Age']['threshold_youden']).sum()),'Features + age FP':int((u['Features + age']>=d['models']['Features + age']['threshold_youden']).sum()),'Ensemble FP':int((u['Ensemble (raw)']>=d['models']['Ensemble (raw)']['threshold_youden']).sum()),'Hybrid FP':int((u['Hybrid']>=d['models']['Hybrid']['threshold_youden']).sum())}])
s3=pd.DataFrame(s3); s3b=pd.DataFrame(s3b)
# S4 rho
e=P['Ensemble (raw)']; s4=pd.DataFrame([{'class':c,'spearman rho(ensemble, age)':round(spearmanr(e[y==k],age[y==k]).correlation,3)} for k,c in [(0,'IDH-wildtype'),(1,'IDH-mutant')]])
for n,t in [('S1_age_matched_within_set',s1),('S1_age_matched_within_set_differences',s1b),('S2_subgroups_paired',s2),('S3_ds007045_mcnemar',s3),('S3_ds007045_specificity_exact_thresholds',s3b),('S3_ds007045_under40',s3c),('S4_ensemble_age_rho',s4)]: t.to_csv(OUT+n+'.csv',index=False)
md=lambda t:t.to_markdown(index=False)
open(OUT+'REPORT.md','w').write(f"""# v8 recomputations (29 Oct 2026)

Requested by the Stage 2 audit of v7. Script: `code/29oct/recompute_v8.py` (run as `python recompute_v8.py results/ results/14_v8_recomputations_29oct/`). Inputs are the package's own per-patient files: `03/cv_predictions.csv`, `09/gating_cv_predictions.csv`, `01/features.csv`, `02/frozen_models.json`, `13/e2_ds007045/frozen_model_predictions.csv`. Seed {SEED}, 2,000 bootstraps. The published age matching (96 sets, 358 patients) is reproduced exactly before anything is computed.

## S1. Age-matched sample: AUROC within matched sets

Each mutant is compared only with its own age-matched, same-site wildtype controls; CIs from a set-level bootstrap. The v7 table pooled all 358 patients, which compares patients across sets and lets age differences between sets back in.

{md(s1)}

{md(s1b)}

## S2. Pooled-CV subgroups with paired CIs (site x class stratified bootstrap within the subgroup)

{md(s2)}

## S3. ds007045: exact McNemar tests at the frozen thresholds

{md(s3b)}

{md(s3)}

{md(s3c)}

## S4. Ensemble and age within IDH class

{md(s4)}
""")
print(open(OUT+'REPORT.md').read())

# S5 calibration slope on the reserved set (frozen models), site x class bootstrap
from sklearn.linear_model import LogisticRegression
rp=pd.read_csv(R+'11_reserved_test/reserved_predictions.csv'); yr=rp.y.values
def cslope(yy,p):
    p=np.clip(p,1e-6,1-1e-6); lg=np.log(p/(1-p))
    return float(LogisticRegression(C=np.inf,max_iter=5000).fit(lg.reshape(-1,1),yy).coef_[0][0])
st=(rp.site+'_'+rp.y.astype(str)).values; GR=[np.where(st==s_)[0] for s_ in np.unique(st)]
s5=[]
for m in ['Features + age','Ensemble (raw)','Ensemble + age','Hybrid','Confidence gate (RSNA)']:
    p=rp[m].values; rr=np.random.default_rng(SEED); bsl=[]
    for _ in range(2000):
        i=np.concatenate([rr.choice(g_,len(g_)) for g_ in GR]); bsl.append(cslope(yr[i],p[i]))
    s5.append({'model':m,'calibration slope':round(cslope(yr,p),2),'ci':f'{np.percentile(bsl,2.5):.2f}-{np.percentile(bsl,97.5):.2f}'})
s5=pd.DataFrame(s5); s5.to_csv(OUT+'S5_reserved_calibration_slope.csv',index=False)
open(OUT+'REPORT.md','a').write(f"""
## S5. Calibration slope on the reserved set (frozen models)

Logistic recalibration slope on the predicted log-odds; the slope is unaffected by the 50/50 case-control design, unlike the Brier score and calibration-in-the-large. Site x class stratified bootstrap.

{s5.to_markdown(index=False)}
""")
print(s5)

# S6 ds007045: specificity without the 32 KA subjects added in dataset v2.0.0 (IDH status not stated for them)
from scipy.stats import binomtest as _bt
NEW_KA = {f'DS7045:sub-KA{n}' for n in [103, 117] + list(range(119, 129)) + list(range(130, 137)) + list(range(138, 153))}
madd = E.patient_id.isin(NEW_KA)
s6 = []
for op in ['threshold_youden', 'threshold_sens90']:
    for mod in ['Features + age', 'Ensemble (raw)', 'Hybrid']:
        for name, df in [('all 359', E), ('original descriptor cohort only', E[~madd]), ('added in v2.0.0 only', E[madd])]:
            fp = int((df[mod] >= d['models'][mod][op]).sum()); n = len(df); c_ = _bt(n - fp, n).proportion_ci(method='exact')
            s6.append({'operating point': op, 'model': mod, 'subset': name, 'n': n, 'false positives': fp,
                       'specificity %': round(100 * (1 - fp / n), 1), 'ci': f'{100*c_.low:.1f}-{100*c_.high:.1f}'})
s6 = pd.DataFrame(s6); s6.to_csv(OUT + 'S6_ds007045_without_added_subjects.csv', index=False)
open(OUT + 'REPORT.md', 'a').write(f"""
## S6. ds007045 without the subjects added in dataset v2.0.0

The data descriptor (Filimonova et al., Sci Data 13:1213, 2026) gives IDH-wildtype GBM (WHO CNS5) as an inclusion criterion for its 337 patients. Dataset v2.0.0 (OpenNeuro CHANGES, 2026-09-04) added Karlsruhe subjects KA103, KA117, KA119–128, KA130–136 and KA138–152 and removed six others. Its README says only "histologically confirmed primary glioblastoma", and participants.tsv has no IDH column. {int(madd.sum())} of the added subjects are among our 359.

{s6.to_markdown(index=False)}
""")

# S7 unseen-site features + age with training-site standardisation only (no per-site z-scoring of the held-out site)
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.pipeline import make_pipeline as _mp
from sklearn.linear_model import LogisticRegression as _LR
_f = pd.read_csv(R + '01_IDH_imaging_correlates/features.csv'); _f = _f[_f.error.isna() | (_f.error.astype(str).str.len() == 0)].drop(columns=['site'])
_d = cv[['patient_id', 'site', 'y', 'age']].merge(_f, on='patient_id')
_FE = ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'no_enhancement', 'necrosis_frac', 'edema_ratio_log', 'mismatch_index', 'mismatch_frac']
_d['necrosis_frac'] = _d.necrosis_frac.fillna(0.0); _d = _d.dropna(subset=_FE).drop_duplicates('patient_id').reset_index(drop=True)
assert len(_d) == 1393
_ZC = [c for c in _FE if c != 'no_enhancement']; _Z = _d.copy()
for c in _ZC: _Z[c] = _d.groupby('site')[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
_fit = lambda X, yy: _mp(_SS(), _LR(C=1.0, max_iter=5000)).fit(X, yy)
s7 = []; cols = _FE + ['age']
for site in ['UCSF', 'UTSW']:
    for name, D_ in [('own-site z (as published)', _Z), ('training-site statistics only', _d)]:
        tr, te = D_[D_.site != site], D_[D_.site == site]; p = _fit(tr[cols].values, tr.y.values).predict_proba(te[cols].values)[:, 1]
        yy = te.y.values; g_ = [np.where(yy == v)[0] for v in (0, 1)]; rr = np.random.default_rng(SEED)
        v_ = [roc_auc_score(yy[i], p[i]) for i in (np.concatenate([rr.choice(x, len(x)) for x in g_]) for _ in range(2000))]
        s7.append({'held-out site': site, 'feature standardisation': name, 'Features + age AUROC': round(roc_auc_score(yy, p), 3), 'ci': f'{np.percentile(v_,2.5):.3f}-{np.percentile(v_,97.5):.3f}'})
s7 = pd.DataFrame(s7); s7.to_csv(OUT + 'S7_unseen_site_training_statistics.csv', index=False)
open(OUT + 'REPORT.md', 'a').write(f"""
## S7. Unseen-site features + age without per-site z-scoring of the held-out site

Features are standardised by the pipeline's scaler fitted on the training sites only, so the held-out site gets no label-free adaptation. Requested by the Stage 2 review of v8. The networks are unaffected; the ensemble scored 0.916 (UCSF) and 0.826 (UTSW).

{s7.to_markdown(index=False)}
""")

# S8 reserved-set sensitivity at the frozen thresholds, all mutants and grade-4 mutants
s8 = []
for op in ['threshold_youden', 'threshold_sens90']:
    for m in ['Features + age', 'Ensemble (raw)', 'Hybrid']:
        for g_, df in [('all', rp), ('grade 4', rp[rp.grade_group == 'grade 4'])]:
            mu = df[df.y == 1]; k = int((mu[m] >= d['models'][m][op]).sum()); n = len(mu); c_ = _bt(k, n).proportion_ci(method='exact')
            s8.append({'operating point': op, 'model': m, 'mutants': g_, 'detected': f'{k}/{n}', 'sensitivity %': round(100 * k / n, 1), 'ci': f'{100*c_.low:.1f}-{100*c_.high:.1f}'})
s8 = pd.DataFrame(s8); s8.to_csv(OUT + 'S8_reserved_sensitivity_frozen_thresholds.csv', index=False)
open(OUT + 'REPORT.md', 'a').write(f"""
## S8. Reserved-set sensitivity at the frozen thresholds (all and grade-4 mutants)

{s8.to_markdown(index=False)}
""")
