import nbformat, numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
nb=nbformat.read('/home/claude/mgmt_nb/IDH_Hybrid_Analysis.ipynb',4)
def after(prefix):
    for i,c in enumerate(nb.cells):
        if c.cell_type=='markdown' and c.source.lstrip().startswith(prefix): return nb.cells[i+1].source
g={'display':lambda *a,**k: None}
exec(after('## 1.').replace("'idh_hybrid'","'idh_sens'"),g); exec(after('## 2.'),g); exec(after('## 3.'),g)
d,Z,MODELS,CONV,MISM=g['d'],g['Z'],g['MODELS'],g['CONV'],g['MISM']
print(d.columns.tolist()[:40]); print({m:c for m,c in MODELS.items()})
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
fit=g['fit']; SEED=g['SEED']
print('Z has dl_logit', 'dl_logit' in Z, len(Z), int(Z.y.sum()))
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
T2=Path=None
base='/mnt/user-data/uploads/Downloads/T2-Flair_RadGenome/'
t2=pd.read_csv(base+'cross_attention_joint_seed42_matched01/ensemble_comparison/paired_predictions.csv')[['patient_id','t2_model']].rename(columns={'t2_model':'t2_ca'})
for m,c in [('mean_concat','t2_mc'),('difference','t2_diff')]:
    t2=t2.merge(pd.read_csv(base+f'{m}_joint_seed42_matched01/ensemble_comparison/paired_predictions.csv')[['patient_id','t2_model']].rename(columns={'t2_model':c}),on='patient_id')
t2['t2_logit']=lg(t2[['t2_ca','t2_mc','t2_diff']].mean(1))
for c in ['t2_ca','t2_mc','t2_diff']: t2[c+'_l']=lg(t2[c])
Zt=Z.merge(t2,on='patient_id',how='left'); print('T2 matched', Zt.t2_logit.notna().sum(),'of',len(Zt))
for c in ['t2_logit','t2_ca_l','t2_mc_l','t2_diff_l']:
    Zt[c]=(Zt[c]-Zt[c].mean())/Zt[c].std()
y=Zt.y.values; strat=(Zt.site+'_'+Zt.y.astype(str)).values
H=MODELS['Hybrid']
SPECS={'Hybrid':H,'Hybrid + T2 network':H+['t2_logit'],'Ensemble':['dl_logit'],'Ensemble + T2 network':['dl_logit','t2_logit'],
       'T2 network (recal.)':['t2_logit'],'Features + age':MODELS['Features + age'],'Features + age + T2 network':MODELS['Features + age']+['t2_logit']}
P={m:np.zeros((5,len(Zt))) for m in SPECS}
for r in range(5):
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED+r).split(Zt,strat):
        for m,cols in SPECS.items(): P[m][r,te]=fit(Zt[cols].values[tr],y[tr]).predict_proba(Zt[cols].values[te])[:,1]
S={m:P[m].mean(0) for m in P}
rng=np.random.default_rng(7); G=[np.where(strat==s)[0] for s in np.unique(strat)]
B=[np.concatenate([rng.choice(x,len(x)) for x in G]) for _ in range(2000)]
def dci(a,b):
    v=[roc_auc_score(y[i],S[a][i])-roc_auc_score(y[i],S[b][i]) for i in B]
    return roc_auc_score(y,S[a])-roc_auc_score(y,S[b]), np.percentile(v,[2.5,97.5])
print('\n--- T2 network as an extra input (n=%d, %d mutant)'%(len(y),y.sum()))
for m in S: print(f'{m:30s} {roc_auc_score(y,S[m]):.3f}')
for a,b in [('Hybrid + T2 network','Hybrid'),('Ensemble + T2 network','Ensemble'),('Features + age + T2 network','Features + age')]:
    pt,ci=dci(a,b); print(f'{a} minus {b}: {pt:+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f})')
# --- full-pipeline LOSO for UCSF with the CA42 member
oof=pd.read_csv('/mnt/user-data/uploads/outputs/clinical_experiments/v2_reference_seed42/oof_predictions.csv')
lo=pd.read_csv('/mnt/user-data/uploads/outputs/clinical_experiments/loso_UCSF/outer_site_predictions.csv')
print('\nloso cols',lo.columns.tolist()[:12]); print('oof cols',oof.columns.tolist()[:10])
oof['ca42_cv']=oof.idh_logit; lo['ca42_loso']=lg(lo.idh_probability)
D=d.merge(oof[['patient_id','ca42_cv']],on='patient_id').merge(lo[['patient_id','ca42_loso']],on='patient_id',how='left')
FE=MODELS['Features + age']
def zsite(F,cols):
    F=F.copy()
    for c in cols:
        if c=='no_enhancement': continue
        for s in F.site.unique():
            k=F.site==s; F.loc[k,c]=(F.loc[k,c]-F.loc[k,c].mean())/(F.loc[k,c].std(ddof=0)+1e-9)
    return F
D=zsite(D,[c for c in FE if c!='age']); D['age']=(D.age-D.age.mean())/D.age.std()
tr=D[D.site!='UCSF']; te=D[D.site=='UCSF']; yt=te.y.values
print('\n--- Full-pipeline leave-UCSF-out (test n=%d, %d mutant)'%(len(te),yt.sum()))
print('CA42 alone, in-CV logits on UCSF         %.3f'%roc_auc_score(yt,te.ca42_cv))
print('CA42 alone, retrained without UCSF       %.3f'%roc_auc_score(yt,te.ca42_loso))
out={}
for name,cols in [('Features + age',FE),('Hybrid (CA42)',['net']+FE),('CA42 + age',['net','age'])]:
    Xtr=tr.assign(net=tr.ca42_cv)[cols].values; m=fit(Xtr,tr.y.values)
    for src in ['ca42_cv','ca42_loso']:
        p=m.predict_proba(te.assign(net=te[src])[cols].values)[:,1]; out[(name,src)]=p
        print(f'{name:16s} combiner on UTSW+UPENN, UCSF net={src:9s}: {roc_auc_score(yt,p):.3f}')
# bootstrap CI for hybrid full LOSO and difference vs features+age
rng=np.random.default_rng(3); st=yt
G2=[np.where(st==v)[0] for v in (0,1)]; BB=[np.concatenate([rng.choice(x,len(x)) for x in G2]) for _ in range(2000)]
h=out[('Hybrid (CA42)','ca42_loso')]; f=out[('Features + age','ca42_loso')]
ci=np.percentile([roc_auc_score(yt[i],h[i]) for i in BB],[2.5,97.5]); print('hybrid full-LOSO CI',ci.round(3))
dv=np.percentile([roc_auc_score(yt[i],h[i])-roc_auc_score(yt[i],f[i]) for i in BB],[2.5,97.5]); print('hybrid - F+A', round(roc_auc_score(yt,h)-roc_auc_score(yt,f),3), dv.round(3))
print('net LOSO CI', np.percentile([roc_auc_score(yt[i],te.ca42_loso.values[i]) for i in BB],[2.5,97.5]).round(3))
# --- clean version: combiner trained on the LOSO model's own inner-CV OOF logits for UTSW+UPENN
ioof=pd.read_csv('/mnt/user-data/uploads/outputs/clinical_experiments/loso_UCSF/inner_cv/oof_predictions.csv')
print('\ninner oof', len(ioof), ioof.site.value_counts().to_dict())
D2=D.merge(ioof[['patient_id','idh_probability']].rename(columns={'idh_probability':'p_in'}),on='patient_id',how='left')
D2['net_clean']=np.where(D2.site=='UCSF', D2.ca42_loso, lg(D2.p_in.fillna(.5)))
tr2=D2[D2.site!='UCSF']; te2=D2[D2.site=='UCSF']; print('train n',len(tr2), 'with inner oof', tr2.p_in.notna().sum())
print('LOSO net inner-CV AUROC on UTSW+UPENN %.3f'%roc_auc_score(tr2.y,tr2.net_clean))
outc={}
for name,cols in [('Features + age',FE),('Hybrid (CA42, clean)',['net_clean']+FE),('CA42 + age',['net_clean','age']),('CA42 recal',['net_clean'])]:
    m=fit(tr2[cols].values,tr2.y.values); p=m.predict_proba(te2[cols].values)[:,1]; outc[name]=p
    ci=np.percentile([roc_auc_score(yt[i],p[i]) for i in BB],[2.5,97.5])
    print(f'{name:22s} UCSF AUROC {roc_auc_score(yt,p):.3f} ({ci[0]:.3f}-{ci[1]:.3f})')
for a in ['Features + age','CA42 + age','CA42 recal']:
    h=outc['Hybrid (CA42, clean)']; f=outc[a]
    dv=np.percentile([roc_auc_score(yt[i],h[i])-roc_auc_score(yt[i],f[i]) for i in BB],[2.5,97.5])
    print(f'hybrid - {a}: {roc_auc_score(yt,h)-roc_auc_score(yt,f):+.3f} ({dv[0]:+.3f} to {dv[1]:+.3f})')
