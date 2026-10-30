"""Patient-level development audits; no inference or training on import."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_manifest(run):
    with (Path(run)/'fold_manifest.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for t in ('idh','mgmt'):
            r[t]=None if r[t] in ('','None') else int(r[t]);r[t+'_observed']=r[t] is not None
        r['cv_fold']=int(r['cv_fold']);r['eligible']=True
        for k in ('baseline',):
            if k in r:r[k]=str(r[k]).lower()=='true'
    if len({r['patient_id'] for r in rows})!=len(rows):raise ValueError('Duplicate patients')
    return rows

def read_predictions(run,task='idh'):
    run=Path(run)
    if not (run/'summary.json').exists():raise ValueError('Complete all folds first')
    manifest={r['patient_id']:r for r in read_manifest(run)}
    with (run/'oof_predictions.csv').open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    if len({r['patient_id'] for r in raw})!=len(raw):raise ValueError('Duplicate predictions')
    if {int(r['fold']) for r in raw}!={1,2,3,4,5}:raise ValueError('Expected all five OOF folds')
    out=[]
    for r in raw:
        m=manifest[r['patient_id']]
        if m['role']!='development' or int(r['fold'])!=m['cv_fold']+1:raise ValueError('Non-OOF patient/fold')
        if r['site']!=m['site']:raise ValueError('Site mismatch')
        label=r[task+'_label']
        if label=='':continue
        if int(label)!=m[task]:raise ValueError('Label mismatch')
        p=float(r[task+'_probability'])
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
        out.append(dict(patient_id=r['patient_id'],site=r['site'],fold=int(r['fold']),y=int(label),p=p,
            mask_source=m.get('tumor_mask_source','unknown'),assay=m.get(task+'_testing_method','') or m.get(task+'_scope','unknown'),
            biopsy=m.get('biopsy_prior_to_imaging','unknown')))
    expected={r['patient_id'] for r in manifest.values() if r['role']=='development' and r[task] is not None}
    if {r['patient_id'] for r in out}!=expected:raise ValueError('Missing known-label OOF patients')
    return sorted(out,key=lambda r:r['patient_id'])

def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=p>=threshold
    tp=int(sum(pred&(y==1)));tn=int(sum(~pred&(y==0)));fp=int(sum(pred&(y==0)));fn=int(sum(~pred&(y==1)))
    div=lambda a,b:a/b if b else None
    sen=div(tp,tp+fn);spe=div(tn,tn+fp)
    fpos=div(2*tp,2*tp+fp+fn);fneg=div(2*tn,2*tn+fp+fn)
    return dict(n=len(y),positive=int(sum(y)),negative=int(sum(1-y)),threshold=threshold,
        tp=tp,tn=tn,fp=fp,fn=fn,auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if len(set(y))==2 else None,
        sensitivity=sen,specificity=spe,precision=div(tp,tp+fp),npv=div(tn,tn+fn),
        accuracy=div(tp+tn,len(y)),balanced_accuracy=(sen+spe)/2 if sen is not None and spe is not None else None,
        macro_f1=(fpos+fneg)/2 if fpos is not None and fneg is not None else None,
        brier=float(np.mean((p-y)**2)),log_loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])))

def bootstrap(rows,threshold=.5,reps=2000,seed=42,other=None):
    """Conditional patient bootstrap stratified by site and class; paired if other supplied."""
    rng=np.random.default_rng(seed);y=np.array([r['y'] for r in rows]);p=np.array([r['p'] for r in rows])
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['site'],r['y']),[]).append(i)
    keys=('auroc','average_precision','sensitivity','specificity','balanced_accuracy','macro_f1','brier')
    samples={k:[] for k in keys}
    for _ in range(reps):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups.values()])
        a=metrics(y[idx],p[idx],threshold)
        b=metrics(y[idx],np.asarray(other)[idx],threshold) if other is not None else None
        for k in keys:
            if a[k] is not None and (b is None or b[k] is not None):samples[k].append(a[k]-(b[k] if b else 0))
    return {k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),valid_replicates=len(v))
            for k,v in samples.items() if v}

def reliability_bins(rows,bins=10):
    y=np.array([r['y'] for r in rows]);p=np.array([r['p'] for r in rows]);out=[]
    for i in range(bins):
        m=(p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else (p<=1))
        out.append(dict(lower=i/bins,upper=(i+1)/bins,n=int(m.sum()),
            mean_probability=float(p[m].mean()) if m.any() else None,
            observed_fraction=float(y[m].mean()) if m.any() else None))
    return out

def audit_run(run,output,task='idh',threshold=.5,reps=2000):
    rows=read_predictions(run,task);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    y=[r['y'] for r in rows];p=[r['p'] for r in rows]
    report=dict(source=str(Path(run).resolve()),task=task,assessment='development OOF, not independent test',
        uncertainty='95% site/class-stratified patient bootstrap, conditional on fitted models and fixed threshold; excludes training/selection uncertainty',
        metrics=metrics(y,p,threshold),ci=bootstrap(rows,threshold,reps),threshold_origin='user-specified; record whether previously selected on these data')
    bins=reliability_bins(rows)
    report['ece_10_equal_width_bins']=sum(b['n']*abs(b['mean_probability']-b['observed_fraction']) for b in bins if b['n'])/len(rows)
    save_csv(output/'calibration.csv',bins)
    subgroups=[]
    for group in ('site','fold','mask_source','assay','biopsy'):
        for value in sorted({str(r[group]) for r in rows}):
            sub=[r for r in rows if str(r[group])==value]
            subgroups.append(dict(group=group,value=value,**metrics([r['y'] for r in sub],[r['p'] for r in sub],threshold)))
    save_csv(output/'subgroups.csv',subgroups)
    failures=[]
    for r in rows:
        pred=int(r['p']>=threshold)
        failures.append(dict(**r,prediction=pred,category=('TP' if r['y'] else 'FP') if pred else ('FN' if r['y'] else 'TN'),absolute_error=abs(r['p']-r['y'])))
    save_csv(output/'patient_predictions.csv',sorted(failures,key=lambda r:-r['absolute_error']))
    # Decision curves retain observed prevalence; no artificial balancing.
    curves=[]
    for t in np.arange(.05,.96,.05):
        m=metrics(y,p,float(t));odds=t/(1-t);prev=np.mean(y)
        curves.append(dict(threshold=float(t),model_net_benefit=(m['tp']-m['fp']*odds)/len(y),
                           treat_all=float(prev-(1-prev)*odds),treat_none=0.))
    save_csv(output/'decision_curve_exploratory.csv',curves)
    report['decision_curve_note']='Exploratory only: probability calibration and a defined clinical decision/cost ratio are required for clinical utility claims.'
    save_json(output/'audit.json',report);return report

def compare_runs(a,b,output,task='idh',threshold=.5,reps=2000):
    ra=read_predictions(a,task);rb=read_predictions(b,task)
    signature=lambda rs:[(r['patient_id'],r['site'],r['fold'],r['y']) for r in rs]
    if signature(ra)!=signature(rb):raise ValueError('Paired comparison requires identical patients, labels and folds')
    ma=metrics([r['y'] for r in ra],[r['p'] for r in ra],threshold)
    mb=metrics([r['y'] for r in rb],[r['p'] for r in rb],threshold)
    report=dict(a=str(a),b=str(b),difference='A minus B; negative Brier/log-loss differences favour A',
        point_difference={k:ma[k]-mb[k] for k in ('auroc','average_precision','brier','balanced_accuracy')},
        conditional_ci=bootstrap(ra,threshold,reps,other=[r['p'] for r in rb]),
        caveat='Exploratory paired patient bootstrap; excludes training variance and multiple-comparison correction.')
    save_json(output,report);return report

def seed_ensemble(runs,output,task='idh'):
    if len(runs)<2:raise ValueError('Provide at least two seed runs')
    settings=[];seeds=[]
    for run in runs:
        path=Path(run)/'study_config.json'
        if not path.exists():raise ValueError('Seed ensemble requires study_config.json provenance')
        cfg=json.loads(path.read_text());cfg.pop('fingerprint',None)
        seeds.append(cfg['training'].pop('seed'));settings.append(cfg)
    if len(set(seeds))!=len(seeds):raise ValueError('Use distinct training seeds')
    if any(s!=settings[0] for s in settings[1:]):raise ValueError('Seed runs differ beyond initialization/shuffle seed')
    all_rows=[read_predictions(r,task) for r in runs]
    sig=lambda rs:[(r['patient_id'],r['y'],r['fold'],r['site']) for r in rs]
    if any(sig(r)!=sig(all_rows[0]) for r in all_rows):raise ValueError('OOF seed ensemble needs fixed patient folds')
    # Each row is already validated against its run manifest; never apply other folds' models here.
    rows=[dict(r,p=float(np.mean([rs[i]['p'] for rs in all_rows]))) for i,r in enumerate(all_rows[0])]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_csv(output/'ensemble_predictions.csv',rows)
    report=dict(runs=list(map(str,runs)),task=task,metrics=metrics([r['y'] for r in rows],[r['p'] for r in rows]),
                method='equal mean of same-patient OOF probabilities across fixed-fold seed runs')
    save_json(output/'summary.json',report);return report

def select_review_cases(run,output,task='idh',threshold=.5,per_site_category=1,seed=42):
    rows=read_predictions(run,task);rng=np.random.default_rng(seed);groups={}
    for r in rows:
        pred=r['p']>=threshold;category=('TP' if r['y'] else 'FP') if pred else ('FN' if r['y'] else 'TN')
        groups.setdefault((r['site'],category),[]).append(r)
    chosen=[]
    for (site,category),group in sorted(groups.items()):
        for i in rng.choice(len(group),min(per_site_category,len(group)),replace=False):
            chosen.append(dict(review_id=f'case_{len(chosen)+1:03d}',**group[int(i)],category=category))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_csv(output/'private_case_key.csv',chosen)
    template=[dict(review_id=r['review_id'],reader='',image_quality_1_to_5='',alignment_acceptable='',
        attention_location='',tumour_focus_1_to_5='',context_plausibility_1_to_5='',artifact_attention='',
        confidence_1_to_5='',comments='') for r in chosen]
    save_csv(output/'blinded_reader_form.csv',template)
    save_json(output/'sampling.json',dict(seed=seed,threshold=threshold,source=str(run),
        strategy='random within site x TP/TN/FP/FN; not a representative prevalence sample',
        note='Keep private_case_key separate from blinded maps; review heatmaps after an image-only assessment.'))
    return chosen

def site_prior_baseline(run,task='idh'):
    rows=read_predictions(run,task);pred=[]
    for r in rows:
        training=[x for x in rows if x['fold']!=r['fold'] and x['site']==r['site']]
        # Excludes this patient's entire validation fold; uses labels only, never images.
        p=(sum(x['y'] for x in training)+1)/(len(training)+2)
        pred.append(p)
    return dict(metrics=metrics([r['y'] for r in rows],pred),
                note='Site-only smoothed prevalence predictor fitted outside each validation fold; diagnostic for dataset shortcuts.')
