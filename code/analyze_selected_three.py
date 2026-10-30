"""CPU-only reliability analysis for CA42 + CA44 + mean-concat42."""
from datetime import datetime
from itertools import combinations
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from clinical_quality import read_predictions, metrics, save_json, save_csv, bootstrap
from analyze_existing_results import boot_models, intervals, threshold_samples, calibration
from analyze_ensemble_voting import youden, retained_mask
from mean_concat_seeds import OLD, PROJECT, aligned_probabilities
from compare_mean_concat44 import THREE_THRESHOLD


def run():
    names=['v2_reference_seed42','v2_reference_seed44','v2_mean_concat_seed42','v2_reference_seed43']
    data=[read_predictions(OLD/n) for n in names]
    matrix=aligned_probabilities(data); ps=matrix[:,:3]
    rows=data[0]; y=np.array([r['y'] for r in rows]); sites=np.array([r['site'] for r in rows])
    folds=np.array([r['fold'] for r in rows]); p=ps.mean(1); t=THREE_THRESHOLD
    assert np.isclose(metrics(y,p)['auroc'],.9238936218726033)
    output=PROJECT/'outputs'/('selected_three_reliability_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    caveat=('Exploratory development OOF analyses after model selection. 2,000 site/class-stratified patient bootstrap resamples condition on fitted models and class/site counts; exclude retraining and model-selection uncertainty. Threshold optima and gating are not independently validated. Multiple comparisons are unadjusted.')
    result=dict(members=names[:3],weights=[1/3]*3,threshold=t,caveat=caveat,
        primary=metrics(y,p,t),default=metrics(y,p,.5))
    result['primary']['positive_f1']=2*result['primary']['tp']/(2*result['primary']['tp']+result['primary']['fp']+result['primary']['fn'])
    print('Paired member, pair and original-four comparisons...',flush=True)
    models={name:ps[:,i] for i,name in enumerate(names[:3])}
    models.update({'+'.join(names[i] for i in pair):ps[:,pair].mean(1) for pair in combinations(range(3),2)})
    models.update(selected_three=p,original_four=matrix.mean(1))
    samples=boot_models(y,sites,models)
    result['models']={name:dict(metrics=metrics(y,prob,t),ci=intervals(samples[name]),
        selected_three_minus_model_ci=intervals({k:samples['selected_three'][k]-samples[name][k] for k in samples[name]})) for name,prob in models.items()}
    ensemble_rows=[dict(r,p=float(p[i])) for i,r in enumerate(rows)]
    result['fixed_threshold_ci']=bootstrap(ensemble_rows,threshold=t,reps=2000)
    result['calibration']={'all':calibration(y,p)}
    result['sites']={}
    print('Site-specific reliability...',flush=True)
    for site in sorted(set(sites)):
        m=sites==site; sub=[ensemble_rows[i] for i in np.flatnonzero(m)]
        result['sites'][str(site)]=dict(metrics=metrics(y[m],p[m],t),ci=bootstrap(sub,threshold=t,reps=2000))
        result['calibration'][str(site)]=calibration(y[m],p[m])
    pairs={s:int(sum(y[sites==s]==1)*sum(y[sites==s]==0)) for s in sorted(set(sites))}
    result['within_site_pair_weighted_auc']=sum(result['sites'][s]['metrics']['auroc']*pairs[s] for s in pairs)/sum(pairs.values())
    result['within_site_macro_auc']=float(np.mean([a['metrics']['auroc'] for a in result['sites'].values()]))
    print('Resampling Youden thresholds...',flush=True)
    ts=threshold_samples(y,p,sites)
    result['threshold_stability']=dict(quantiles=dict(zip(['p2.5','p25','median','p75','p97.5'],map(float,np.quantile(ts,[.025,.25,.5,.75,.975])))),
        note='Distribution of reselected apparent thresholds, not an interval for the externally optimal threshold.',
        nearby=[metrics(y,p,float(x)) for x in (.10,.12,.13,.14,t,.15,.16,.18,.20,.25,.30,.40,.50)],
        per_fold_exploratory=[dict(fold=int(f),**youden(y[folds==f],p[folds==f])) for f in sorted(set(folds))])
    save_csv(output/'threshold_bootstrap.csv',[dict(replicate=i,threshold=v) for i,v in enumerate(ts)])
    save_csv(output/'threshold_sweep.csv',[metrics(y,p,float(x)) for x in np.linspace(0,1,1001)])
    # Threshold transfer across OOF folds is diagnostic only: other-fold models may have
    # trained on this fold. This is not nested CV or independent threshold validation.
    transfer=[]; predictions=np.zeros(len(y))
    for f in sorted(set(folds)):
        m=folds==f; cutoff=youden(y[~m],p[~m])['threshold']; predictions[m]=(p[m]>=cutoff)
        transfer.append(dict(fold=int(f),**metrics(y[m],p[m],cutoff)))
    result['threshold_transfer_diagnostic']=dict(folds=transfer,
        aggregate_classification={k:v for k,v in metrics(y,predictions,.5).items() if k not in ('auroc','average_precision','brier','log_loss','threshold')},
        caveat='Other OOF folds select each cutoff, but their generating models may have trained on target-fold patients. Diagnostic only; not independent or nested validation.')
    median=np.median(ps,axis=1)
    result['voting']=dict(mean_fixed=metrics(y,p,t),median_fixed=metrics(y,median,t),median_exploratory_youden=youden(y,median))
    for cutoff in (.5,t):
        votes=(ps>=cutoff).sum(1); hard=metrics(y,votes/3,.5)
        for k in ('brier','log_loss'):hard.pop(k)
        hard['vote_fraction_auc']=hard.pop('auroc');hard['vote_fraction_ap']=hard.pop('average_precision')
        result['voting'][f'hard_majority_{cutoff}']=hard
        assert np.array_equal(votes>=2,median>=cutoff)
    print('Error overlap and selective prediction...',flush=True)
    errors=(p>=t)!=y; member_errors=(ps>=t)!=y[:,None]
    persistent=member_errors.all(1)
    result['error_overlap']=dict(member_names=names[:3],probability_correlations=np.corrcoef(ps.T).tolist(),
        member_error_counts=member_errors.sum(0).tolist(),ensemble_errors=int(errors.sum()),
        all_members_wrong=int(persistent.sum()),all_wrong_positive=int(sum(persistent&(y==1))),all_wrong_negative=int(sum(persistent&(y==0))),
        pair_error_jaccard={names[a]+'__'+names[b]:float(sum(member_errors[:,a]&member_errors[:,b])/sum(member_errors[:,a]|member_errors[:,b])) for a,b in combinations(range(3),2)})
    save_csv(output/'patient_predictions.csv',[dict(patient_id=r['patient_id'],site=r['site'],fold=r['fold'],label=r['y'],probability=float(p[i]),
        category=('TP' if y[i] else 'FP') if p[i]>=t else ('FN' if y[i] else 'TN'),all_members_wrong=bool(persistent[i]),
        **{n:float(ps[i,j]) for j,n in enumerate(names[:3])}) for i,r in enumerate(rows)])
    std=ps.std(1);margin=-abs(p-t)
    uncertainties={'member_disagreement':std,'threshold_margin':margin,'combined_rank':(rankdata(std)+rankdata(margin))/(2*len(y))}
    result['gating']={}; curve=[]
    for name,u in uncertainties.items():
        entries=[]
        for coverage in (.95,.90,.80):
            keep,cutoff=retained_mask(u,coverage)
            entries.append(dict(target_coverage=coverage,actual_coverage=float(keep.mean()),cutoff=float(cutoff),
                retained=metrics(y[keep],p[keep],t),positive_coverage=float(keep[y==1].mean()),
                deferred=int(sum(~keep)),deferred_positive=int(sum(y[~keep])),deferred_error_rate=float(errors[~keep].mean()),
                retained_error_rate=float(errors[keep].mean()),fraction_errors_deferred=float(sum(errors[~keep])/sum(errors)),
                by_site={str(s):dict(coverage=float(keep[sites==s].mean()),positive_coverage=float(keep[(sites==s)&(y==1)].mean())) for s in sorted(set(sites))}))
        result['gating'][name]=dict(error_detection_auc=float(roc_auc_score(errors,u)),coverage=entries)
        for coverage in np.linspace(.1,1,91):
            keep,_=retained_mask(u,float(coverage));curve.append(dict(method=name,coverage=float(keep.mean()),risk=float(errors[keep].mean()),positive_coverage=float(keep[y==1].mean())))
    result['gating_caveat']='Retained-subset performance is conditional on selection. Deferred patients are not correct predictions. Raw disagreement/margin are not calibrated confidence; cutoffs require independent validation.'
    save_csv(output/'risk_coverage.csv',curve)
    save_json(output/'results.json',result)
    make_report(result,output,ts,curve)
    print('OUTPUT:',output,flush=True)
    return result,output


def make_report(r,out,ts,curve):
    lines=['# Selected three-member ensemble: reliability analyses','',r['caveat'],'',
        'Members: cross-attention seeds42/44 and mean-concat seed42, equal weights.','',
        '## Primary IDH metrics','',json.dumps(r['primary'],indent=2),'',
        '## Threshold stability','',json.dumps(r['threshold_stability'],indent=2),'',
        '## Site results','', '| Site | N / positive | AUC | Sensitivity | Specificity |','|---|---:|---:|---:|---:|']
    for s,a in r['sites'].items():
        m=a['metrics'];lines.append(f"| {s} | {m['n']} / {m['positive']} | {m['auroc']:.4f} | {m['sensitivity']:.1%} | {m['specificity']:.1%} |")
    for name,key in [('Confidence intervals','fixed_threshold_ci'),('Members, pairs and original four','models'),('Calibration','calibration'),('Error overlap','error_overlap'),('Voting','voting'),('Gating','gating'),('Threshold-transfer diagnostic','threshold_transfer_diagnostic')]:
        lines+=['','## '+name,'','```json',json.dumps(r[key],indent=2),'```']
    lines+=['',r['gating_caveat']]
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(16,4.5))
    axs[0].hist(ts,bins=35);axs[0].axvline(r['threshold'],color='red',label='Selected threshold');axs[0].legend()
    axs[0].set(title='Youden threshold resampling',xlabel='Threshold',ylabel='Resamples')
    for name,c in r['calibration'].items():
        bins=[b for b in c['bins'] if b['n']>=10]
        axs[1].plot([b['mean_probability'] for b in bins],[b['observed_fraction'] for b in bins],marker='o',label=name)
    axs[1].plot([0,1],[0,1],'k--');axs[1].legend();axs[1].set(title='Calibration (bins n ≥ 10)',xlabel='Mean probability',ylabel='Observed positive fraction')
    for name in r['gating']:
        sub=[a for a in curve if a['method']==name]
        axs[2].plot([a['coverage'] for a in sub],[a['risk'] for a in sub],label=name)
    axs[2].legend();axs[2].set(title='Selective prediction: exploratory',xlabel='Retained fraction',ylabel='Retained error rate')
    fig.tight_layout();fig.savefig(out/'reliability.png',dpi=160);plt.close(fig)

if __name__=='__main__':run()
