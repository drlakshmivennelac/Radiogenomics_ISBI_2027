"""Exploratory equal-weight subset sweep of completed, aligned IDH OOF runs."""
from pathlib import Path
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_curve
from clinical_quality import read_predictions, metrics, save_json, save_csv, bootstrap

def main():
    project=Path(__file__).resolve().parent
    old=Path.home()/'Downloads/outputs/clinical_experiments'
    local=project/'outputs/clinical_experiments'
    paths={**{f'CA{s}':old/f'v2_reference_seed{s}' for s in (42,43,44)},
           'MC42':old/'v2_mean_concat_seed42',
           **{f'MC{s}':local/f'v2_mean_concat_seed{s}' for s in (43,44)},
           'T1c42':old/'v2_t1c_only_seed42',
           'Difference42':Path.home()/'Downloads/T2-Flair_RadGenome/difference_joint_seed42_matched01',
           'T2MeanConcat42':Path.home()/'Downloads/T2-Flair_RadGenome/mean_concat_joint_seed42_matched01'}
    loaded={k:read_predictions(p) for k,p in paths.items()}
    rows=loaded['CA42']
    signature=lambda rs:[(r['patient_id'],r['site'],r['fold'],r['y']) for r in rs]
    for k,rs in loaded.items():
        if signature(rs)!=signature(rows): raise ValueError('Cohort/fold mismatch: '+k)
    probs={k:np.array([r['p'] for r in rs]) for k,rs in loaded.items()}
    y=np.array([r['y'] for r in rows]); threshold=.14600733667612076
    baseline=('CA42','CA44','MC42'); base=np.mean([probs[k] for k in baseline],axis=0)
    records=[]; ensembles={}
    for size in range(1,len(probs)+1):
        for names in combinations(probs,size):
            key='+'.join(names); p=np.mean([probs[k] for k in names],axis=0)
            fpr,tpr,ts=roc_curve(y,p,drop_intermediate=False)
            best=max(range(len(ts)),key=lambda i:(tpr[i]-fpr[i],-abs(tpr[i]-(1-fpr[i])),-abs(ts[i]-.5)))
            optimum=metrics(y,p,float(ts[best])); frozen=metrics(y,p,threshold)
            records.append(dict(members=key,size=size,auroc=frozen['auroc'],average_precision=frozen['average_precision'],
                brier=frozen['brier'],frozen_sensitivity=frozen['sensitivity'],frozen_specificity=frozen['specificity'],
                youden_threshold=optimum['threshold'],youden_sensitivity=optimum['sensitivity'],
                youden_specificity=optimum['specificity'],youden_precision=optimum['precision'],
                youden_balanced_accuracy=optimum['balanced_accuracy'],youden_macro_f1=optimum['macro_f1']))
            ensembles[key]=p
    records.sort(key=lambda r:r['auroc'],reverse=True)
    selected={'original_three':next(r for r in records if set(r['members'].split('+'))==set(baseline)),
              'best_auc':records[0],
              'best_ap':max(records,key=lambda r:r['average_precision']),
              'best_brier':min(records,key=lambda r:r['brier']),
              'best_youden_balanced_accuracy':max(records,key=lambda r:r['youden_balanced_accuracy']),
              'best_auc_with_difference':next(r for r in records if 'Difference42' in r['members'])}
    cis={}
    for name,r in selected.items():
        if name=='original_three': continue
        key=r['members']
        if key not in cis:
            print('Bootstrap:',key,flush=True)
            cis[key]=bootstrap([dict(row,p=float(v)) for row,v in zip(rows,ensembles[key])],threshold,reps=2000,other=base)
    out=project/'outputs/ensemble_subset_sweep_20261012';out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'all_subsets.csv',records)
    report=dict(n=len(rows),combinations=len(records),sources={k:str(v) for k,v in paths.items()},
        note='Exploratory OOF subset and threshold selection; bootstrap intervals are conditional and not corrected for selection across subsets.',
        selected=selected,paired_delta_vs_original_95ci=cis,top10_auc=records[:10])
    save_json(out/'summary.json',report)
    print(__import__('json').dumps(selected,indent=2))
    print('Saved:',out)

if __name__=='__main__':main()
