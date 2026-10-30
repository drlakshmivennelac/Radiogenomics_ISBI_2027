"""Frozen preoperative RHUH IDH inference; no training or threshold optimisation."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,gc
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from scipy.stats import beta
from clinical_quality import metrics,save_json,save_csv,bootstrap
from research_experiments import load_checkpoint,dataset_for

PROJECT=Path(__file__).resolve().parent
BASE=Path.home()/'Downloads/outputs/clinical_experiments'
NAMES=['v2_reference_seed42','v2_reference_seed44','v2_mean_concat_seed42']
THRESHOLD=.14600733667612076
OUT=PROJECT/'outputs/rhuh_external_three'
FIELDS=['subject_id','timepoint','t1c_path','flair_path','tumor_mask_path','brain_mask_path',
        'idh','prior_treatment','idh_original','prior_treatment_original','clinical_source','exclude_reason']

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');save_json(tmp,obj);tmp.replace(path)

def inventory(root,output=OUT):
    """List NIfTI files and propose only the documented Subject/0/t1ce layout."""
    root=Path(root).resolve();output=Path(output)
    if not root.is_dir():raise FileNotFoundError(f'Set DATA_ROOT to the downloaded NIfTI folder: {root}')
    files=sorted(p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz')))
    if not files:raise ValueError('No NIfTI files found; use the brain-extracted NIfTI release, not raw DICOM')
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'file_inventory.csv',[dict(relative_path=str(p.relative_to(root)),path=str(p)) for p in files])
    proposals=[]
    for p in files:
        if p.name not in ('t1ce.nii.gz','t1ce.nii') or p.parent.name!='0':continue
        folder=p.parent
        def find(stem):
            matches=[q for q in files if q.parent==folder and q.name in (stem+'.nii',stem+'.nii.gz')]
            if len(matches)>1:raise ValueError(f'Ambiguous {stem} in {folder}')
            return str(matches[0]) if matches else ''
        proposals.append(dict(zip(FIELDS,[folder.parent.name,'preoperative',str(p),find('flair'),
            find('segmentations'),'','','unknown','','','',''])))
    dest=output/'reviewed_manifest.csv'
    if not dest.exists():
        with dest.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(proposals)
    print(f'{len(files)} NIfTI files; {len(proposals)} proposed preoperative pairs. Review: {dest}')
    if not proposals:print('Release layout differs from the documented layout: fill the manifest using file_inventory.csv. No filenames/timepoints were guessed.')
    return dest

def attach_clinical(manifest,clinical,subject_column,idh_column,prior_column,
                    idh_map,prior_map,delimiter=','):
    """Explicit raw-value maps; no guessing numeric clinical codes or patient aliases."""
    manifest=Path(manifest)
    if (manifest.parent/'protocol.json').exists():raise ValueError('Protocol frozen: cannot change clinical data here')
    with manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    with Path(clinical).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter=delimiter)
        for col in (subject_column,idh_column,prior_column):
            if col not in reader.fieldnames:raise ValueError(f'Missing clinical column: {col}')
        records=list(reader)
    ids=[r[subject_column].strip() for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate clinical patient IDs')
    lookup=dict(zip(ids,records))
    for r in rows:
        if r['subject_id'] not in lookup:raise ValueError(f'No exact clinical ID match: {r["subject_id"]}')
        c=lookup[r['subject_id']];a=c[idh_column].strip();b=c[prior_column].strip()
        if a not in idh_map or b not in prior_map:raise ValueError(f'Unmapped clinical values: IDH={a!r}, prior={b!r}')
        label=idh_map[a];prior=prior_map[b]
        if label not in (None,0,1) or prior not in ('yes','no','unknown'):raise ValueError('Invalid mapping')
        r.update(idh='' if label is None else str(label),prior_treatment=prior,idh_original=a,
                 prior_treatment_original=b,clinical_source=str(Path(clinical).resolve()))
    save_csv(manifest,rows)

def read_manifest(path):
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(FIELDS)<=set(reader.fieldnames or []):raise ValueError('Missing manifest columns')
        raw=list(reader)
    if not raw:raise ValueError('Manifest is empty')
    if len({r['subject_id'].strip() for r in raw})!=len(raw):raise ValueError('One row per patient required')
    rows=[];excluded=[]
    for r in raw:
        r={k:(v or '').strip() for k,v in r.items()}
        if not r['subject_id']:raise ValueError('Missing subject ID')
        if r['exclude_reason']:
            excluded.append(dict(subject_id=r['subject_id'],reason=r['exclude_reason']));continue
        if r['timepoint']!='preoperative':raise ValueError('Only explicitly preoperative scans are allowed')
        if r['idh'] not in ('','0','1'):raise ValueError('IDH must be 0=wildtype, 1=mutant, or blank=unknown')
        if r['prior_treatment'] not in ('yes','no','unknown'):raise ValueError('Prior treatment must be yes/no/unknown')
        if not r['clinical_source']:raise ValueError('Record the clinical source for each patient')
        for k in ('t1c_path','flair_path','tumor_mask_path','brain_mask_path'):
            if not r[k]:
                if k in ('t1c_path','flair_path'):raise ValueError(f'Missing {k}')
                continue
            p=Path(r[k]);p=p if p.is_absolute() else path.parent/p
            if not p.is_file():raise FileNotFoundError(p)
            r[k]=str(p.resolve())
        label=int(r['idh']) if r['idh'] else None
        rows.append(dict(r,patient_id='RHUH:'+r['subject_id'],site='RHUH',idh=label,mgmt=None,
                         idh_observed=label is not None,mgmt_observed=False))
    if not rows:raise ValueError('No included patients')
    for key in ('t1c_path','flair_path'):
        if len({r[key] for r in rows})!=len(rows):raise ValueError(f'Same {key} assigned to multiple patients')
    return sorted(rows,key=lambda r:r['patient_id']),excluded

def image_audit(rows):
    audit=[]
    for r in rows:
        ref=None;entry=dict(patient_id=r['patient_id'],mask_status='missing_brain_fallback')
        for key in ('t1c_path','flair_path','tumor_mask_path','brain_mask_path'):
            if not r[key]:continue
            im=nib.load(r[key]);a=im.get_fdata(dtype=np.float32)
            if a.ndim!=3 or not np.isfinite(a).all() or not np.isfinite(im.affine).all() or abs(np.linalg.det(im.affine[:3,:3]))<1e-8:
                raise ValueError(f'Invalid finite 3D image: {r[key]}')
            if ref is None:ref=im
            elif im.shape!=ref.shape or not np.allclose(im.affine,ref.affine,atol=1e-4,rtol=0):
                raise ValueError(f'Grids differ for {r["subject_id"]}; registration review needed')
            if key in ('t1c_path','flair_path') and (np.count_nonzero(a)<1000 or np.std(a[a!=0])==0):raise ValueError('Empty/constant MRI')
            if key=='tumor_mask_path':
                if np.any(a<0) or not np.allclose(a,np.round(a)):raise ValueError('Expected discrete nonnegative tumour labels')
                entry['mask_status']='nonempty' if np.any(a>0) else 'empty_brain_fallback'
                entry['mask_labels']=np.unique(a).tolist()
        audit.append(entry)
    return audit

def stat(path):
    p=Path(path);s=p.stat();return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)

def freeze(manifest,output=OUT):
    rows,excluded=read_manifest(manifest);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checkpoints=[dict(member=n,fold=f,**stat(BASE/n/f'fold_{f}/best.pt')) for n in NAMES for f in range(1,6)]
    images={r['patient_id']:{k:stat(r[k]) for k in ('t1c_path','flair_path','tumor_mask_path','brain_mask_path') if r[k]} for r in rows}
    protocol=dict(version=1,rows=rows,excluded=excluded,images=images,checkpoints=checkpoints,
        threshold=THRESHOLD,member_weights=[1/3]*3,fold_weights=[.2]*5,tta=False,amp=False,
        aggregation='Mean sigmoid probability over five folds within each family, then equal mean over three families (15 checkpoints). OOF used one held-out fold per family; external fold averaging may change calibration.',
        primary='Preoperative patients explicitly prior_treatment=no; unknown history excluded from primary.',
        secondary='All preoperative patients, with previously treated and unknown-history strata reported separately.',
        selection='No external tuning, threshold search, balancing, or checkpoint selection.')
    dest=output/'protocol.json'
    if dest.exists():
        if json.loads(dest.read_text(encoding='utf-8'))!=protocol:raise ValueError('Frozen sources/settings changed; use a new output folder')
    else:
        audit=image_audit(rows);save_json(output/'image_audit.json',audit);atomic_json(dest,protocol)
    return rows,protocol

def score(rows,reps=2000):
    known=[r for r in rows if r['idh'] is not None]
    if not known:return dict(total=len(rows),labelled=0,metrics=None)
    y=np.array([r['idh'] for r in known]);p=np.array([r['idh_probability'] for r in known])
    m=metrics(y,p,THRESHOLD);den=2*m['tp']+m['fp']+m['fn'];m['positive_f1']=2*m['tp']/den if den else None
    def exact(k,n):
        if not n:return None
        return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
    rr=[dict(y=r['idh'],p=r['idh_probability'],site='RHUH') for r in known]
    return dict(total=len(rows),labelled=len(known),unknown_labels=len(rows)-len(known),metrics=m,
        exact_binomial_95ci=dict(sensitivity=exact(m['tp'],m['positive']),specificity=exact(m['tn'],m['negative']),precision=exact(m['tp'],m['tp']+m['fp'])),
        conditional_class_stratified_bootstrap_95ci=bootstrap(rr,THRESHOLD,reps=reps),
        caution='Very few mutant patients: bootstrap is conditional on observed cases/class counts. Exact binomial intervals supplied for sensitivity/specificity/precision; neither removes selection or population uncertainty.')

def evaluate(records,reps=2000):
    return {name:score(group,reps) for name,group in {
        'primary_treatment_naive':[r for r in records if r['prior_treatment']=='no'],
        'secondary_all_preoperative':records,
        'previously_treated':[r for r in records if r['prior_treatment']=='yes'],
        'unknown_treatment_history':[r for r in records if r['prior_treatment']=='unknown']}.items()}

def run(manifest,output=OUT,device='cuda'):
    rows,protocol=freeze(manifest,output);output=Path(output);torch.set_num_threads(4)
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    arrays=[];reference_prep=None
    for j,e in enumerate(protocol['checkpoints']):
        dest=output/f'{e["member"]}_fold{e["fold"]}.npz'
        print(f'Checkpoint {j+1}/15: {e["member"]}, fold {e["fold"]}',flush=True)
        state=torch.load(e['path'],map_location='cpu',weights_only=True)
        if state.get('target')!='joint':raise ValueError('Expected joint checkpoint')
        prep={k:v for k,v in state['preprocessing'].items() if k!='cache_dir'}
        if reference_prep is None:reference_prep=prep
        if prep!=reference_prep:raise ValueError('Member preprocessing mismatch')
        for r in rows:
            if r['patient_id'] in set(state['train_ids'])|set(state['validation_ids']):raise ValueError('External patient occurs in development checkpoint')
        del state
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                if str(z['signature'])!=signature:raise ValueError('Prediction fingerprint mismatch')
                a=z['probabilities'].copy()
        else:
            model,state=load_checkpoint(e['path'],device)
            try:
                state['preprocessing']=dict(state['preprocessing'],cache_dir=str(output/'image_cache'))
                loader=DataLoader(dataset_for(state,rows),batch_size=1,shuffle=False,num_workers=0)
                batches=[]
                with torch.inference_mode():
                    for i,b in enumerate(loader):
                        logits=model(b['t1c'].to(device),b['flair'].to(device))
                        prob=logits.float().sigmoid()[:,0].cpu().numpy()
                        if not np.isfinite(prob).all():raise FloatingPointError('Nonfinite predictions')
                        batches.append(prob)
                        if i==0 or (i+1)%10==0:print(f'  Patient {i+1}/{len(rows)}',flush=True)
                a=np.concatenate(batches)
                with dest.with_suffix('.tmp').open('wb') as f:np.savez(f,probabilities=a,signature=signature)
                dest.with_suffix('.tmp').replace(dest)
            finally:
                # CPU cleanup only; avoid masking a CUDA failure with empty_cache errors.
                del model,state;gc.collect()
        if a.shape!=(len(rows),) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid prediction array')
        arrays.append(a)
    family=np.stack(arrays).reshape(3,5,len(rows)).mean(1);mean=family.mean(0)
    records=[dict(r,idh_probability=float(mean[i]),prediction=int(mean[i]>=THRESHOLD),
                  **{name+'_probability':float(family[j,i]) for j,name in enumerate(NAMES)}) for i,r in enumerate(rows)]
    save_csv(output/'predictions.csv',records)
    result=evaluate(records);atomic_json(output/'results.json',result)
    print(json.dumps(result,indent=2));return result
