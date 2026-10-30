"""GPU experiments for the RadGenome ISBI paper (27 Oct 2026).

E1  region-occlusion test on the development cohort (inference only, held-out checkpoints)
E2  pre-operative external specificity on OpenNeuro ds007045 (frozen 15-checkpoint ensemble, inference only)
E4  leave-UCSF-out retraining of the two missing ensemble members (CA44, MC42)
E5  leave-UTSW-out retraining (CA42; optionally all three members)

Nothing trains or touches the GPU on import. Every long step writes its results to disk and resumes from them.
Uses the project's own modules (research_experiments, ensemble_oof_xrai, xrai_review, rhuh_external conventions).
"""
from pathlib import Path
from dataclasses import replace
import csv, gc, hashlib, json, time, os, sys, urllib.request, shutil
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import torch

ROOT = Path(__file__).resolve().parent
BASE = Path.home() / 'Downloads' / 'outputs' / 'clinical_experiments'
MEMBERS = ['v2_reference_seed42', 'v2_reference_seed44', 'v2_mean_concat_seed42']   # CA42, CA44, MC42 (paper)
SHORT = {'v2_reference_seed42': 'CA42', 'v2_reference_seed44': 'CA44', 'v2_mean_concat_seed42': 'MC42'}
THRESHOLD = 0.14600733667612076                                                      # frozen ensemble threshold
FOLD_SEED = 42      # inner LOSO folds: seed 42 reproduces loso_UCSF/inner_cv exactly, so members' inner OOF align


# ============================================================================ shared helpers
def remap(path, path_remap):
    if not path or Path(path).exists():
        return path
    for old, new in (path_remap or {}).items():
        if str(path).lower().startswith(old.lower()):
            cand = new + str(path)[len(old):]
            if Path(cand).exists():
                return cand
    return path


def remap_row(row, path_remap):
    row = dict(row)
    for k in ('t1c_path', 'flair_path', 'brain_mask_path', 'tumor_mask_path'):
        row[k] = remap(row.get(k) or '', path_remap)
    return row


def free(device):
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()


# ============================================================================ E1 region occlusion
CONDITIONS = ['original', 'tumour_ring_only', 'lesion_only', 'brain_only', 't1c_blank', 'flair_blank',
              'tumour_ring_only_blurred_brain', 'brain_only_blurred_tumour']
CONDITION_TEXT = {
    'original': 'unchanged input',
    'tumour_ring_only': 'lesion + 10 mm ring kept; all other voxels set to 0 (the background value of the normalised input)',
    'lesion_only': 'lesion kept; everything else set to 0',
    'brain_only': 'lesion + 10 mm ring set to 0; rest of the brain kept',
    't1c_blank': 'T1c channel set to 0 everywhere; FLAIR unchanged',
    'flair_blank': 'FLAIR channel set to 0 everywhere; T1c unchanged',
    'tumour_ring_only_blurred_brain': 'lesion + ring kept; the rest of the brain replaced by its own Gaussian blur (sigma 4 voxels): removes fine brain texture, keeps gross intensity',
    'brain_only_blurred_tumour': 'rest of the brain kept; lesion + ring replaced by their Gaussian blur (sigma 4 voxels): removes tumour texture, keeps the mass effect',
}


def crop_label_map(ds, row):
    """Tumour labels on the network's input grid, as [D,H,W] = Z,Y,X (same convention as the saved XRAI maps)."""
    from xrai_review import input_grid
    shape, affine = input_grid(ds, row)
    if not row.get('tumor_mask_path'):
        return np.zeros(tuple(reversed(shape)), np.int16), affine
    lab = resample_from_to(nib.load(row['tumor_mask_path']), (tuple(shape), affine), order=0).get_fdata()
    return np.rint(lab).astype(np.int16).transpose(2, 1, 0), affine


def region_masks(labels, affine, ring_mm=10.0):
    lesion = labels > 0
    spacing = float(abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))
    if not lesion.any():
        return lesion, np.zeros_like(lesion)
    dist = ndimage.distance_transform_edt(~lesion, sampling=spacing)
    ring = (~lesion) & (dist <= ring_mm)
    return lesion, ring


def make_conditions(t, f, lesion, ring, blur_sigma=4.0):
    """t, f: [D,H,W] float32 numpy (normalised; 0 outside the brain). Returns {name: (t, f)}."""
    keep = lesion | ring
    zero = lambda a, m: np.where(m, a, 0).astype(np.float32)
    tb, fb = gaussian_filter(t, blur_sigma), gaussian_filter(f, blur_sigma)
    brain_t, brain_f = (t != 0), (f != 0)
    out = {
        'original': (t, f),
        'tumour_ring_only': (zero(t, keep), zero(f, keep)),
        'lesion_only': (zero(t, lesion), zero(f, lesion)),
        'brain_only': (zero(t, ~keep), zero(f, ~keep)),
        't1c_blank': (np.zeros_like(t), f),
        'flair_blank': (t, np.zeros_like(f)),
        'tumour_ring_only_blurred_brain': (np.where(keep, t, tb * brain_t).astype(np.float32),
                                           np.where(keep, f, fb * brain_f).astype(np.float32)),
        'brain_only_blurred_tumour': (np.where(keep, tb, t).astype(np.float32), np.where(keep, fb, f).astype(np.float32)),
    }
    assert list(out) == CONDITIONS
    return out


@torch.inference_mode()
def predict_batch(model, pairs, device, chunk=4):
    probs = []
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        t = torch.from_numpy(np.stack([p[0] for p in part]))[:, None].to(device)
        f = torch.from_numpy(np.stack([p[1] for p in part]))[:, None].to(device)
        logits = model(t, f).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('non-finite logits')
        probs.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
    return np.concatenate(probs)


def occlusion_masks(cases, out_dir, input_cache, path_remap=None, ring_mm=10.0):
    """Crop-space tumour labels for every case (CPU; ~1 s per case, cached). Uses member 0's preprocessing."""
    from research_experiments import load_checkpoint
    from ensemble_oof_xrai import case_dataset
    out = Path(out_dir) / 'crop_masks'; out.mkdir(parents=True, exist_ok=True)
    state = None; t0 = time.time(); todo = [c for c in cases if not (out / f"{key(c['patient_id'])}.npz").exists()]
    for i, c in enumerate(todo):
        if state is None:
            _, state = load_checkpoint(c['members'][0]['checkpoint'], 'cpu')
        row = remap_row(c['row'], path_remap)
        ds = case_dataset(state, row, input_cache)
        labels, affine = crop_label_map(ds, row)
        lesion, ring = region_masks(labels, affine, ring_mm)
        np.savez_compressed(out / f"{key(c['patient_id'])}.npz", labels=labels, affine=affine, lesion=lesion, ring=ring,
                            ring_mm=ring_mm)
        if (i + 1) % 100 == 0:
            print(f'  masks {i + 1}/{len(todo)} | {time.time() - t0:.0f}s', flush=True)
    return out


def key(pid):
    return pid.replace(':', '_').replace('/', '_').replace('\\', '_')


def run_occlusion(cases, out_dir, input_cache, device='cuda', path_remap=None, ring_mm=10.0, chunk=4, tol=0.01):
    """For each member and held-out checkpoint: probabilities of every case under every condition. Resumable."""
    from research_experiments import load_checkpoint
    from ensemble_oof_xrai import case_dataset, assert_held_out
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = occlusion_masks(cases, out_dir, input_cache, path_remap, ring_mm)
    for j in range(len(cases[0]['members'])):
        by_ckpt = {}
        for c in cases:
            by_ckpt.setdefault(c['members'][j]['checkpoint'], []).append(c)
        for ckpt, cs in sorted(by_ckpt.items()):
            fold = cs[0]['members'][j]['fold']; member = Path(ckpt).parent.parent.name
            part = out_dir / f'occlusion_{member}_fold{fold}.npz'
            if part.exists():
                continue
            t0 = time.time(); model, state = load_checkpoint(ckpt, device)
            try:
                ids, P, saved, nles = [], [], [], []
                for k, c in enumerate(cs):
                    assert_held_out(state, c['patient_id'])
                    row = remap_row(c['row'], path_remap)
                    b = case_dataset(state, row, input_cache)[0]
                    t, f = b['t1c'].numpy()[0], b['flair'].numpy()[0]
                    with np.load(mask_dir / f"{key(c['patient_id'])}.npz") as z:
                        lesion, ring = z['lesion'], z['ring']
                        if float(z['ring_mm']) != ring_mm:
                            raise ValueError('cached masks use a different ring width: delete crop_masks/')
                    if lesion.shape != t.shape:
                        raise ValueError(f"mask grid {lesion.shape} != input {t.shape} for {c['patient_id']}")
                    conds = make_conditions(t, f, lesion, ring)
                    P.append(predict_batch(model, [conds[n] for n in CONDITIONS], device, chunk))
                    ids.append(c['patient_id']); saved.append(c['members'][j]['probability']); nles.append(int(lesion.sum()))
                    if (k + 1) % 50 == 0:
                        print(f'    {member} fold {fold}: {k + 1}/{len(cs)} | {time.time() - t0:.0f}s', flush=True)
                P = np.stack(P); saved = np.array(saved)
                dev = np.abs(P[:, 0] - saved)
                tmp = part.with_suffix('.tmp.npz')
                np.savez(tmp, patient_id=np.array(ids), conditions=np.array(CONDITIONS), prob=P, saved_oof=saved,
                         lesion_voxels=np.array(nles), fold=fold, member=member)
                os.replace(tmp, part)
                print(f'  {member} fold {fold}: {len(cs)} cases in {time.time() - t0:.0f}s | original vs saved OOF: '
                      f'max |diff| {dev.max():.4f}, {(dev > tol).sum()} above {tol}', flush=True)
            finally:
                del model, state; free(device)
    return out_dir


def load_occlusion(out_dir):
    """Long table -> wide: one row per patient, columns <member>|<condition>, plus ensemble means."""
    parts = sorted(Path(out_dir).glob('occlusion_*_fold*.npz'))
    rows = []
    for p in parts:
        with np.load(p, allow_pickle=False) as z:
            conds = [str(c) for c in z['conditions']]
            for i, pid in enumerate(z['patient_id']):
                rows.append(dict(patient_id=str(pid), member=str(z['member']), lesion_voxels=int(z['lesion_voxels'][i]),
                                 saved_oof=float(z['saved_oof'][i]), **{c: float(z['prob'][i, k]) for k, c in enumerate(conds)}))
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    members = sorted(long.member.unique())
    ens = long.groupby('patient_id')[CONDITIONS + ['saved_oof']].mean()
    ens['n_members'] = long.groupby('patient_id').member.nunique()
    ens['lesion_voxels'] = long.groupby('patient_id').lesion_voxels.first()
    return long, ens.reset_index()


# ============================================================================ E2 ds007045
DS_ID = 'ds007045'
DS_S3 = 'https://s3.amazonaws.com/openneuro.org/ds007045/'
N4 = 'derivatives/processing_MNI152_skull-stripped_N4/processing_MNI152_skull-stripped_N4'
SEG = 'derivatives/segmentation'


def ds_paths(root, sub):
    root = Path(root)
    return dict(t1c_path=root / N4 / sub / f'{sub}_t1ce_MNI152_brain_n4.nii.gz',
                flair_path=root / N4 / sub / f'{sub}_flair_MNI152_brain_n4.nii.gz',
                t2_path=root / N4 / sub / f'{sub}_t2_MNI152_brain_n4.nii.gz',
                tumor_mask_path=root / SEG / sub / f'{sub}_MNI152_tumor-mask.nii.gz')


def ds_download(root, files_csv, workers=8, verify_sha=True):
    """Download only the files E2 needs (~7 GB: N4 T1c, FLAIR, T2 and the masks) from OpenNeuro's public S3 bucket.
    files_csv lists relative paths with the size and SHA-256 recorded in the dataset's git-annex keys (v2.0.1)."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    files = pd.read_csv(files_csv)
    def one(r):
        dest = root / r.path
        if dest.exists() and (r.size < 0 or dest.stat().st_size == r.size):
            return 'present'
        dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_name(dest.name + '.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(DS_S3 + r.path, timeout=120) as resp, tmp.open('wb') as fh:
                    shutil.copyfileobj(resp, fh, 1 << 20)
                if r.size >= 0 and tmp.stat().st_size != r.size:
                    raise IOError(f'size {tmp.stat().st_size} != {r.size}')
                if verify_sha and isinstance(r.sha256, str) and len(r.sha256) == 64:
                    h = hashlib.sha256()
                    with tmp.open('rb') as fh:
                        for blk in iter(lambda: fh.read(1 << 22), b''): h.update(blk)
                    if h.hexdigest() != r.sha256:
                        raise IOError('sha256 mismatch')
                os.replace(tmp, dest); return 'downloaded'
            except Exception as e:
                err = e; time.sleep(2 * (attempt + 1))
        return f'FAILED: {type(err).__name__}: {err}'
    t0 = time.time(); status = []
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(one, files.itertuples())):
            status.append(s)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(files)} files | {time.time() - t0:.0f}s', flush=True)
    files['status'] = status
    print(files.status.str.split(':').str[0].value_counts().to_dict())
    return files


def ds_rows(root, require_all=True):
    """Rows in the RadGenome dataset format. IDH is set to 0 (wildtype) for everyone: see the notebook's caveat."""
    root = Path(root)
    part = pd.read_csv(root / 'participants.tsv', sep='\t', dtype=str)
    rows, missing = [], []
    for r in part.itertuples():
        sub = r.participant_id; p = ds_paths(root, sub)
        miss = [k for k, v in p.items() if not v.exists()]
        if miss:
            missing.append(dict(subject=sub, missing=' '.join(miss))); continue
        fs = str(r.field_strength).replace(',', '.')
        rows.append(dict(patient_id=f'DS7045:{sub}', subject_id=sub, site='DS7045', centre=''.join(ch for ch in sub[4:] if ch.isalpha()),
                         t1c_path=str(p['t1c_path']), flair_path=str(p['flair_path']), t2_path=str(p['t2_path']),
                         tumor_mask_path=str(p['tumor_mask_path']), brain_mask_path='',
                         idh=0, mgmt=None, idh_observed=True, mgmt_observed=False,
                         age=pd.to_numeric(r.age, errors='coerce'), sex=r.sex, mgmt_methylated=r.MGMT,
                         field_strength=pd.to_numeric(fs, errors='coerce'), scanner_model=r.scanner_model))
    if missing and require_all:
        print(f'{len(missing)} subjects have missing files, e.g. {missing[:3]}')
    return rows, pd.DataFrame(missing)


def ds_audit(rows, out_csv=None):
    """Geometry and label audit: T1c, FLAIR, T2 and mask on one grid; mask labels. CPU, ~1 s per subject."""
    out = []
    for r in rows:
        rec = dict(patient_id=r['patient_id'])
        try:
            ims = {k: nib.load(r[k]) for k in ('t1c_path', 'flair_path', 't2_path', 'tumor_mask_path')}
            ref = ims['t1c_path']
            rec['shape'] = 'x'.join(map(str, ref.shape)); rec['zooms'] = 'x'.join(f'{z:g}' for z in ref.header.get_zooms()[:3])
            rec['same_grid'] = all(im.shape == ref.shape and np.allclose(im.affine, ref.affine, atol=1e-3) for im in ims.values())
            lab = np.rint(np.asarray(ims['tumor_mask_path'].dataobj)).astype(int)
            rec['labels'] = ' '.join(map(str, sorted(set(np.unique(lab)) - {0})))
            rec['lesion_ml'] = float((lab > 0).sum() * abs(np.linalg.det(ref.affine[:3, :3])) / 1000)
            a = np.asarray(ims['t1c_path'].dataobj, dtype=np.float32)
            rec['t1c_nonzero_frac'] = float((a != 0).mean()); rec['t1c_min'] = float(a.min())
            rec['error'] = ''
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        out.append(rec)
    df = pd.DataFrame(out)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def frozen_inference(rows, out_dir, device='cuda', members=MEMBERS, base=BASE, dev_ids_check=True):
    """Mean sigmoid over 5 folds per member, then equal mean over members (15 checkpoints): identical to the RHUH
    and reserved-test protocol. One npz per checkpoint; resumable. Returns a DataFrame of probabilities."""
    from research_experiments import load_checkpoint, dataset_for
    from torch.utils.data import DataLoader
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha256(json.dumps(sorted((r['patient_id'], r['t1c_path'], r['flair_path'], r['tumor_mask_path']) for r in rows)).encode()).hexdigest()
    arrays = {}
    for m in members:
        for fold in range(1, 6):
            ck = Path(base) / m / f'fold_{fold}' / 'best.pt'; dest = out_dir / f'{m}_fold{fold}.npz'
            if dest.exists():
                with np.load(dest) as z:
                    if str(z['signature']) != sig:
                        raise ValueError(f'{dest.name}: cached predictions are for a different row set; use a new folder')
                    arrays[(m, fold)] = z['p'].copy(); continue
            t0 = time.time(); model, state = load_checkpoint(ck, device)
            try:
                if dev_ids_check and ({r['patient_id'] for r in rows} & (set(state['train_ids']) | set(state['validation_ids']))):
                    raise ValueError('external patient inside a development checkpoint')
                st = dict(state); st['preprocessing'] = dict(state['preprocessing'], cache_dir=str(out_dir / 'image_cache'))
                ps = []
                with torch.inference_mode():
                    for i, b in enumerate(DataLoader(dataset_for(st, rows), batch_size=1, shuffle=False, num_workers=0)):
                        ps.append(model(b['t1c'].to(device), b['flair'].to(device)).float().sigmoid()[:, 0].cpu().numpy())
                        if fold == 1 and m == members[0] and (i == 0 or (i + 1) % 50 == 0):
                            print(f'    {i + 1}/{len(rows)} (first pass also builds the image cache)', flush=True)
                a = np.concatenate(ps)
                if a.shape != (len(rows),) or not np.isfinite(a).all():
                    raise ValueError('invalid predictions')
                tmp = dest.with_suffix('.tmp.npz'); np.savez(tmp, p=a, signature=sig); os.replace(tmp, dest)
                arrays[(m, fold)] = a
                print(f'  {SHORT.get(m, m)} fold {fold}: {len(a)} patients in {time.time() - t0:.0f}s', flush=True)
            finally:
                del model, state; free(device)
    fam = {m: np.mean([arrays[(m, k)] for k in range(1, 6)], 0) for m in members}
    df = pd.DataFrame(dict(patient_id=[r['patient_id'] for r in rows], probability=np.mean([fam[m] for m in members], 0),
                           **{f'{m}_probability': fam[m] for m in members}))
    return df


def reproduce_oof(member, device, n=4, base=BASE, path_remap=None, cache_dir=None):
    """Check that inference on this machine reproduces saved OOF probabilities for a few fold-1 validation patients."""
    from research_experiments import load_checkpoint, dataset_for
    from clinical_quality import read_manifest
    from torch.utils.data import DataLoader
    model, state = load_checkpoint(Path(base) / member / 'fold_1' / 'best.pt', device)
    try:
        man = {r['patient_id']: r for r in read_manifest(Path(base) / member)}
        oof = pd.read_csv(Path(base) / member / 'oof_predictions.csv').set_index('patient_id')
        pids = sorted(state['validation_ids'])[:n]
        rows = [remap_row(man[p], path_remap) for p in pids]
        st = dict(state)
        if cache_dir:
            st['preprocessing'] = dict(state['preprocessing'], cache_dir=str(cache_dir))
        with torch.inference_mode():
            p = np.concatenate([model(b['t1c'].to(device), b['flair'].to(device)).float().sigmoid()[:, 0].cpu().numpy()
                                for b in DataLoader(dataset_for(st, rows), batch_size=1)])
        return float(np.abs(p - oof.loc[pids, 'idh_probability'].values).max())
    finally:
        del model, state; free(device)


# ============================================================================ E4 / E5 leave-one-site-out retraining
def loso_member(member, site, output, device='cuda', fold_seed=FOLD_SEED, base=BASE, reference_folds=None, path_remap=None):
    """Retrain one ensemble member without `site`, exactly as research_experiments.loso, except that the inner folds
    use fold_seed (42) for every member so that the members' inner out-of-fold predictions are aligned. The member's
    own seed still sets initialisation and shuffling. Resumes an interrupted run from its last epoch."""
    from research_experiments import config_from_run, train_experiment, ExperimentModel, load_checkpoint, dataset_for
    from experiments_v3 import v3_folds
    from experiments_v2 import TumorCropDataset
    from cv_training import evaluate_model, _write_rows
    from clinical_quality import read_manifest, save_json, metrics
    from torch.utils.data import DataLoader
    source = Path(base) / member; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = [remap_row(r, path_remap) for r in read_manifest(source) if r['role'] == 'development']
    bad = check_paths(rows)
    if bad:
        raise FileNotFoundError(f'{len(bad)} image paths do not exist (add the moved folder to PATH_REMAP), e.g. {bad[:3]}')
    train = [r for r in rows if r['site'] != site]; held = [r for r in rows if r['site'] == site]
    cfg, model_cfg, prep = config_from_run(source)
    member_seed = cfg.seed
    inner = v3_folds(train, replace(cfg, seed=fold_seed))
    if reference_folds is not None:
        ref = {r['patient_id']: r['cv_fold'] for r in read_manifest(reference_folds)}
        got = {r['patient_id']: r['cv_fold'] for r in inner}
        if got != ref:
            raise ValueError(f'inner folds differ from {reference_folds}')
    print(f'{SHORT.get(member, member)} without {site}: train {len(train)} | held-out {len(held)} | member seed {member_seed} | '
          f'fusion {model_cfg.get("fusion_mode", "cross_attention")}', flush=True)
    t0 = time.time()
    train_experiment(inner, cfg, model_cfg, prep, output / 'inner_cv', device=device,
                     model_factory=ExperimentModel, dataset_factory=TumorCropDataset)
    print(f'  inner CV complete in {(time.time() - t0) / 3600:.1f} h', flush=True)
    dest = output / 'outer_site_predictions.csv'
    if dest.exists():
        return pd.read_csv(dest)
    preds = []
    for fold in range(1, 6):
        net, state = load_checkpoint(output / 'inner_cv' / f'fold_{fold}' / 'best.pt', device)
        if set(state['train_ids']) & {r['patient_id'] for r in held}:
            raise ValueError('outer-site leakage')
        loader = DataLoader(dataset_for(state, held), batch_size=cfg.batch_size, num_workers=0)
        _, pred = evaluate_model(net, loader, torch.device(device), cfg.amp and str(device).startswith('cuda'))
        preds.append(pred); del net, loader; free(device)
    comb = preds[0]
    for i, r in enumerate(comb):
        if any(p[i]['patient_id'] != r['patient_id'] for p in preds):
            raise ValueError('prediction ordering mismatch')
        for task in ('idh', 'mgmt'):
            r[task + '_probability'] = float(np.mean([p[i][task + '_probability'] for p in preds])); r.pop(task + '_logit', None)
    _write_rows(dest, comb)
    known = [r for r in comb if r['idh_label'] is not None]
    save_json(output / 'outer_site_metrics.json', dict(site=site, member=member, fold_seed=fold_seed, member_seed=member_seed,
              idh=metrics([r['idh_label'] for r in known], [r['idh_probability'] for r in known])))
    return pd.read_csv(dest)


def check_paths(rows, keys=('t1c_path', 'flair_path', 'tumor_mask_path')):
    bad = [(r['patient_id'], k, r[k]) for r in rows for k in keys if r.get(k) and not Path(r[k]).exists()]
    return bad


def loso_combine(site, member_dirs, Z, fit, n_boot=2000, seed=20261021):
    """Full-pipeline leave-site-out: ensemble = mean of the retrained members' probabilities (outer site: mean of each
    member's 5 inner checkpoints; training sites: the members' aligned inner out-of-fold predictions). Every combiner
    is fitted on the training sites only, using those inner OOF logits, and scored on the held-out site."""
    from sklearn.metrics import roc_auc_score
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    held, inner = [], []
    for d_ in member_dirs:
        d_ = Path(d_)
        held.append(pd.read_csv(d_ / 'outer_site_predictions.csv')[['patient_id', 'idh_probability']].set_index('patient_id').idh_probability)
        inner.append(pd.read_csv(d_ / 'inner_cv' / 'oof_predictions.csv')[['patient_id', 'idh_probability', 'fold']].set_index('patient_id'))
    if len(inner) > 1 and any(not (i.fold == inner[0].fold.reindex(i.index)).all() for i in inner[1:]):
        raise ValueError('inner folds of the members are not aligned')
    p_held = pd.concat(held, axis=1).mean(1); p_inner = pd.concat([i.idh_probability for i in inner], axis=1).mean(1)
    D = Z.copy()
    D['net'] = np.where(D.site == site, lg(D.patient_id.map(p_held)), lg(D.patient_id.map(p_inner)))
    D['net_prob'] = np.where(D.site == site, D.patient_id.map(p_held), D.patient_id.map(p_inner))
    if D.net.isna().any():
        raise ValueError(f'{int(D.net.isna().sum())} analysis patients lack a LOSO prediction')
    tr, te = D[D.site != site], D[D.site == site]; y = te.y.values
    FE = [c for c in ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'no_enhancement', 'necrosis_frac',
                      'edema_ratio_log', 'mismatch_index', 'mismatch_frac']] + ['age']
    specs = {'Ensemble (recalibrated)': ['net'], 'Features + age': FE, 'Ensemble + age': ['net', 'age'],
             'Ensemble + features': ['net'] + FE[:-1], 'Hybrid': ['net'] + FE}
    P = {'Ensemble (raw, retrained)': te.net_prob.values, 'Ensemble (pooled CV, site seen in training)': te.probability.values}
    for m, cols in specs.items():
        P[m] = fit(tr[cols].values, tr.y.values).predict_proba(te[cols].values)[:, 1]
    rng = np.random.default_rng(seed); g = [np.where(y == v)[0] for v in (0, 1)]
    B = [np.concatenate([rng.choice(x, len(x)) for x in g]) for _ in range(n_boot)]
    rows = []
    for m, p in P.items():
        v = [roc_auc_score(y[i], p[i]) for i in B]
        rows.append(dict(model=m, n=len(y), mutant=int(y.sum()), auroc=roc_auc_score(y, p),
                         ci=f'{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}'))
    diffs = []
    for a, b in [('Hybrid', 'Features + age'), ('Hybrid', 'Ensemble + age'), ('Hybrid', 'Ensemble (recalibrated)'),
                 ('Features + age', 'Ensemble (recalibrated)'), ('Ensemble + features', 'Features + age')]:
        v = [roc_auc_score(y[i], P[a][i]) - roc_auc_score(y[i], P[b][i]) for i in B]
        diffs.append(dict(comparison=f'{a} minus {b}', delta=roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]),
                          ci=f'{np.percentile(v, 2.5):+.3f} to {np.percentile(v, 97.5):+.3f}'))
    inner_auc = roc_auc_score(tr.y.values, tr.net_prob.values)
    return pd.DataFrame(rows), pd.DataFrame(diffs), dict(members=len(member_dirs), inner_oof_auroc_training_sites=inner_auc)


# ============================================================================ analysis helpers (CPU)
def strat_boot(frame, reps, seed):
    s = (frame.site + '_' + frame.y.astype(str)).values
    groups = [np.where(s == g)[0] for g in np.unique(s)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(reps)]


def age_matched_sets(Z, seed=20261021, caliper=2.0, k=3):
    """The paper's pre-declared matching (same site, |age difference| <= 2 y, up to 3 wildtype per mutant)."""
    y, AGE, sites = Z.y.values, Z.age.values, Z.site.values
    rng = np.random.default_rng(seed); used = set(); sets = []
    mut = np.where(y == 1)[0].copy(); rng.shuffle(mut)
    for m in mut:
        cand = np.where((y == 0) & (sites == sites[m]) & (np.abs(AGE - AGE[m]) <= caliper))[0]
        cand = sorted((c for c in cand if c not in used), key=lambda c: (abs(AGE[c] - AGE[m]), c))[:k]
        if cand:
            used.update(cand); sets.append(np.array([m] + list(cand)))
    return sets


def analyse_occlusion(ens, Z, run_cv, MODELS, n_boot=2000, seed=20261021):
    """E1 pre-declared outputs for every condition: AUROC (all patients), AUROC in the age-matched sample, within-class
    Spearman rho with age; plus paired AUROC differences vs the original input and refitted image-only/hybrid models."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    sets = age_matched_sets(Z, seed)                              # matched on the full 1,393, exactly as published
    D = Z.merge(ens, on='patient_id', how='left', validate='one_to_one')
    ok = (D.n_members == 3) & (D.lesion_voxels > 0)
    info = dict(analysis_patients=int(ok.sum()), excluded_no_prediction=int(D.n_members.isna().sum()),
                excluded_no_lesion=int(((D.lesion_voxels == 0) & D.n_members.notna()).sum()),
                max_abs_original_minus_saved_oof=float((D.original - D.probability).abs().max()))
    keep = np.where(ok.values)[0]
    sets = [s[np.isin(s, keep)] for s in sets]; sets = [s for s in sets if len(s) >= 2 and D.y.values[s[0]] == 1]
    MATCHED = np.concatenate(sets)
    D = D.reset_index(drop=True); y = D.y.values; age = D.age.values
    B = strat_boot(D.iloc[keep], n_boot, seed); B = [keep[i] for i in B]
    rngm = np.random.default_rng(seed); BM = [np.concatenate([sets[j] for j in rngm.integers(0, len(sets), len(sets))]) for _ in range(1000)]
    rows, diffs = [], []
    for c in CONDITIONS:
        p = D[c].values
        a_all = roc_auc_score(y[keep], p[keep]); v = [roc_auc_score(y[i], p[i]) for i in B]
        a_m = roc_auc_score(y[MATCHED], p[MATCHED]); vm = [roc_auc_score(y[i], p[i]) for i in BM]
        rec = dict(condition=c, description=CONDITION_TEXT[c], n=len(keep), mutant=int(y[keep].sum()),
                   auroc=a_all, auroc_ci=f'{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}',
                   auroc_age_matched=a_m, auroc_age_matched_ci=f'{np.percentile(vm, 2.5):.3f}-{np.percentile(vm, 97.5):.3f}',
                   mean_probability_mutant=float(p[keep][y[keep] == 1].mean()), mean_probability_wildtype=float(p[keep][y[keep] == 0].mean()))
        for lab, cls in (('wildtype', 0), ('mutant', 1)):
            k = keep[y[keep] == cls]; r_ = spearmanr(lg(p[k]), age[k])[0]
            rb = [spearmanr(lg(p[i][y[i] == cls]), age[i][y[i] == cls])[0] for i in B[:1000]]
            rec[f'rho_age_{lab}'] = r_; rec[f'rho_age_{lab}_ci'] = f'{np.percentile(rb, 2.5):+.2f} to {np.percentile(rb, 97.5):+.2f}'
        rows.append(rec)
        if c != 'original':
            po = D['original'].values
            dv = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], po[i]) for i in B]
            dm = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], po[i]) for i in BM]
            diffs.append(dict(condition=c, delta_auroc=a_all - roc_auc_score(y[keep], po[keep]),
                              ci=f'{np.percentile(dv, 2.5):+.3f} to {np.percentile(dv, 97.5):+.3f}',
                              delta_auroc_age_matched=a_m - roc_auc_score(y[MATCHED], po[MATCHED]),
                              ci_age_matched=f'{np.percentile(dm, 2.5):+.3f} to {np.percentile(dm, 97.5):+.3f}'))
    # refitted combiners with the occluded ensemble logit (same 5x5 CV as the paper; analysis patients only)
    refit = []
    Zk = Z[Z.patient_id.isin(D.patient_id.values[keep])].reset_index(drop=True)
    pmap = D.set_index('patient_id')
    for c in ('original', 'tumour_ring_only', 'brain_only', 't1c_blank', 'flair_blank'):
        Zc = Zk.copy(); Zc['dl_logit'] = lg(Zc.patient_id.map(pmap[c]).values); Zc['probability'] = Zc.patient_id.map(pmap[c]).values
        cvc = run_cv(Zc, {m: MODELS[m] for m in ('Ensemble + features', 'Ensemble + age', 'Hybrid')}, include_raw=False)
        refit.append(dict(condition=c, **{m: roc_auc_score(Zc.y.values, cvc[m]['p']) for m in cvc}))
    return pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(refit), dict(info, matched_sets=len(sets), matched_patients=len(MATCHED))


def exact_ci(k, n):
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def specificity_table(frame, preds, thresholds, group=None):
    """All patients are IDH-wildtype: specificity = share below the frozen threshold, with exact binomial CIs."""
    rows = []
    groups = [('all', np.ones(len(frame), bool))] if group is None else \
        [(f'{group}: {g}', (frame[group] == g).values) for g in sorted(frame[group].dropna().unique(), key=str)]
    for gname, m in groups:
        for model, p in preds.items():
            p = np.asarray(p)[m]; ok = np.isfinite(p); p = p[ok]
            if not len(p):
                continue
            for tag, th in thresholds[model].items():
                fp = int((p >= th).sum()); n = len(p); lo, hi = exact_ci(n - fp, n)
                rows.append(dict(group=gname, model=model, operating_point=tag, threshold=th, n=n, false_positives=fp,
                                 specificity=(n - fp) / n, spec_ci=f'{lo:.3f}-{hi:.3f}', mean_predicted_p_mutant=float(p.mean()),
                                 median_predicted_p_mutant=float(np.median(p))))
    return pd.DataFrame(rows)
