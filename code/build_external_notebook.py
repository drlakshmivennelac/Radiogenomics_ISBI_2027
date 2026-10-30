"""Build IDH_External_Validation.ipynb: apply the frozen hybrid models (from IDH_Hybrid_Analysis.ipynb) unchanged to
RHUH-GBM and MU-Glioma-Post.  Reuses the feature extraction from build_idh_correlates_notebook.py so the features are
computed identically.  Run: python build_external_notebook.py [output.ipynb]
"""
import sys, re
from pathlib import Path
import nbformat as nbf

src = Path(__file__).with_name('build_idh_correlates_notebook.py').read_text()
i = src.index('def canonical(path):'); j = src.index("FEATURE_VERSION = ")
EXTRACT_CODE = src[i:j].rstrip() + '\n'
_old_labels = '''        unexpected = set(labels) - {1, 2, 3, 4}'''
_new_labels = '''        override = LABEL_OVERRIDE.get(row.site)
        if override:
            lab[np.isin(lab, override.get('exclude', ()))] = 0          # e.g. resection cavity: not tumour
            labels = sorted(int(v) for v in np.unique(lab) if v > 0)
            if not labels: raise ValueError('empty tumour mask after excluding cavity')
        unexpected = set(labels) - {1, 2, 3, 4}'''
assert EXTRACT_CODE.count(_old_labels) == 1
EXTRACT_CODE = EXTRACT_CODE.replace(_old_labels, _new_labels)
_old_conv = '''        if labels == [3]: raise ValueError('only label 3: enhancing (BraTS 2023) or non-enhancing (BraTS 2017) is ambiguous')'''
_new_conv = '''        if override:
            pass
        elif labels == [3]: raise ValueError('only label 3: enhancing (BraTS 2023) or non-enhancing (BraTS 2017) is ambiguous')'''
assert EXTRACT_CODE.count(_old_conv) == 1
EXTRACT_CODE = EXTRACT_CODE.replace(_old_conv, _new_conv)
_old_reg = '''        if 4 in labels:
            regions'''
_new_reg = '''        if override:
            regions = {k: v for k, v in override.items() if k != 'exclude'}; rec['label_convention'] = 'override'
        elif 4 in labels:
            regions'''
assert EXTRACT_CODE.count(_old_reg) == 1
EXTRACT_CODE = EXTRACT_CODE.replace(_old_reg, _new_reg)
_old_brain = '''            t1c, _ = on_grid(row.t1c_path, mimg, order=1); brain = t1c > 0'''
_new_brain = '''            t1c, _ = on_grid(row.t1c_path, mimg, order=1)
            # robust head/brain support: ignore near-zero background left by resampling or registration
            brain = ndimage.binary_fill_holes(t1c > 0.05 * np.percentile(t1c[t1c > 0], 99))'''
assert EXTRACT_CODE.count(_old_brain) == 1
EXTRACT_CODE = EXTRACT_CODE.replace(_old_brain, _new_brain)
_old_grid = """        img = img.slicer[..., 0]
    same ="""
_new_grid = """        img = img.slicer[..., 0]
    if order > 0:
        img = undo_zscore(img)
    same ="""
assert EXTRACT_CODE.count(_old_grid) == 1
EXTRACT_CODE = EXTRACT_CODE.replace(_old_grid, _new_grid)
EXTRACT_CODE = """def background_offset(a):
    # Some public releases (e.g. RHUH-GBM) ship z-scored images: brain about mean 0 / SD 1, background a constant negative
    # value (the z-score of raw 0).  Ratios to normal brain are meaningless on such data.  Subtracting that constant
    # gives raw / SD, so ratio features become identical to those on the raw image.  Returns 0 for ordinary images.
    if (a < 0).mean() < 0.01: return 0.0
    vals, counts = np.unique(a[::4, ::4, ::4], return_counts=True)
    k = counts.argmax()
    if vals[k] < 0 and counts[k] > 0.2 * counts.sum(): return float(vals[k])
    return 0.0

def undo_zscore(img):
    a = np.asarray(img.dataobj, dtype=np.float32)
    off = background_offset(a)
    if off == 0.0: return img
    return nib.Nifti1Image(np.clip(a - off, 0, None), img.affine, img.header)

""" + EXTRACT_CODE

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# External validation of the frozen IDH hybrid models

Applies the models frozen in `outputs/idh_hybrid/frozen_models.json` **unchanged** to two external cohorts:

| Cohort | Setting | Primary analysis |
|---|---|---|
| RHUH-GBM | pre-operative glioblastoma, 40 patients | treatment-naive patients (previously treated excluded) |
| MU-Glioma-Post | post-treatment follow-up scans, 203 patients | earliest labelled scan per patient; all scans secondary |

**Same features as development.** Imaging features come from the same extraction code as `IDH_Imaging_Correlates.ipynb`,
are z-scored within each external cohort without labels (the same protocol as each development site), and use only
inputs available everywhere: conventional lesion features, T2-FLAIR mismatch and age. Diffusion and perfusion models
are not externally testable.

**Ensemble probabilities** are the selected three-member ensemble's external predictions (equal-weight average of 5 fold
checkpoints x 3 members): read from the RHUH run, and recomputed for MU from the saved per-checkpoint arrays.

**Interpretation limits.** RHUH has only a handful of IDH-mutant patients, so its sensitivity is essentially
uninformative (exact binomial CIs are shown). MU is post-treatment, so it tests a different question from a
pre-operative model: report it as such. Thresholds are the frozen development thresholds; nothing is tuned on these labels.
''')

md('## 1. Configuration')
code(r'''
from pathlib import Path
PROJECT = Path.cwd()
FROZEN = PROJECT / 'outputs' / 'idh_hybrid' / 'frozen_models.json'
OUTPUT_DIR = PROJECT / 'outputs' / 'idh_hybrid_external'

COHORTS = {
    'RHUH': dict(
        manifest=PROJECT / 'outputs' / 'rhuh_external_three' / 'reviewed_manifest.csv',      # t1c, flair, mask, idh, prior_treatment
        predictions=PROJECT / 'outputs' / 'rhuh_external_three' / 'predictions.csv',         # three-member idh_probability
        clinical=None,          # None: use the manifest's 'clinical_source' column; or a Path to the clinical CSV
        include=lambda r: str(r.get('prior_treatment', '')).strip().lower() in ('no', '0', 'false'),
    ),
    'MU': dict(
        predictions=PROJECT / 'outputs' / 'mu_external_four_equal' / 'predictions.csv',      # one row per scan: paths, labels
        arrays=PROJECT / 'outputs' / 'mu_external_four_equal',                                 # per-checkpoint npy arrays
        clinical=Path(r'C:\Users\claks\Downloads\MU-Glioma-Post_ClinicalData-July2025.xlsx'),
    ),
}
THREE = ['v2_reference_seed42', 'v2_reference_seed44', 'v2_mean_concat_seed42']
PATH_REMAP = {r'C:\Users\claks\Downloads\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5': r'C:\Users\claks\Downloads\UCSF-PDGM'}
N_JOBS, N_BOOT, SEED = 6, 2000, 20261021
MIN_REGION_ML, REFERENCE_MARGIN_MM = 0.2, 10
# MU-Glioma-Post contributed to BraTS 2024 post-treatment, whose masks use 1 = non-enhancing tumour core,
# 2 = surrounding FLAIR hyperintensity, 3 = enhancing tissue, 4 = resection cavity. Check one case visually.
LABEL_OVERRIDE = {'MU': dict(necrosis=(1,), edema=(2,), enhancing=(3,), exclude=(4,))}
FEATURE_VERSION = 'ext-v3'      # bump to force re-extraction
PORTABLE_MODELS = ['Age', 'Features', 'Features + age', 'Ensemble (raw)', 'Ensemble (recalibrated)',
                   'Ensemble + age', 'Ensemble + features', 'Hybrid']
''')

md('## 2. Setup')
code(r'''
import json, re, os, time
from types import SimpleNamespace
import numpy as np, pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
frozen = json.loads(Path(FROZEN).read_text())
print('Frozen models from', frozen['created'], ':', list(frozen['models']))

def norm_name(c): return ' ' + re.sub(r'[^a-z0-9]+', ' ', c.lower()) + ' '
def binary(v):
    v = str(v).strip().lower()
    return 1 if v in ('1', '1.0', 'true', 'mut', 'mutant') else 0 if v in ('0', '0.0', 'false', 'wt', 'wildtype') else np.nan

def read_table(path):
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        try:
            return pd.read_excel(path, dtype=str).fillna('')
        except ImportError:
            raise ImportError('Reading the MU clinical workbook needs openpyxl: run  %pip install openpyxl  and re-run this cell')
    return pd.read_csv(path, sep='\t' if path.suffix.lower() in ('.tsv', '.txt') else ',', dtype=str,
                       keep_default_na=False, encoding_errors='replace')

def age_lookup(clinical_path, subject_ids):
    """Age by subject from a clinical CSV/TSV/XLSX: tries every sheet, exact IDs, then digit-only IDs (e.g. 'PatientID_0003' ~ '3')."""
    if clinical_path is None or not Path(clinical_path).exists():
        print('  clinical file not found:', clinical_path); return {}
    path = Path(clinical_path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        try:
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        except ImportError:
            raise ImportError('Reading the MU clinical workbook needs openpyxl: run  %pip install openpyxl  and re-run this cell')
        tables = {k: v.fillna('') for k, v in sheets.items()}
    else:
        tables = {path.name: read_table(path)}
    ids = [str(x).strip() for x in subject_ids]
    digits = lambda v: str(int(re.sub(r'\D', '', v))) if re.sub(r'\D', '', str(v)) else None
    def search(mode):
        best = None
        key = (lambda v: str(v).strip()) if mode == 'exact' else digits
        want = {key(x): x for x in ids if key(x)}
        for sheet, t in tables.items():
            t.columns = [str(c) for c in t.columns]
            for c in t.columns:
                col = t[c].map(key)
                if col.nunique() < 0.9 * col.notna().sum():      # an ID column has (nearly) unique values
                    continue
                hits = int(col.isin(want).sum())
                if best is None or hits > best[0]:
                    best = (hits, sheet, c, mode, t, key, want)
        return best
    best = search('exact')
    if best is None or best[0] < 0.5 * len(ids):                  # digits only as a fallback
        alt = search('digits')
        if alt is not None and (best is None or alt[0] > best[0]): best = alt
    if best is None:
        print('  no ID column found in', path.name); return {}
    hits, sheet, idcol, mode, t, key, want = best
    acol = next((c for c in t.columns if ' age ' in norm_name(c) and pd.to_numeric(t[c], errors='coerce').notna().mean() > .5), None)
    print(f'  {path.name} [{sheet}]: id column {idcol!r} ({mode} match) matched {hits}/{len(ids)}; age column {acol!r}')
    if acol is None:
        print('    no age column found; columns are:', list(t.columns)[:40]); return {}
    out = {}
    for v, a in zip(t[idcol], pd.to_numeric(t[acol], errors='coerce')):
        k = key(v)
        if k in want: out[want[k]] = a
    return out

def resolve(p):
    if not p: return ''
    q = Path(p)
    if not q.exists():
        for old, new in PATH_REMAP.items():
            if str(q).lower().startswith(old.lower()):
                c = Path(new + str(q)[len(old):])
                if c.exists(): return str(c)
    return str(q)

def find_t2(t1c_path, flair_path):
    # T2 beside the T1c/FLAIR: RHUH '<id>_t2.nii.gz', MU '..._brain_t2w.nii.gz' (never the T2-FLAIR 't2f')
    folder = Path(t1c_path).parent
    hits = sorted([f for f in folder.glob('*.nii*') if re.search(r'(^|_)t2w?(_bias)?\.nii', f.name.lower())], key=lambda f: len(f.name))
    return str(hits[0]) if hits else ''
''')

md('## 3. Cohorts and ensemble probabilities')
code(r'''
frames = {}
# ---- RHUH ----
cfg = COHORTS['RHUH']
if Path(cfg['manifest']).exists():
    m = pd.read_csv(cfg['manifest'], dtype=str, keep_default_na=False)
    m = m[[cfg['include'](r) for r in m.to_dict('records')]]
    pred = pd.read_csv(cfg['predictions'], dtype=str, keep_default_na=False)[['subject_id', 'idh_probability']]
    r = m.merge(pred, on='subject_id')
    clin = cfg['clinical'] or (r.clinical_source.iloc[0] if 'clinical_source' in r and len(r) else None)
    ages = age_lookup(clin, r.subject_id)
    frames['RHUH'] = pd.DataFrame(dict(patient_id='RHUH:' + r.subject_id, subject_id=r.subject_id, scan_id=r.subject_id,
        y=r.idh.map(binary), probability=r.idh_probability.astype(float), age=r.subject_id.map(ages),
        t1c_path=r.t1c_path.map(resolve), flair_path=r.flair_path.map(resolve), mask_path=r.tumor_mask_path.map(resolve),
        brain_mask_path='', timepoint=0))
# ---- MU ----
cfg = COHORTS['MU']
if Path(cfg['predictions']).exists():
    mp = pd.read_csv(cfg['predictions'], dtype=str, keep_default_na=False)
    arr = Path(cfg['arrays'])
    member = lambda n: np.mean([np.load(arr / f'{n}_fold_{k}.npy')[:, 0] for k in range(1, 6)], axis=0)
    mp['p3'] = np.mean([member(n) for n in THREE], axis=0)
    four = np.mean([member(n) for n in THREE + ['v2_reference_seed43']], axis=0)
    print('MU check: four-member reconstruction max |diff| =', float(np.max(np.abs(four - mp.idh_probability.astype(float)))))
    ages = age_lookup(cfg['clinical'], mp.subject_id.unique())
    days = pd.to_numeric(mp.get('days_from_diagnosis', pd.Series(np.nan, index=mp.index)), errors='coerce').fillna(0)
    frames['MU'] = pd.DataFrame(dict(patient_id='MU:' + mp.subject_id, subject_id=mp.subject_id, scan_id=mp.patient_id,
        y=mp.idh.map(binary), probability=mp.p3, age=mp.subject_id.map(ages) + days / 365.25,
        t1c_path=mp.t1c_path.map(resolve), flair_path=mp.flair_path.map(resolve), mask_path=mp.tumor_mask_path.map(resolve),
        brain_mask_path='', timepoint=pd.to_numeric(mp.timepoint, errors='coerce')))
for name, f in frames.items():
    f['t2_path'] = [find_t2(a, b) for a, b in zip(f.t1c_path, f.flair_path)]
    for c in ('adc_path', 'fa_path', 'perf_path', 'adc_kind', 'fa_kind', 'perf_kind'): f[c] = ''
    f['site'] = name
    ok = f.mask_path.map(lambda p: bool(p) and Path(p).exists()) & f.y.isin([0, 1])
    print(f"{name}: {len(f)} scans, {ok.sum()} with mask and IDH label, {int(f.loc[ok, 'y'].sum())} IDH-mutant; "
          f"T2 found {int((f.t2_path.str.len() > 0).sum())}; age found {int(f.age.notna().sum())}")
    frames[name] = f[ok].reset_index(drop=True)
''')

md('## 4. Feature extraction (identical code to the development notebook)')
code(EXTRACT_CODE)
code(r'''
feat_frames = {}
for name, f in frames.items():
    cache = OUTPUT_DIR / f'features_{name}_{FEATURE_VERSION}.csv'
    if cache.exists():
        ff = pd.read_csv(cache)
    else:
        t0 = time.time()
        ff = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=0)(delayed(extract)(SimpleNamespace(**r)) for r in f.to_dict('records')))
        ff['scan_id'] = f.scan_id.values; ff.to_csv(cache, index=False)
        print(f'{name}: extracted {len(ff)} scans in {(time.time() - t0) / 60:.1f} min')
    ff['error'] = ff['error'].fillna('').astype(str)
    print(f"{name}: {int((ff.error != '').sum())} extraction errors", ff.loc[ff.error != '', 'error'].value_counts().head(3).to_dict())
    feat_frames[name] = f.merge(ff.drop(columns=['patient_id', 'site']), on='scan_id')
''')

md(r'''
## 5. Apply the frozen models

Imaging features are z-scored within each external cohort, the same way as each development site. For MU this uses the
analysed scans: the earliest scan per patient for the primary analysis, and all scans for the secondary one.
''')
code(r'''
CONV = ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'no_enhancement', 'necrosis_frac', 'edema_ratio_log']
MISM = ['mismatch_index', 'mismatch_frac']

def prepare(df_):
    d = df_[df_.error == ''].copy()
    d['necrosis_frac'] = d.necrosis_frac.fillna(0.0)
    for c in CONV + MISM:
        if c == 'no_enhancement': continue
        d[c] = (d[c] - d[c].mean()) / (d[c].std(ddof=0) + 1e-9)
    p = d.probability.clip(1e-6, 1 - 1e-6); d['dl_logit'] = np.log(p / (1 - p))
    return d

def predict(d, name):
    spec = frozen['models'][name]
    if name == 'Ensemble (raw)': return d.probability.values
    X = d[spec['inputs']].values.astype(float)
    z = (X - np.array(spec['scaler_mean'])) / np.array(spec['scaler_scale'])
    return 1 / (1 + np.exp(-(z @ np.array(spec['coef']) + spec['intercept'])))

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))

def exact_ci(k, n):
    if n == 0: return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1.0 if k == n else beta.ppf(.975, k + 1, n - k))

def evaluate(d, cohort, analysis, cluster=None):
    rows = []; y = d.y.values.astype(int)
    rng = np.random.default_rng(SEED)
    for name in PORTABLE_MODELS:
        if name not in frozen['models']: continue
        need = frozen['models'][name]['inputs']
        dd = d.dropna(subset=[c for c in need if c != 'probability']) if name != 'Ensemble (raw)' else d
        yy = dd.y.values.astype(int); p = predict(dd, name)
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        th = frozen['models'][name]['threshold_youden']; pred = (p >= th).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum()); fn = int(((pred == 0) & (yy == 1)).sum())
        tn = int(((pred == 0) & (yy == 0)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
        # bootstrap AUROC: resample patients (whole patients for repeated scans)
        units = dd[cluster].values if cluster else np.arange(len(dd)); uniq = np.unique(units)
        aucs = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq)); idx = np.concatenate([np.where(units == u)[0] for u in pick])
            if yy[idx].min() != yy[idx].max(): aucs.append(roc_auc_score(yy[idx], p[idx]))
        sens_ci, spec_ci = exact_ci(tp, tp + fn), exact_ci(tn, tn + fp)
        rows.append(dict(cohort=cohort, analysis=analysis, model=name, n=len(yy), idh_mutant=int(yy.sum()),
                         auroc=roc_auc_score(yy, p), auroc_ci=f'{np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f}',
                         average_precision=average_precision_score(yy, p), brier=float(np.mean((p - yy) ** 2)), ece=ece(yy, p),
                         threshold=th, sensitivity=tp / (tp + fn), sensitivity_exact_ci=f'{sens_ci[0]:.2f}-{sens_ci[1]:.2f}',
                         specificity=tn / (tn + fp), specificity_exact_ci=f'{spec_ci[0]:.2f}-{spec_ci[1]:.2f}',
                         ppv=tp / (tp + fp) if tp + fp else np.nan, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                         tp=tp, fp=fp, tn=tn, fn=fn))
    return rows

all_rows = []
if 'RHUH' in feat_frames:
    all_rows += evaluate(prepare(feat_frames['RHUH']), 'RHUH-GBM', 'treatment-naive')
if 'MU' in feat_frames:
    mu = feat_frames['MU']
    first = mu.sort_values(['subject_id', 'timepoint']).groupby('subject_id').head(1)
    all_rows += evaluate(prepare(first), 'MU-Glioma-Post', 'earliest scan per patient')
    all_rows += evaluate(prepare(mu), 'MU-Glioma-Post', 'all scans (patient-cluster bootstrap)', cluster='subject_id')
ext = pd.DataFrame(all_rows); ext.to_csv(OUTPUT_DIR / 'external_metrics.csv', index=False)
display(ext[['cohort', 'analysis', 'model', 'n', 'idh_mutant', 'auroc', 'auroc_ci', 'average_precision', 'brier', 'ece',
             'sensitivity', 'sensitivity_exact_ci', 'specificity', 'specificity_exact_ci', 'f1']].round(3))
''')

md('## 6. Distribution shift check')
code(r'''
# How far do the external features sit from development? (raw, un-standardised medians)
dev = pd.read_csv(PROJECT / 'outputs' / 'idh_imaging_correlates' / 'features.csv')
cols = ['vol_whole', 'enhancing_frac_whole', 'no_enhancement', 'edema_ratio_log', 'mismatch_index', 'nt2_nonenh_median', 'nflair_nonenh_median']
shift = [dict(cohort='development', **dev[cols].median().to_dict())]
for name, f in feat_frames.items():
    shift.append(dict(cohort=name, **f[f.error == ''][cols].median().to_dict()))
shift = pd.DataFrame(shift); shift.to_csv(OUTPUT_DIR / 'feature_shift.csv', index=False); display(shift.round(3))
print('Check: normalised T2 and FLAIR in the lesion should be near the development values (about 1.5-2.2); much higher values mean the reference region picked up background.')
for _, r in shift.iterrows():
    if r.cohort != 'development' and (r.nt2_nonenh_median > 3 or r.nflair_nonenh_median > 3):
        print(f'WARNING {r.cohort}: lesion T2/FLAIR are {r.nt2_nonenh_median:.1f}x / {r.nflair_nonenh_median:.1f}x normal brain. '
              'Intensity normalisation failed (e.g. images not skull-stripped); mismatch features for this cohort are not valid.')
print('Ensemble probability medians:', {n: round(float(f.probability.median()), 3) for n, f in feat_frames.items()})
''')

code(r'''
lines = ['# External validation of the frozen IDH hybrid models', '', f'Run: {time.strftime("%Y-%m-%d %H:%M")}', '',
         'Frozen development thresholds and coefficients; imaging features z-scored within each external cohort.', '']
cols_ = ['cohort', 'analysis', 'model', 'n', 'idh_mutant', 'auroc', 'auroc_ci', 'brier', 'ece', 'sensitivity',
         'sensitivity_exact_ci', 'specificity', 'specificity_exact_ci', 'f1']
fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
lines += ['| ' + ' | '.join(cols_) + ' |', '|' + '---|' * len(cols_)]
lines += ['| ' + ' | '.join(fmt(r[c]) for c in cols_) + ' |' for r in ext.to_dict('records')]
lines += ['', '## Feature medians (raw)', '', shift.round(3).to_string(index=False), '',
          '- RHUH has very few IDH-mutant patients: treat sensitivity as descriptive.',
          '- MU is post-treatment imaging, outside the pre-operative intended use.']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8'); print('\n'.join(lines))
''')

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'IDH_External_Validation.ipynb'
nbf.write(nb, out); print('Wrote', out)
