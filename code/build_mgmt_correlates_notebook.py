"""Build MGMT_Imaging_Correlates.ipynb: a hypothesis-driven test of published MRI correlates of
MGMT promoter methylation in IDH-wildtype glioblastoma (UCSF-PDGM, UTSW, UPENN-GBM).
Run: python build_mgmt_correlates_notebook.py [output.ipynb]
"""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

md(r'''
# MGMT imaging correlates: pre-specified hypothesis test

Tests whether imaging features reported in the literature separate **MGMT-methylated** from **unmethylated**
**IDH-wildtype glioblastoma** in the RadGenome development cohorts. No deep learning; everything is computed from
tumour sub-region masks and, where available, diffusion and perfusion maps.

| ID | Feature (primary measure) | Expected in methylated | Source | Cohorts |
|---|---|---|---|---|
| H1 | Oedema relative to tumour core (`edema_ratio_log`) | **Lower** | Suh 2018 meta-analysis | UCSF, UTSW, UPENN |
| H2 | Enhancing-tumour ADC, normalised (`nadc_enhancing_median`) | **Higher** | Suh 2018; Han 2018 | UCSF, UPENN |
| H3 | Peritumoral-oedema ADC, normalised (`nadc_edema_median`) | **Lower** | Ladenhauf 2023 | UCSF, UPENN |
| H4 | Enhancing-tumour perfusion, normalised 90th percentile (`nperf_enhancing_p90`) | **Lower** | Suh 2018; Han 2018 (ASL) | UCSF (ASL CBF), UPENN (DSC rCBV) |
| H5 | Necrosis fraction of tumour core (`necrosis_frac`) | Two-sided (conflicting reports) | Han 2018 vs Suh 2018 | UCSF, UTSW, UPENN |
| E1 | Left-hemisphere tumour (`left_hemisphere`) | Two-sided, exploratory (expected null in IDH-wt) | Ellingson 2012/13 vs Incekara 2020 | all |
| E2 | Minimum ADC: 10th percentile in tumour core (`nadc_core_p10`) | Higher, exploratory | Rundle-Thiele 2015 | UCSF, UPENN |

**Decision rule (fixed before running).** H1–H5 form the primary family. Each gets a site-stratified permutation test
(one-sided for H1–H4, two-sided for H5). Holm correction is applied across the five. A hypothesis is *supported*
if it passes Holm at alpha = 0.05 **and** the direction agrees in every cohort that has the feature.
The multivariable models are judged against a fixed bar: **cross-site AUROC >= 0.65**.

**How to run.** Set the paths in Section 1, then *Run All*. Feature extraction is CPU and disk bound and does not use the GPU.
It is cached in `OUTPUT_DIR/features.csv`, so re-runs skip patients already processed.
Outputs: `features.csv`, `qc_report.csv`, `hypothesis_results.csv`, `model_results.csv`, figures and `REPORT.md`.
''')

md('## 1. Configuration')
code(r'''
from pathlib import Path

PROJECT = Path.cwd()                                    # RadGenome project folder
MANIFEST = PROJECT / 'outputs' / 'patient_manifest.csv' # one row per patient: site, labels, image paths
OUTPUT_DIR = PROJECT / 'outputs' / 'mgmt_imaging_correlates'

# Map logical names to manifest columns. None = auto-detect from the candidates below.
COLUMN_MAP = dict(patient_id=None, site=None, idh=None, mgmt=None, t1c=None, mask=None,
                  brain_mask=None, age=None, grade=None)
# Optional manifest columns that override automatic map discovery, if present:
#   adc_path, fa_path, perf_path (perfusion: UCSF ASL CBF or UPENN DSC rCBV)

GRADE4_ONLY = True                 # keep WHO grade 4 when a grade column exists
ASSUME_GRADE4_SITES = {'UPENN'}    # cohorts that are glioblastoma-only by design
EXCLUDE_HOLDOUT = False            # True: drop the reserved 200-patient IDH test set (no model is trained here)
HOLDOUT_JSON = PROJECT / 'outputs' / 'balanced_holdout_ids.json'
# Tumour-mask paths, if the manifest has none: RadGenome keeps them in the v2 mask audit (patient_id, tumor_mask_path, ...)
MASK_TABLE = PROJECT / 'outputs' / 'v2_mask_content_audit.csv'
# WHO grade and age, if the manifest has none: read from each row's clinical file in the 'metadata_path' column
USE_METADATA_FILES = True
# Old -> new folder prefixes for data that moved since the manifest was written (applied only when the old path is missing)
PATH_REMAP = {
    r'C:\Users\claks\Downloads\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5': r'C:\Users\claks\Downloads\UCSF-PDGM',
}

N_JOBS = 6                 # parallel workers for feature extraction (CPU cores; GPU unused)
RECOMPUTE = False          # True: ignore the feature cache
MIN_REGION_ML = 0.2        # minimum sub-region volume (mL) for a map statistic
REFERENCE_MARGIN_MM = 10   # tumour dilation excluded from the normal-appearing reference region
N_BOOT = 2000
N_PERM = 10000
SEED = 20261021
SUCCESS_AUROC = 0.65

# Optional external cohorts for the conventional-feature model (needs tumour masks and MGMT), e.g. OpenNeuro ds007045.
# Each: dict(name='ds007045', manifest=Path(...), column_map=dict(...), idh_wildtype_only=False)
EXTERNAL_COHORTS = []
''')

md('## 2. Setup')
code(r'''
import json, re, os, warnings, platform, time, hashlib
from types import SimpleNamespace
import numpy as np, pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.stats import rankdata, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng_global = np.random.default_rng(SEED)
print('Python', platform.python_version(), '| numpy', np.__version__, '| nibabel', nib.__version__,
      '| pandas', pd.__version__)
print('Manifest:', MANIFEST, '| exists:', MANIFEST.exists())
''')

md('## 3. Cohort: IDH-wildtype, MGMT known')
code(r'''
CANDIDATES = dict(
    patient_id=['patient_id', 'subject_id', 'id'],
    site=['site', 'cohort', 'dataset'],
    idh=['idh', 'idh_label', 'idh_status'],
    mgmt=['mgmt', 'mgmt_label', 'mgmt_status'],
    t1c=['t1c_path', 't1c', 't1gd_path', 't1ce_path'],
    mask=['tumor_mask_path', 'tumour_mask_path', 'mask_path', 'seg_path', 'segmentation_path'],
    brain_mask=['brain_mask_path', 'brainmask_path'],
    age=['age', 'age_at_imaging', 'age_years', 'patient_age'],
    grade=['who_grade', 'grade', 'who_cns_grade', 'tumor_grade'])

raw = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
print('Manifest columns:', list(raw.columns))
has_mask_col = any(c.lower() in CANDIDATES['mask'] for c in raw.columns) or COLUMN_MAP.get('mask')
if not has_mask_col and MASK_TABLE and Path(MASK_TABLE).exists():
    mt = pd.read_csv(MASK_TABLE, dtype=str, keep_default_na=False)
    keep = ['patient_id', 'tumor_mask_path'] + [c for c in ('source', 'content_status', 'fallback') if c in mt.columns]
    mt = mt[keep].rename(columns={'source': 'mask_source'})
    raw = raw.merge(mt, on='patient_id', how='left')
    print(f'Tumour masks taken from {Path(MASK_TABLE).name}: {(raw.tumor_mask_path.str.len() > 0).sum()} of {len(raw)} rows have a path')
    if 'content_status' in raw:
        bad = raw.content_status.isin(['empty', 'missing']) | (raw.get('fallback', '') == 'True')
        raw.loc[bad, 'tumor_mask_path'] = ''
        print(f'  {bad.sum()} rows with empty/missing masks (brain-crop fallback) are excluded')

def norm_name(c):
    return ' ' + re.sub(r'[^a-z0-9]+', ' ', c.lower()) + ' '

def metadata_lookup(raw):
    # Per clinical file: the ID column = the one sharing most values with the patients' IDs;
    # grade = first column named like 'grade'; age = first mostly-numeric column named like 'age'.
    grade, age = {}, {}
    if 'metadata_path' not in raw: return grade, age
    for meta, g in raw.groupby('metadata_path'):
        if not meta or not Path(meta).exists():
            print('  metadata not found:', meta); continue
        sep = '\t' if meta.lower().endswith(('.tsv', '.txt')) else ','
        md_ = pd.read_csv(meta, sep=sep, dtype=str, keep_default_na=False, encoding_errors='replace')
        to_pid = {}
        for r in g.itertuples():
            for k in {str(getattr(r, 'subject_id', '')), r.patient_id.split(':', 1)[-1], r.patient_id}:
                if k:
                    to_pid[k.strip()] = r.patient_id
                    to_pid.setdefault(re.sub(r'_\d+$', '', k.strip()), r.patient_id)   # UPENN-GBM-00001_11 -> UPENN-GBM-00001
        idcol = max(md_.columns, key=lambda c: md_[c].str.strip().isin(to_pid).sum())
        gcol = next((c for c in md_.columns if ' grade ' in norm_name(c)), None)
        acol = next((c for c in md_.columns if ' age ' in norm_name(c)
                     and pd.to_numeric(md_[c], errors='coerce').notna().mean() > .5), None)
        hits = 0
        for row in md_.to_dict('records'):
            pid = to_pid.get(str(row[idcol]).strip())
            if not pid: continue
            hits += 1
            if gcol: grade[pid] = row[gcol]
            if acol: age[pid] = row[acol]
        print(f'  {Path(meta).name}: id column {idcol!r} matched {hits}/{len(g)}; grade column {gcol!r}; age column {acol!r}')
    return grade, age

if USE_METADATA_FILES and (COLUMN_MAP.get('grade') is None or COLUMN_MAP.get('age') is None):
    lower_cols = {c.lower() for c in raw.columns}
    need_grade = not any(c in lower_cols for c in CANDIDATES['grade'])
    need_age = not any(c in lower_cols for c in CANDIDATES['age'])
    if need_grade or need_age:
        print('Reading grade/age from clinical metadata files:')
        g_map, a_map = metadata_lookup(raw)
        if need_grade and g_map: raw['who_grade'] = raw.patient_id.map(g_map).fillna('')
        if need_age and a_map: raw['age'] = raw.patient_id.map(a_map).fillna('')
cols = {}
for key, cands in CANDIDATES.items():
    chosen = COLUMN_MAP.get(key)
    if chosen is None:
        lower = {c.lower(): c for c in raw.columns}
        chosen = next((lower[c] for c in cands if c in lower), None)
    cols[key] = chosen
missing = [k for k in ('patient_id', 'site', 'idh', 'mgmt', 't1c', 'mask') if cols[k] is None]
if missing:
    raise ValueError(f'Set COLUMN_MAP for: {missing}')
print('Using columns:', cols)

def binary(v):
    v = str(v).strip().lower()
    if v in ('1', '1.0', 'true', 'mutant', 'mutated', 'methylated', 'positive', 'yes'):
        return 1
    if v in ('0', '0.0', 'false', 'wildtype', 'wild-type', 'wt', 'unmethylated', 'negative', 'no'):
        return 0
    return np.nan

def grade4(v):
    v = str(v).strip().upper().replace('GRADE', '').replace('WHO', '').replace('CNS', '').strip(' :-')
    if v in ('', 'NAN', 'NONE', 'UNKNOWN', 'NA', 'N/A'):
        return np.nan
    if v in ('4', '4.0', 'IV') or 'GLIOBLASTOMA' in v or v == 'GBM':
        return 1.0
    if v in ('1', '2', '3', '1.0', '2.0', '3.0', 'I', 'II', 'III'):
        return 0.0
    return np.nan

df = pd.DataFrame(dict(
    patient_id=raw[cols['patient_id']], site=raw[cols['site']].str.upper(),
    idh=raw[cols['idh']].map(binary), mgmt=raw[cols['mgmt']].map(binary),
    t1c_path=raw[cols['t1c']], mask_path=raw[cols['mask']],
    brain_mask_path=raw[cols['brain_mask']] if cols['brain_mask'] else '',
    age=pd.to_numeric(raw[cols['age']], errors='coerce') if cols['age'] else np.nan,
    grade4=raw[cols['grade']].map(grade4) if cols['grade'] else np.nan,
    mask_source=raw['mask_source'] if 'mask_source' in raw else ''))
for extra in ('adc_path', 'fa_path', 'perf_path'):
    df[extra] = raw[extra] if extra in raw.columns else ''
if df.patient_id.duplicated().any():
    raise ValueError('Duplicate patient IDs in manifest')

steps = [('manifest', len(df))]
if EXCLUDE_HOLDOUT and HOLDOUT_JSON.exists():
    obj = json.loads(HOLDOUT_JSON.read_text())
    ids = set()
    def walk(o):
        if isinstance(o, str): ids.add(o)
        elif isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, list): [walk(v) for v in o]
    walk(obj)
    df = df[~df.patient_id.isin(ids)]; steps.append(('minus reserved holdout', len(df)))
df = df[df.idh == 0]; steps.append(('IDH-wildtype', len(df)))
df = df[df.mgmt.isin([0, 1])]; steps.append(('MGMT known', len(df)))
if GRADE4_ONLY and cols['grade']:
    print('Grade-4 status by site (NaN = grade not found):')
    display(pd.crosstab(df.site, df.grade4.map({1.0: 'grade 4', 0.0: 'grade 2-3'}).fillna('unknown')))
    keep = (df.grade4 == 1) | (df.grade4.isna() & df.site.isin(ASSUME_GRADE4_SITES))
    df = df[keep]; steps.append(('WHO grade 4', len(df)))
elif GRADE4_ONLY:
    print('No grade column: cohort is IDH-wildtype adult diffuse glioma (mostly glioblastoma under WHO 2021).')
def resolve(p):
    if not p: return ''
    q = Path(p)
    q = q if q.is_absolute() else (MANIFEST.parent / q).resolve()
    if not q.exists():
        for old, new in PATH_REMAP.items():
            if str(q).lower().startswith(old.lower()):
                cand = Path(new + str(q)[len(old):])
                if cand.exists(): return str(cand)
    return str(q)
for c in ('t1c_path', 'mask_path', 'brain_mask_path', 'adc_path', 'fa_path', 'perf_path'):
    df[c] = df[c].map(resolve)
df = df[df.mask_path.str.len() > 0]; steps.append(('tumour mask listed', len(df)))
exists = df.mask_path.map(lambda p: Path(p).exists()) & df.t1c_path.map(lambda p: Path(p).exists())
if (~exists).any():
    print(f'{(~exists).sum()} patients have a missing T1c or mask file, e.g.:', df.loc[~exists, 'mask_path'].head(3).tolist())
df = df[exists]; steps.append(('files present', len(df)))
df = df.reset_index(drop=True)
display(pd.DataFrame(steps, columns=['step', 'patients']))
display(pd.crosstab(df.site, df.mgmt.map({1: 'methylated', 0: 'unmethylated'}), margins=True))
''')

md(r'''
## 4. Find diffusion and perfusion maps

Searches each cohort's folder tree once and matches files to patients by ID prefix. `UCSF-PDGM-004` does not match
`UCSF-PDGM-0040`. Expected names: UCSF `*_ADC`, `*_ASL`, `*_DTI*_FA`/`*_DTI*_MD`; UPENN `images_DTI/*_DTI_TR`,
`*_DTI_FA`, `images_DSC/*_DSC_ap-rCBV`. Check the table below before running extraction. Columns `adc_path`, `fa_path` and
`perf_path` in the manifest override the search.
''')
code(r'''
ID_PATTERNS = [r'UCSF-PDGM-\d+', r'UPENN-GBM-\d+_\d+', r'BT\d{3,5}']
MAP_PATTERNS = {   # (kind, regex on lower-case filename); first matching rule wins per map
    'adc':  [('adc', r'(^|_)adc(_bias)?\.nii'), ('md', r'dti[^/]*_md\.nii'), ('trace', r'dti_tr\.nii')],
    'fa':   [('fa', r'(dti[^/]*_)?fa\.nii')],
    'perf': [('asl_cbf', r'(^|_)asl(_cbf)?\.nii'), ('dsc_rcbv', r'rcbv\.nii')],
}
EXCLUDE = re.compile(r'(seg|mask|label|_bias_field)')

def id_prefix(t1c_path):
    name = Path(t1c_path).name
    for pat in ID_PATTERNS:
        m = re.search(pat, name)
        if m: return m.group(0)
    return Path(t1c_path).parent.name

SITE_ROOTS = {}   # optional override, e.g. {'UCSF': Path(r'C:/.../UCSF-PDGM-v3')}

def site_root(site, paths):
    if site in SITE_ROOTS: return Path(SITE_ROOTS[site])
    common = Path(os.path.commonpath([str(Path(p).parent) for p in paths]))
    if len(paths) == 1: common = common.parent
    # UPENN keeps structural, DTI and DSC in sibling 'images_*' folders
    return common.parent if common.name.lower().startswith('images_') else common

index = {}
for site, g in df.groupby('site'):
    paths = [p for p in g.t1c_path if p]
    root = site_root(site, paths)
    t0 = time.time()
    files = [p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.nii.gz') or p.name.endswith('.nii'))]
    index[site] = files
    print(f'{site}: scanned {root} -> {len(files)} NIfTI files in {time.time()-t0:.1f}s')

def find_maps(row):
    out = {}
    prefix = id_prefix(row.t1c_path)
    rx = re.compile(r'(^|[/\\])' + re.escape(prefix) + r'(_|\.|$)')
    t1c_dir = Path(row.t1c_path).parent
    mine = [f for f in index.get(row.site, []) if rx.search(f.name) or rx.search(f.parent.name)]
    same_dir = [f for f in mine if f.parent == t1c_dir]
    # UCSF keeps every map beside the T1c; UPENN uses sibling images_* folders (never a follow-up folder of another exam)
    mine = same_dir if any(re.search(p_, f.name.lower()) for f in same_dir for rules in MAP_PATTERNS.values() for _, p_ in rules) \
        else [f for f in mine if not re.search(r'_fu\d+d', str(f).lower())]
    for key, rules in MAP_PATTERNS.items():
        override = getattr(row, f'{key}_path', '')
        if override:
            out[key], out[key + '_kind'] = override, 'manifest'
            continue
        out[key], out[key + '_kind'] = '', ''
        for kind, pat in rules:
            hits = sorted([f for f in mine if re.search(pat, f.name.lower()) and not EXCLUDE.search(f.name.lower())],
                          key=lambda f: len(f.name))
            if hits:
                out[key], out[key + '_kind'] = str(hits[0]), kind
                break
    return out

found = pd.DataFrame([find_maps(r) for r in df.itertuples()])
df = pd.concat([df.drop(columns=['adc_path', 'fa_path', 'perf_path']), found.rename(
    columns={'adc': 'adc_path', 'fa': 'fa_path', 'perf': 'perf_path'})], axis=1)
avail = df.assign(**{k: df[f'{k}_path'].str.len() > 0 for k in ('adc', 'fa', 'perf')}).groupby('site')[['adc', 'fa', 'perf']].sum()
avail['patients'] = df.groupby('site').size()
display(avail)
display(df.groupby(['site', 'adc_kind', 'perf_kind'], dropna=False).size().rename('patients').reset_index())
''')

md(r'''
## 5. Feature extraction

Everything is computed on each patient's tumour-mask grid after reorienting to RAS. A map on a different grid is resampled
to the mask grid with linear interpolation, and this is logged in the QC table. Sub-regions use BraTS labels:
1 = necrosis, 2 = oedema, 4 = enhancing; label 3 is non-enhancing core when 4 is present (BraTS 2017) and enhancing tumour otherwise (BraTS 2023). Masks containing only label 3 are excluded as ambiguous.

**Normalisation.** Each map is divided by the median of normal-appearing brain in the contralateral hemisphere. The
tumour, dilated by `REFERENCE_MARGIN_MM`, is excluded, and so are the top 20% of values (CSF for ADC, vessels for
perfusion). This makes the ratios unit-free, so UCSF ADC and UPENN DTI trace can be compared, and so can ASL CBF and DSC rCBV.
''')
code(r'''

def canonical(path):
    return nib.as_closest_canonical(nib.load(str(path)))

def on_grid(path, ref, order):
    img = canonical(path)
    if img.ndim == 4:
        if img.shape[3] != 1:
            raise ValueError(f'4D image, expected a parameter map: {path}')
        img = img.slicer[..., 0]
    same = img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3)
    if not same:
        img = resample_from_to(img, (ref.shape[:3], ref.affine), order=order)
    return np.asarray(img.dataobj, dtype=np.float32), (not same)

def region_stats(arr, region, voxel_ml, positive=True):
    vals = arr[region & np.isfinite(arr)]
    if positive: vals = vals[vals > 0]
    if vals.size * voxel_ml < MIN_REGION_ML: return np.nan, np.nan, np.nan
    return float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))

def extract(row):
    rec = dict(patient_id=row.patient_id, site=row.site, error='')
    try:
        mimg = canonical(row.mask_path)
        lab = np.rint(np.asarray(mimg.dataobj)).astype(np.int16)
        labels = sorted(int(v) for v in np.unique(lab) if v > 0)
        rec['mask_labels'] = ' '.join(map(str, labels))
        if not labels: raise ValueError('empty tumour mask')
        unexpected = set(labels) - {1, 2, 3, 4}
        if unexpected: raise ValueError(f'unexpected mask labels {sorted(unexpected)}')
        if labels == [3]: raise ValueError('only label 3: enhancing (BraTS 2023) or non-enhancing (BraTS 2017) is ambiguous')
        # BraTS <=2017: 1 necrosis, 2 oedema, 3 non-enhancing tumour, 4 enhancing.  BraTS 2023: 1, 2, 3 = enhancing.
        if 4 in labels:
            regions = dict(necrosis=(1, 3), edema=(2,), enhancing=(4,)); rec['label_convention'] = 'brats2017' if 3 in labels else 'brats2020'
        else:
            regions = dict(necrosis=(1,), edema=(2,), enhancing=(3,)); rec['label_convention'] = 'brats2023' if 3 in labels else 'no_enhancing'
        voxel_ml = abs(np.linalg.det(mimg.affine[:3, :3])) / 1000.0
        reg = {k: np.isin(lab, v) for k, v in regions.items()}
        reg['core'] = reg['necrosis'] | reg['enhancing']
        reg['whole'] = lab > 0
        for k in ('enhancing', 'necrosis', 'edema', 'core', 'whole'):
            rec[f'vol_{k}'] = float(reg[k].sum() * voxel_ml)
            rec[f'log_vol_{k}'] = float(np.log1p(rec[f'vol_{k}']))
        core = rec['vol_core']
        rec['edema_ratio_log'] = float(np.log((rec['vol_edema'] + 0.1) / (core + 0.1)))
        rec['necrosis_frac'] = rec['vol_necrosis'] / core if core > 0 else np.nan
        rec['enhancing_frac'] = rec['vol_enhancing'] / core if core > 0 else np.nan

        # brain mask: provided, else stripped T1c support
        if row.brain_mask_path:
            brain, _ = on_grid(row.brain_mask_path, mimg, order=0); brain = brain > 0
        else:
            t1c, _ = on_grid(row.t1c_path, mimg, order=1); brain = t1c > 0
        brain |= reg['whole']
        ijk = np.argwhere(reg['whole']).mean(0); bijk = np.argwhere(brain).mean(0)
        to_world = lambda p: mimg.affine[:3, :3] @ p + mimg.affine[:3, 3]
        tx, bx = to_world(ijk)[0], to_world(bijk)[0]          # RAS: larger x = right
        rec['left_hemisphere'] = float(tx < bx)
        rec['midline_offset_mm'] = float(abs(tx - bx))
        A = mimg.affine.astype(np.float32); ii, jj, kk = (np.arange(n, dtype=np.float32) for n in lab.shape)
        xs = A[0, 0] * ii[:, None, None] + A[0, 1] * jj[None, :, None] + A[0, 2] * kk[None, None, :] + A[0, 3]
        contra = (xs > bx) if tx < bx else (xs < bx)
        spacing = np.sqrt((mimg.affine[:3, :3] ** 2).sum(0))
        it = max(1, int(round(REFERENCE_MARGIN_MM / spacing.min())))
        dil = ndimage.binary_dilation(reg['whole'], iterations=it)
        refmask = brain & contra & ~dil

        resampled = []
        for key in ('adc', 'fa', 'perf'):
            path = getattr(row, f'{key}_path')
            if not path: continue
            arr, rs = on_grid(path, mimg, order=1)
            if rs: resampled.append(key)
            rec[f'{key}_kind'] = getattr(row, f'{key}_kind')
            refvals = arr[refmask & np.isfinite(arr) & (arr > 0)]
            ref = np.nan
            if refvals.size > 100:
                cut = np.percentile(refvals, 80)
                ref = float(np.median(refvals[refvals <= cut]))
            rec[f'{key}_reference'] = ref
            for r in ('enhancing', 'necrosis', 'edema', 'core'):
                med, p10, p90 = region_stats(arr, reg[r], voxel_ml)
                rec[f'{key}_{r}_median'], rec[f'{key}_{r}_p10'], rec[f'{key}_{r}_p90'] = med, p10, p90
                for stat, v in (('median', med), ('p10', p10), ('p90', p90)):
                    rec[f'n{key}_{r}_{stat}'] = v / ref if ref and np.isfinite(ref) and ref > 0 else np.nan
        rec['resampled_maps'] = ' '.join(resampled)
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec

FEATURE_VERSION = 'v2-labels'   # bump when extraction logic changes so cached rows are recomputed

def signature(row):
    parts = []
    for p in (row.mask_path, row.brain_mask_path, row.t1c_path, row.adc_path, row.fa_path, row.perf_path):
        if p and Path(p).exists():
            s = Path(p).stat(); parts.append(f'{p}|{s.st_size}|{int(s.st_mtime)}')
    return hashlib.md5('||'.join(parts + [str(MIN_REGION_ML), str(REFERENCE_MARGIN_MM), FEATURE_VERSION]).encode()).hexdigest()

cache_path = OUTPUT_DIR / 'features.csv'
df['signature'] = [signature(r) for r in df.itertuples()]
cached = pd.read_csv(cache_path, dtype={'patient_id': str, 'signature': str}) if cache_path.exists() and not RECOMPUTE else None
todo = df if cached is None else df[~df.set_index(['patient_id', 'signature']).index.isin(
    cached.set_index(['patient_id', 'signature']).index)]
print(f'{len(df)} patients, {len(todo)} to extract with {N_JOBS} workers')
t0 = time.time()
new = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(extract)(SimpleNamespace(**r._asdict())) for r in todo.itertuples()) if len(todo) else []
print(f'Extraction took {(time.time()-t0)/60:.1f} min')
new = pd.DataFrame(new)
if len(new): new['signature'] = todo.signature.values
feats = pd.concat([x for x in (cached, new) if x is not None and len(x)], ignore_index=True)
feats = feats[feats.patient_id.isin(df.patient_id)].drop_duplicates('patient_id', keep='last')
feats.to_csv(cache_path, index=False)
data = df[['patient_id', 'site', 'mgmt', 'age']].merge(feats.drop(columns=['site']), on='patient_id')
errors = data[data.error.fillna('').str.len() > 0]
print(f'Features for {len(data)} patients; {len(errors)} errors')
if len(errors): display(errors[['patient_id', 'site', 'error']].head(20))
data = data[data.error.fillna('').str.len() == 0].reset_index(drop=True)
if data.empty:
    raise RuntimeError('Feature extraction failed for every patient: see the error column above')
''')

md('## 6. Quality control')
code(r'''
qc_cols = ['vol_whole', 'vol_enhancing', 'vol_edema', 'nadc_enhancing_median', 'nadc_edema_median',
           'nperf_enhancing_p90', 'adc_reference', 'perf_reference']
qc = data.groupby('site').agg(
    patients=('patient_id', 'size'),
    methylated=('mgmt', 'sum'),
    with_adc=('nadc_enhancing_median', lambda s: s.notna().sum()),
    with_perf=('nperf_enhancing_p90', lambda s: s.notna().sum()),
    resampled=('resampled_maps', lambda s: (s.fillna('').str.len() > 0).sum()),
    median_whole_ml=('vol_whole', 'median'))
display(qc)
display(data.groupby('site')['mask_labels'].value_counts().rename('patients').reset_index())
desc = data.groupby('site')[[c for c in qc_cols if c in data]].describe(percentiles=[.05, .5, .95]).T
display(desc)
pd.concat([qc, data.groupby('site')['mask_labels'].agg(lambda s: '; '.join(sorted(set(s))))], axis=1).to_csv(OUTPUT_DIR / 'qc_report.csv')
print('Check: normalised ADC in enhancing tumour should mostly sit near 1.0-1.6; values far outside suggest a unit or registration problem.')
''')

md(r'''
## 7. Pre-specified hypothesis tests

**Effect size.** The AUROC of the feature for methylated tumours, oriented in the hypothesised direction: above 0.5
means the data agree with the hypothesis. For two-sided hypotheses it is the AUROC of "higher in methylated".

**Pooled estimate.** The within-site, pair-weighted AUROC: only methylated–unmethylated pairs from the same site are
compared, so site differences in scanner or units cannot create an effect.

**Inference.** P-values come from a permutation test that shuffles labels within site (`N_PERM` permutations). CIs come
from a bootstrap stratified by site and class (`N_BOOT` resamples).
''')
code(r'''
HYPOTHESES = [
    dict(id='H1', feature='edema_ratio_log', direction='lower', primary=True, label='Oedema / core volume (log ratio)'),
    dict(id='H2', feature='nadc_enhancing_median', direction='higher', primary=True, label='Enhancing-tumour ADC (normalised)'),
    dict(id='H3', feature='nadc_edema_median', direction='lower', primary=True, label='Peritumoral-oedema ADC (normalised)'),
    dict(id='H4', feature='nperf_enhancing_p90', direction='lower', primary=True, label='Enhancing-tumour perfusion p90 (normalised)'),
    dict(id='H5', feature='necrosis_frac', direction='two-sided', primary=True, label='Necrosis fraction of core'),
    dict(id='E1', feature='left_hemisphere', direction='two-sided', primary=False, label='Left hemisphere'),
    dict(id='E2', feature='nadc_core_p10', direction='higher', primary=False, label='Minimum ADC (core p10, normalised)'),
    dict(id='E3', feature='log_vol_edema', direction='lower', primary=False, label='Oedema volume (log mL)'),
    dict(id='E4', feature='nfa_edema_median', direction='two-sided', primary=False, label='Oedema FA (normalised)'),
]

def auc_from_ranks(r, y):
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def site_blocks(sub, feature, direction):
    blocks = []
    for site, g in sub.groupby('site'):
        g = g[g[feature].notna()]
        y = g.mgmt.values.astype(int)
        if y.sum() < 5 or (len(y) - y.sum()) < 5: continue
        x = g[feature].values.astype(float)
        if direction == 'lower': x = -x
        blocks.append((site, x, y))
    return blocks

def pooled_auc(blocks):
    num = den = 0.0
    for _, x, y in blocks:
        w = y.sum() * (len(y) - y.sum()); num += w * auc_from_ranks(rankdata(x), y); den += w
    return num / den

def perm_test(blocks, observed, two_sided, rng):
    stats = np.zeros(N_PERM); den = 0.0
    for _, x, y in blocks:
        r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1; w = n1 * n0; den += w
        perms = rng.permuted(np.tile(y, (N_PERM, 1)), axis=1)
        stats += w * ((perms * r).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    stats /= den
    if two_sided:
        return (1 + np.sum(np.abs(stats - .5) >= abs(observed - .5) - 1e-12)) / (N_PERM + 1)
    return (1 + np.sum(stats >= observed - 1e-12)) / (N_PERM + 1)

def boot_ci(blocks, rng):
    vals = []
    for _ in range(N_BOOT):
        bb = []
        for site, x, y in blocks:
            i1 = rng.choice(np.where(y == 1)[0], y.sum()); i0 = rng.choice(np.where(y == 0)[0], (y == 0).sum())
            idx = np.concatenate([i1, i0]); bb.append((site, x[idx], y[idx]))
        vals.append(pooled_auc(bb))
    return np.percentile(vals, [2.5, 97.5])

rows = []
rng = np.random.default_rng(SEED)
for h in HYPOTHESES:
    if h['feature'] not in data:
        print('Not available:', h['feature']); continue
    direction = 'higher' if h['direction'] == 'two-sided' else h['direction']
    blocks = site_blocks(data, h['feature'], direction)
    if not blocks:
        print('Too few cases for', h['id']); continue
    for site, x, y in blocks:
        a = auc_from_ranks(rankdata(x), y); lo, hi = boot_ci([(site, x, y)], rng)
        alt = 'two-sided' if h['direction'] == 'two-sided' else 'greater'
        p = mannwhitneyu(x[y == 1], x[y == 0], alternative=alt).pvalue
        sign = -1 if direction == 'lower' else 1
        rows.append(dict(id=h['id'], label=h['label'], feature=h['feature'], direction=h['direction'], primary=h['primary'],
                         cohort=site, n=len(y), methylated=int(y.sum()), oriented_auroc=a, ci_low=lo, ci_high=hi, p=p,
                         median_methylated=sign * np.median(x[y == 1]), median_unmethylated=sign * np.median(x[y == 0])))
    a = pooled_auc(blocks); lo, hi = boot_ci(blocks, rng)
    p = perm_test(blocks, a, h['direction'] == 'two-sided', rng)
    p2 = perm_test(blocks, a, True, rng)
    rows.append(dict(id=h['id'], label=h['label'], feature=h['feature'], direction=h['direction'], primary=h['primary'],
                     cohort='Pooled (within-site)', n=sum(len(y) for _, _, y in blocks),
                     methylated=int(sum(y.sum() for _, _, y in blocks)), oriented_auroc=a, ci_low=lo, ci_high=hi, p=p,
                     p_two_sided=p2,
                     sites=' '.join(s for s, _, _ in blocks)))
res = pd.DataFrame(rows)

# Holm correction across the primary family (pooled tests)
pooled = res[(res.cohort == 'Pooled (within-site)') & res.primary].sort_values('p')
m = len(pooled); holm = {}
running = 0
for k, (i, r) in enumerate(pooled.iterrows()):
    running = max(running, min(1, (m - k) * r.p)); holm[i] = running
res['p_holm'] = res.index.map(holm)
def verdict(h_id):
    sub = res[res.id == h_id]; pool = sub[sub.cohort == 'Pooled (within-site)']
    if pool.empty or not pool.primary.iloc[0]: return ''
    sites = sub[sub.cohort != 'Pooled (within-site)']
    consistent = (sites.oriented_auroc > .5).all() if pool.direction.iloc[0] != 'two-sided' else \
        ((sites.oriented_auroc > .5).all() or (sites.oriented_auroc < .5).all())
    if pool.p_holm.iloc[0] < .05 and consistent: return 'supported'
    if pool.p_holm.iloc[0] < .05: return 'significant but inconsistent across sites'
    return 'not supported'
res['verdict'] = res.id.map({h: verdict(h) for h in res.id.unique()})
res.to_csv(OUTPUT_DIR / 'hypothesis_results.csv', index=False)
show = res[['id', 'label', 'cohort', 'n', 'methylated', 'oriented_auroc', 'ci_low', 'ci_high', 'p', 'p_holm', 'verdict']]
display(show.round(3))
''')

code(r'''
# Forest plot of oriented AUROC per hypothesis and cohort
plot = res.copy()
order = [h['id'] for h in HYPOTHESES if h['id'] in set(plot.id)]
cohorts = ['UCSF', 'UTSW', 'UPENN', 'Pooled (within-site)']
fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(plot) + 1))
y = 0; ticks = []; labels = []
for hid in order:
    sub = plot[plot.id == hid]
    for c in cohorts + sorted(set(sub.cohort) - set(cohorts)):
        r = sub[sub.cohort == c]
        if r.empty: continue
        r = r.iloc[0]; pooled_row = c.startswith('Pooled')
        color = '#2a6fdb' if pooled_row else '#8a8a86'
        ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=1.6)
        ax.plot(r.oriented_auroc, y, 'D' if pooled_row else 'o', color=color, ms=6 if pooled_row else 4.5)
        ticks.append(y); labels.append(f'{hid} {r.label[:34]} | {c} (n={r.n})' if pooled_row else f'      {c} (n={r.n})')
        y -= 1
    y -= 0.6
ax.axvline(.5, color='#666', ls='--', lw=.8)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('AUROC, hypothesised direction (> 0.5 agrees)')
ax.set_xlim(max(0, min(.3, plot.ci_low.min() - .03)), min(1.0, max(.8, plot.ci_high.max() + .03)))
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'hypothesis_forest.png', dpi=200); plt.show()
''')

code(r'''
# Distributions of the advanced-imaging features by MGMT status
feats_plot = [f for f in ('edema_ratio_log', 'nadc_enhancing_median', 'nadc_edema_median', 'nperf_enhancing_p90') if f in data]
sites = sorted(data.site.unique())
fig, axes = plt.subplots(len(feats_plot), len(sites), figsize=(3.2 * len(sites), 2.4 * len(feats_plot)), squeeze=False)
for i, f in enumerate(feats_plot):
    for j, s in enumerate(sites):
        ax = axes[i, j]; g = data[(data.site == s) & data[f].notna()]
        if g.mgmt.nunique() < 2:
            ax.set_axis_off(); ax.set_title(f'{s}: no {f}', fontsize=8); continue
        parts = [g.loc[g.mgmt == 0, f], g.loc[g.mgmt == 1, f]]
        ax.boxplot(parts, widths=.5, showfliers=False); ax.set_xticks([1, 2]); ax.set_xticklabels(['unmeth.', 'meth.'])
        for k, p in enumerate(parts):
            ax.scatter(np.full(len(p), k + 1) + rng_global.uniform(-.12, .12, len(p)), p, s=4, alpha=.35, color='#2a6fdb')
        ax.set_title(f'{s}: {f}', fontsize=8)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=200); plt.show()
''')

md(r'''
## 8. Multivariable models: do the features predict MGMT across sites?

* **M1 conventional** (volumes, oedema ratio, necrosis and enhancing fractions): leave-one-site-out across all three sites.
* **M2 conventional + diffusion**: train on UCSF, test on UPENN, and the reverse. These are the only cohorts with ADC.
* **M3 perfusion, exploratory**: within-site repeated 5-fold CV for UCSF (ASL) and UPENN (DSC). The question is whether
  perfusion adds to conventional + diffusion. No cross-site test exists because the perfusion types differ.
* **Age baseline**, if the manifest has age.

Features are z-scored within each site without using labels, then fed to an L2 logistic regression. Rows missing a
feature in the set are dropped for that model. Success bar: AUROC >= 0.65 on the held-out site.
''')
code(r'''
CONV = ['log_vol_enhancing', 'log_vol_necrosis', 'log_vol_edema', 'edema_ratio_log', 'necrosis_frac', 'enhancing_frac']
DIFF = ['nadc_enhancing_median', 'nadc_core_p10', 'nadc_edema_median']
PERF = ['nperf_enhancing_p90', 'nperf_enhancing_median']

def site_z(frame, cols_):
    out = frame.copy()
    for c in cols_:
        out[c] = frame.groupby('site')[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return out

def model():
    return LogisticRegression(C=1.0, max_iter=2000)

def boot_auc(y, p, rng, reps=N_BOOT):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; vals = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])

results = []; rng = np.random.default_rng(SEED + 1)
def record(name, setting, train, test, y, p, n_feat):
    lo, hi = boot_auc(y, p, rng)
    results.append(dict(model=name, setting=setting, train=train, test=test, n_test=len(y), methylated=int(y.sum()),
                        features=n_feat, auroc=roc_auc_score(y, p), ci_low=lo, ci_high=hi,
                        meets_bar=bool(roc_auc_score(y, p) >= SUCCESS_AUROC)))

def loso(name, feats_, sites_, require=()):
    d = site_z(data[data.site.isin(sites_)].dropna(subset=list(feats_) + list(require)), feats_)
    for test in sites_:
        tr, te = d[d.site != test], d[d.site == test]
        if te.mgmt.nunique() < 2 or tr.mgmt.nunique() < 2: continue
        clf = model().fit(tr[feats_], tr.mgmt); p = clf.predict_proba(te[feats_])[:, 1]
        record(name, 'cross-site', '+'.join(sorted(set(tr.site))), test, te.mgmt.values.astype(int), p, len(feats_))

def within(name, feats_, site, repeats=5):
    d = site_z(data[data.site == site].dropna(subset=feats_), feats_)
    if len(d) < 30 or d.mgmt.nunique() < 2: return
    y = d.mgmt.values.astype(int); P = np.zeros((repeats, len(d)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(d, y):
            P[r, te] = model().fit(d.iloc[tr][feats_], y[tr]).predict_proba(d.iloc[te][feats_])[:, 1]
    record(name, 'within-site 5-fold x5', site, site, y, P.mean(0), len(feats_))

all_sites = sorted(data.site.unique())
adc_sites = sorted(data.loc[data.nadc_enhancing_median.notna(), 'site'].unique()) if 'nadc_enhancing_median' in data else []
perf_sites = sorted(data.loc[data.nperf_enhancing_p90.notna(), 'site'].unique()) if 'nperf_enhancing_p90' in data else []

loso('M1 conventional', CONV, all_sites)
if len(adc_sites) >= 2:
    loso('M2 conventional + diffusion', CONV + DIFF, adc_sites)
    loso('M2b conventional only (same patients as M2)', CONV, adc_sites, require=DIFF)
for s in perf_sites:
    have_diff = s in adc_sites
    base = CONV + (DIFF if have_diff else [])
    within('M3a conventional' + (' + diffusion' if have_diff else ''), base, s)
    within('M3b ... + perfusion', base + PERF, s)
if data.age.notna().sum() > 30:
    loso('Age only', ['age'], all_sites)
    loso('M1 + age', CONV + ['age'], all_sites)

mres = pd.DataFrame(results)
mres.to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
display(mres.round(3))
''')

md(r'''
## 9. Optional external test of the conventional model

Trains M1 on all development sites and applies it to each cohort in `EXTERNAL_COHORTS`, for example OpenNeuro ds007045:
363 primary GBM patients, MGMT by methylation-specific PCR, conventional MRI with tumour masks. Only conventional
features are used. External features are z-scored within the external cohort, the same way as each development site.
''')
code(r'''
ext_rows = []
for cfg in EXTERNAL_COHORTS:
    e = pd.read_csv(cfg['manifest'], dtype=str, keep_default_na=False)
    cm = cfg.get('column_map', {})
    edf = pd.DataFrame(dict(patient_id=e[cm.get('patient_id', 'patient_id')], site=cfg['name'].upper(),
                            mgmt=e[cm.get('mgmt', 'mgmt')].map(binary),
                            t1c_path=e[cm.get('t1c', 't1c_path')], mask_path=e[cm.get('mask', 'tumor_mask_path')],
                            brain_mask_path=e[cm['brain_mask']] if cm.get('brain_mask') else '',
                            adc_path='', fa_path='', perf_path='', adc_kind='', fa_kind='', perf_kind=''))
    if cfg.get('idh_wildtype_only') and cm.get('idh'):
        edf = edf[e[cm['idh']].map(binary).values == 0]
    edf = edf[edf.mgmt.isin([0, 1])]
    ef = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(extract)(SimpleNamespace(**r._asdict())) for r in edf.itertuples()))
    ef['error'] = ef['error'].fillna('')
    ef = ef[ef.error == ''].merge(edf[['patient_id', 'mgmt']], on='patient_id').dropna(subset=CONV)
    dev = site_z(data.dropna(subset=CONV), CONV); ext = site_z(ef, CONV)
    clf = model().fit(dev[CONV], dev.mgmt); p = clf.predict_proba(ext[CONV])[:, 1]
    record('M1 conventional', 'external', '+'.join(all_sites), cfg['name'], ext.mgmt.values.astype(int), p, len(CONV))
    ef.to_csv(OUTPUT_DIR / f"features_external_{cfg['name']}.csv", index=False)
if EXTERNAL_COHORTS:
    mres = pd.DataFrame(results); mres.to_csv(OUTPUT_DIR / 'model_results.csv', index=False); display(mres.round(3))
else:
    print('No external cohorts configured.')
''')

md('## 10. Report')
code(r'''
def to_md(frame):
    frame = frame.copy()
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    sep = '|' + '---|' * len(frame.columns)
    cell = lambda v: '' if pd.isna(v) else (str(int(v)) if isinstance(v, float) and float(v).is_integer() and abs(v) > 1
                                           else f'{v:.3f}' if isinstance(v, float) else str(v))
    body = ['| ' + ' | '.join(cell(v) for v in row) + ' |'
            for row in frame.itertuples(index=False)]
    return '\n'.join([head, sep] + body)

def fmt(r):
    return f"{r.oriented_auroc:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f})"
lines = ['# MGMT imaging correlates: results', '',
         f'Run: {time.strftime("%Y-%m-%d %H:%M")} | manifest: `{MANIFEST}`', '',
         '## Cohort', '', to_md(data.groupby('site').agg(patients=('patient_id', 'size'), methylated=('mgmt', 'sum')).reset_index()), '',
         '## Pre-specified hypotheses (pooled within-site AUROC in the hypothesised direction)', '',
         '| ID | Feature | Expected in methylated | Sites | AUROC (95% CI) | p (pre-specified) | Holm p | p (two-sided) | Verdict |', '|---|---|---|---|---|---|---|---|---|']
for h in HYPOTHESES:
    r = res[(res.id == h['id']) & (res.cohort == 'Pooled (within-site)')]
    if r.empty: continue
    r = r.iloc[0]
    lines.append(f"| {h['id']} | {h['label']} | {h['direction']} | {r.get('sites', '')} | {fmt(r)} | {r.p:.4f} | "
                 f"{'' if pd.isna(r.p_holm) else f'{r.p_holm:.4f}'} | {r.p_two_sided:.4f} | {r.verdict} |")
lines += ['', 'AUROC < 0.5 means the effect runs opposite to the published direction; the two-sided p tests a difference either way.']
lines += ['', '## Models (success bar: AUROC >= %.2f)' % SUCCESS_AUROC, '', to_md(mres), '',
          '## Caveats', '',
          '- Masks have mixed provenance (manual, corrected, automated); sub-region volumes inherit their errors.',
          '- Normalised maps divide by contralateral normal-appearing brain; this removes units, not protocol differences.',
          '- UCSF perfusion is ASL CBF and UPENN is DSC rCBV: same hypothesis, different measurements.',
          '- Exploratory hypotheses (E*) and models M3 are not corrected for multiplicity.']
(OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
''')

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
out = sys.argv[1] if len(sys.argv) > 1 else 'MGMT_Imaging_Correlates.ipynb'
nbf.write(nb, out)
print('Wrote', out)
