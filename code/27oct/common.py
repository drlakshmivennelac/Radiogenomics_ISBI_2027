"""Shared setup for the 27 Oct CPU re-analyses.

Rebuilds the paper's 1,393-patient analysis frame (Z) from files inside the v6 package, then executes the helper
cell of IDH_Hybrid_Analysis.ipynb verbatim (run_cv, bootstrap, metrics), so every number below is produced by the
paper's own code path. No images, no GPU.

Paths (override with environment variables):
  RG_PKG     the unzipped RadGenome_ISBI_package_v6 folder        default ~/Downloads/RadGenome_ISBI_package_v6
  RG_DL_OUT  the Downloads/outputs folder (member runs, v3 runs)  default ~/Downloads/outputs
  RG_RG_OUT  the RadGenome project's outputs folder               default <this folder's parent>/outputs
"""
import os, sys, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import nbformat

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get('RG_PKG', Path.home() / 'Downloads' / 'RadGenome_ISBI_package_v6'))
DL_OUT = Path(os.environ.get('RG_DL_OUT', Path.home() / 'Downloads' / 'outputs'))
RG_OUT = Path(os.environ.get('RG_RG_OUT', HERE.parent / 'outputs'))
RES = PKG / 'results'
BUILD = HERE / '_built_notebooks'


def built(name):
    """Notebook generated from the package's own builder script (so the executed code is the paper's)."""
    import subprocess
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f'{name}.ipynb'
    if not out.exists():
        subprocess.run([sys.executable, str(PKG / 'code' / f'build_{name}_notebook.py'), str(out)], check=True,
                       stdout=subprocess.DEVNULL, cwd=str(PKG / 'code'))
    return out
warnings.filterwarnings('ignore')


def display(*a, **k):
    pass


def _cell_after(nb, prefix):
    for i, c in enumerate(nb.cells):
        if c.cell_type == 'markdown' and c.source.lstrip().startswith(prefix):
            return nb.cells[i + 1].source
    raise KeyError(prefix)


def build():
    """Return the namespace of the hybrid notebook's helper cell with Z, d rebuilt from package files."""
    nb = nbformat.read(built('hybrid'), 4)
    g = {'display': display}
    cfg = _cell_after(nb, '## 1.')
    exec(cfg, g)                                   # constants: N_FOLDS, REPEATS, N_BOOT, SEED, C_REG, SENS_TARGET
    exec('import json, re, time, warnings, platform\nimport numpy as np, pandas as pd\n'
         'from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n'
         'from sklearn.pipeline import make_pipeline\nfrom sklearn.model_selection import StratifiedKFold\n'
         'from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve\n', g)
    # ---- cohort rebuilt from package files, in the order of the paper's cv_predictions.csv
    cvp = pd.read_csv(RES / '03_what_the_ensemble_adds' / 'cv_predictions.csv')
    gate = pd.read_csv(RES / '09_confidence_gating' / 'gating_cv_predictions.csv')[['patient_id', 'grade_group']]
    oof = pd.read_csv(RES / '06_ensemble_development' / 'internal_OOF_three_member' / 'patient_predictions.csv')
    feats = pd.read_csv(RES / '01_IDH_imaging_correlates' / 'features.csv')
    feats = feats[feats.error.isna() | (feats.error.astype(str).str.len() == 0)].drop(columns=['site'])
    d = cvp[['patient_id', 'site', 'y', 'age']].merge(gate, on='patient_id', how='left')
    d = d.merge(oof[['patient_id', 'probability', 'fold']], on='patient_id').merge(feats, on='patient_id')
    mask_tab = RG_OUT / 'v2_mask_content_audit.csv'
    if mask_tab.exists():
        mt = pd.read_csv(mask_tab, dtype=str, keep_default_na=False)[['patient_id', 'source']]
        d = d.merge(mt.rename(columns={'source': 'mask_source'}), on='patient_id', how='left')
    else:
        d['mask_source'] = 'unknown'
    CONV = ['log_vol_whole', 'log_vol_enhancing', 'enhancing_frac_whole', 'no_enhancement', 'necrosis_frac', 'edema_ratio_log']
    MISM = ['mismatch_index', 'mismatch_frac']
    d['necrosis_frac'] = d.necrosis_frac.fillna(0.0)
    d = d.dropna(subset=CONV + MISM).reset_index(drop=True)
    assert len(d) == 1393 and list(d.patient_id) == list(cvp.patient_id), 'cohort reconstruction failed'
    d['y'] = d.y.astype(int)
    p_ = d.probability.clip(1e-6, 1 - 1e-6); d['dl_logit'] = np.log(p_ / (1 - p_))
    g['d'] = d; g['CONV'] = CONV; g['MISM'] = MISM; g['ADV'] = ['nadc_whole_median', 'nadc_nonenh_p10', 'nperf_whole_p90']
    exec(_cell_after(nb, '## 3.'), g)              # site_z, Z, MODELS, run_cv, bootstrap, all_metrics, ...
    g['CVP'] = cvp
    return g


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def md_table(df, nd=3):
    cols = list(df.columns)
    f = lambda v: (f'{v:.{nd}f}' if isinstance(v, (float, np.floating)) and np.isfinite(v) else ('' if isinstance(v, float) else str(v)))
    return '\n'.join(['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(f(v) for v in r) + ' |' for r in df.itertuples(index=False)])
