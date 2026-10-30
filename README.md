# RadGenome: MRI-based IDH prediction, ISBI 2027

**Paper:** *Benchmarking a hybrid model for MRI-based IDH prediction against deep learning, radiological features and age.*

**Authors:** Lakshmi Vennela Chowdary Kaza, Viswanathan Murugaiah, Pallavi Shyla Revikumar (CalibRAItion Health, Bengaluru, India).

**Status:** final manuscript, v13 (`manuscript/RadGenome_ISBI_v13.pdf`, 4 pages including references).

---

## Contents

| Folder | Contents |
|---|---|
| `manuscript/RadGenome_ISBI_v13.pdf` | The final manuscript. |
| `results/` | Result tables, figures, reports and frozen-model outputs, in folders `01` to `15`. |
| `code/` | Analysis scripts. `build_*.py` generate the notebooks; `27oct/` and `29oct/` hold later analyses. |
| `notebooks/` | Analysis notebooks, numbered 1 to 8 in the order of the paper. |

### Results folders

| Folder | Contents |
|---|---|
| `01_IDH_imaging_correlates` | Imaging correlates of IDH status. |
| `02_hybrid_model` | Hybrid model: primary metrics, paired differences, decision curves, calibration, leave-one-site-out. |
| `03_what_the_ensemble_adds` | What the DL ensemble adds beyond features and age. |
| `04_external_MU_RHUH` | External validation, MU-Glioma-Post and RHUH. |
| `05_feature_audit_sample` | Feature audit sample. |
| `06_ensemble_development` | Ensemble development. |
| `07_ensemble_only_external_earlier` | Earlier ensemble-only external results. |
| `08_supplementary_MGMT_negative` | Supplementary MGMT analysis. |
| `09_confidence_gating` | Confidence-gated predictions and risk-coverage. |
| `10_T2_network_and_full_LOSO` | T2 network and full leave-one-site-out. |
| `11_reserved_test` | Reserved test set: predictions, metrics and subgroups. |
| `12_cpu_reanalyses_27oct` | CPU re-analyses (R0 to R12). |
| `13_held_out_and_external_27oct` | Occlusion analysis, ds007045 external validation, and leave-site-out retraining. |
| `14_v8_recomputations_29oct` | Recomputed statistics: age-matched AUROC, subgroup CIs, McNemar tests, calibration slopes, unseen-site checks. `recompute_v8.py` in `code/29oct/` regenerates it. |
| `15_e1b_shape_control_29oct` | Shape control for the occlusion analysis. |

Most results folders have a `REPORT.md` describing their contents and numbers. Folders `05`, `06` and `07` do not.

---

## Headline numbers

AUROC (95% CI where reported) from Table 1 of the v13 manuscript. The last column is specificity at Youden / 90% development sensitivity.

| Setting (n / IDH-mutant) | Ensemble | Features + age | Hybrid |
|---|---|---|---|
| Pooled cross-validation, all sites seen (1,393 / 186) | 0.926 | 0.940 | 0.961 |
| Age-matched, within matched sets (358 / 96) | 0.908 | 0.859 | 0.924 |
| Reserved patients, frozen models (190 / 92) | 0.879 | 0.943 | 0.953 |
| Reserved, grade 4 only (127 / 31) | 0.834 | 0.933 | 0.952 |
| UCSF never seen in training, 3 networks (433 / 68) | 0.916 | 0.952 | 0.964 |
| UTSW never seen in training, 1 network (514 / 108) | 0.826 | 0.918 | 0.931 |
| ds007045 external IDH-wildtype GBM, specificity (359 / 0) | 90.8 / 79.9% | 87.7 / 78.6% | 95.3 / 88.9% |

---

## Notes

- **Patient-level data.** The `results/` CSVs contain patient-level rows (identifiers, age, grade, site, and per-patient predictions). Check the data-use terms of each source cohort before making this repository public.
- **Earlier versions.** Earlier manuscript versions, change logs and the reader brief for the radiologist audit have been removed from the working tree. The audit was dropped from the ISBI version and is planned for the journal version. All of these remain in the git history.
