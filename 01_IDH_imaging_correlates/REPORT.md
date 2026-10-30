# IDH imaging correlates: results

Run: 2026-10-21 18:14 | manifest: `c:\Users\claks\OneDrive\文件\ChatGPT\RadGenome\outputs\patient_manifest.csv`

## Cohort

| site | grade_group | patients | idh_mutant |
|---|---|---|---|
| UCSF | 2-3 | 99 | 75 |
| UCSF | 4 | 396 | 28 |
| UPENN | 4 | 508 | 15 |
| UTSW | 2-3 | 169 | 133 |
| UTSW | 4 | 411 | 27 |
| UTSW | unknown | 4 | 0.000 |

## Pre-specified hypotheses: primary analysis (within site)

| ID | Feature | Expected in IDH-mutant | Strata | AUROC (95% CI) | p (pre-specified) | Holm p | p (two-sided) | Verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | Enhancing fraction of lesion | lower | UCSF UPENN UTSW | 0.85 (0.83-0.87) | 0.0001 | 0.0004 | 0.0001 | supported |
| H2 | Whole-lesion ADC (normalised) | higher | UCSF UPENN | 0.77 (0.72-0.81) | 0.0001 | 0.0004 | 0.0001 | supported |
| H3 | Whole-lesion perfusion p90 (normalised) | lower | UCSF UPENN | 0.75 (0.69-0.80) | 0.0001 | 0.0004 | 0.0001 | supported |
| H4 | T2-FLAIR mismatch index | higher | UCSF UPENN UTSW | 0.65 (0.62-0.69) | 0.0001 | 0.0004 | 0.0001 | supported |
| E1 | Age (clinical reference) | lower | UCSF UPENN UTSW | 0.89 (0.87-0.91) | 0.0001 |  | 0.0001 |  |
| E2 | Necrosis/non-enh. core fraction of core | lower | UCSF UPENN UTSW | 0.31 (0.26-0.35) | 1.0000 |  | 0.0001 |  |
| E3 | Whole-lesion volume (log mL) | two-sided | UCSF UPENN UTSW | 0.49 (0.46-0.54) | 0.7722 |  | 0.7834 |  |
| E4 | Minimum ADC (non-enh. p10, normalised) | higher | UCSF UPENN | 0.81 (0.77-0.86) | 0.0001 |  | 0.0001 |  |
| E5 | FLAIR-suppressed voxel fraction | higher | UCSF UPENN UTSW | 0.60 (0.57-0.65) | 0.0001 |  | 0.0001 |  |
| E6 | Left hemisphere | two-sided | UCSF UPENN UTSW | 0.51 (0.48-0.55) | 0.4533 |  | 0.4366 |  |

## Sensitivity analysis (within site x grade group)

| ID | Feature | Expected in IDH-mutant | Strata | AUROC (95% CI) | p (pre-specified) | Holm p | p (two-sided) | Verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | Enhancing fraction of lesion | lower | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.71 (0.66-0.75) | 0.0001 | 0.0004 | 0.0001 | supported |
| H2 | Whole-lesion ADC (normalised) | higher | UCSF / 2-3 UCSF / 4 UPENN / 4 | 0.74 (0.67-0.81) | 0.0001 | 0.0004 | 0.0001 | supported |
| H3 | Whole-lesion perfusion p90 (normalised) | lower | UCSF / 2-3 UCSF / 4 UPENN / 4 | 0.68 (0.58-0.77) | 0.0001 | 0.0004 | 0.0002 | supported |
| H4 | T2-FLAIR mismatch index | higher | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.69 (0.63-0.74) | 0.0001 | 0.0004 | 0.0001 | supported |
| E1 | Age (clinical reference) | lower | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.90 (0.87-0.92) | 0.0001 |  | 0.0001 |  |
| E2 | Necrosis/non-enh. core fraction of core | lower | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.40 (0.33-0.46) | 0.9996 |  | 0.0015 |  |
| E3 | Whole-lesion volume (log mL) | two-sided | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.67 (0.62-0.72) | 0.0001 |  | 0.0001 |  |
| E4 | Minimum ADC (non-enh. p10, normalised) | higher | UCSF / 2-3 UCSF / 4 UPENN / 4 | 0.72 (0.63-0.79) | 0.0001 |  | 0.0001 |  |
| E5 | FLAIR-suppressed voxel fraction | higher | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.70 (0.64-0.76) | 0.0001 |  | 0.0001 |  |
| E6 | Left hemisphere | two-sided | UCSF / 2-3 UCSF / 4 UPENN / 4 UTSW / 2-3 UTSW / 4 | 0.51 (0.46-0.56) | 0.7197 |  | 0.7153 |  |

AUROC < 0.5 means the effect runs opposite to the published direction.

## Models (success bar: AUROC >= 0.80)

| model | setting | train | test | n_test | idh_mutant | features | auroc | ci_low | ci_high | meets_bar |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 conventional | cross-site | UPENN+UTSW | UCSF | 495 | 103 | 6 | 0.888 | 0.847 | 0.926 | True |
| M1 conventional | cross-site | UCSF+UTSW | UPENN | 508 | 15 | 6 | 0.736 | 0.605 | 0.856 | False |
| M1 conventional | cross-site | UCSF+UPENN | UTSW | 584 | 160 | 6 | 0.840 | 0.806 | 0.873 | True |
| M1m conventional + mismatch | cross-site | UPENN+UTSW | UCSF | 494 | 103 | 8 | 0.889 | 0.849 | 0.924 | True |
| M1m conventional + mismatch | cross-site | UCSF+UTSW | UPENN | 508 | 15 | 8 | 0.799 | 0.701 | 0.892 | False |
| M1m conventional + mismatch | cross-site | UCSF+UPENN | UTSW | 581 | 160 | 8 | 0.842 | 0.805 | 0.875 | True |
| M2 conventional + diffusion | cross-site | UPENN | UCSF | 493 | 103 | 8 | 0.865 | 0.823 | 0.901 | True |
| M2 conventional + diffusion | cross-site | UCSF | UPENN | 442 | 11 | 8 | 0.754 | 0.594 | 0.893 | False |
| M2b conventional only (same patients as M2) | cross-site | UPENN | UCSF | 493 | 103 | 6 | 0.807 | 0.755 | 0.852 | True |
| M2b conventional only (same patients as M2) | cross-site | UCSF | UPENN | 442 | 11 | 6 | 0.744 | 0.610 | 0.865 | False |
| M3a conventional + diffusion | within-site 5-fold x5 | UCSF | UCSF | 493 | 103 | 8 | 0.905 | 0.870 | 0.937 | True |
| M3b ... + perfusion | within-site 5-fold x5 | UCSF | UCSF | 493 | 103 | 10 | 0.903 | 0.866 | 0.935 | True |
| M3a conventional + diffusion | within-site 5-fold x5 | UPENN | UPENN | 442 | 11 | 8 | 0.671 | 0.492 | 0.844 | False |
| Age only | cross-site | UPENN+UTSW | UCSF | 495 | 103 | 1 | 0.905 | 0.871 | 0.936 | True |
| Age only | cross-site | UCSF+UTSW | UPENN | 508 | 15 | 1 | 0.917 | 0.853 | 0.969 | True |
| Age only | cross-site | UCSF+UPENN | UTSW | 584 | 160 | 1 | 0.882 | 0.852 | 0.910 | True |
| M1 + mismatch + age | cross-site | UPENN+UTSW | UCSF | 494 | 103 | 9 | 0.954 | 0.933 | 0.972 | True |
| M1 + mismatch + age | cross-site | UCSF+UTSW | UPENN | 508 | 15 | 9 | 0.926 | 0.863 | 0.974 | True |
| M1 + mismatch + age | cross-site | UCSF+UPENN | UTSW | 581 | 160 | 9 | 0.919 | 0.894 | 0.940 | True |

## RadGenome feature audit

### Head-to-head on identical patients

| model | auroc | ci_low | ci_high | n | idh_mutant |
|---|---|---|---|---|---|
| RadGenome ensemble (OOF) | 0.926 |  |  | 1396 | 186 |
| Features + age (site-stratified CV) | 0.936 |  |  | 1396 | 186 |
| Features + age + ensemble logit (CV) | 0.960 |  |  | 1396 | 186 |
| Delta: ensemble - features | -0.011 | -0.029 | 0.008 | 1396 | 186 |
| Delta: combined - ensemble | 0.034 | 0.021 | 0.048 | 1396 | 186 |

### Spearman correlation of the ensemble logit with each feature (pooled within site)

| feature | rho_all | rho_within_wildtype | rho_within_mutant | n |
|---|---|---|---|---|
| enhancing_frac_whole | -0.567 | -0.439 | -0.455 | 1396 |
| edema_ratio_log | 0.415 | 0.343 | 0.309 | 1396 |
| nperf_whole_p90 | -0.278 | -0.180 | -0.206 | 784 |
| nflair_nonenh_median | -0.072 | -0.158 | 0.010 | 1393 |
| nfa_whole_median | -0.277 | -0.144 | -0.137 | 823 |
| log_vol_whole | -0.111 | -0.123 | -0.149 | 1396 |
| nadc_nonenh_p10 | 0.251 | 0.122 | 0.094 | 821 |
| nt2_nonenh_median | 0.047 | -0.101 | 0.097 | 1393 |
| age | -0.285 | -0.078 | -0.176 | 1396 |
| mismatch_index | 0.131 | 0.060 | 0.092 | 1393 |
| nadc_whole_median | 0.177 | 0.053 | 0.093 | 822 |
| mismatch_frac | 0.028 | -0.033 | 0.091 | 1393 |
| necrosis_frac | 0.003 | 0.006 | 0.008 | 1396 |

## Caveats

- Masks have mixed provenance; label 1 means necrosis in glioblastoma but non-enhancing tumour in many lower-grade gliomas.
- The T2-FLAIR mismatch index is an intensity surrogate of a visual sign; FLAIR/T2 protocols differ by site.
- UPENN IDH status is IDH1 only and the cohort is glioblastoma-only, so it has few IDH-mutant cases.
- Exploratory hypotheses (E*), model M3 and the audit correlations are not corrected for multiplicity.