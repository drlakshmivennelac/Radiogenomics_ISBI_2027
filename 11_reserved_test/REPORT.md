# Reserved internal test (one-time evaluation of the frozen models)

Run: 2026-10-22 17:31. Protocol written: 2026-10-22 17:26. frozen_models.json sha256 79f0b24386ef.

Patients: 200 reserved; 190 analysed (92 IDH-mutant). Excluded (no features/age): 10 ({('UPENN', 0.0): 1, ('UTSW', 0.0): 1, ('UTSW', 1.0): 8}).

Case-control design (about 50% prevalence): AUROC, sensitivity and specificity transfer; PPV/NPV are re-weighted to the development prevalence (0.134); Brier is shown for the case-control mix only.

## Inference reproduction check

               member  patients  max_abs_diff  mean_abs_diff
  v2_reference_seed42         8        0.0001            0.0
  v2_reference_seed44         8        0.0000            0.0
v2_mean_concat_seed42         8        0.0000            0.0

## All models (frozen thresholds)

| model | n | idh_mutant | auroc | auroc_ci | sensitivity@youden | sens_ci@youden | specificity@youden | spec_ci@youden | ppv_dev_prev@youden | npv_dev_prev@youden | calibration_slope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Age | 190 | 92 | 0.887 | 0.835-0.933 | 0.837 | 0.745-0.906 | 0.806 | 0.714-0.879 | 0.399 | 0.970 | 1.014 |
| Features | 190 | 92 | 0.870 | 0.818-0.918 | 0.696 | 0.591-0.787 | 0.898 | 0.820-0.950 | 0.512 | 0.950 | 1.004 |
| Features + age | 190 | 92 | 0.943 | 0.910-0.971 | 0.913 | 0.836-0.962 | 0.847 | 0.760-0.912 | 0.479 | 0.984 | 1.060 |
| Ensemble (raw) | 190 | 92 | 0.879 | 0.828-0.922 | 0.761 | 0.661-0.844 | 0.867 | 0.784-0.927 | 0.469 | 0.959 | 0.973 |
| Ensemble (recalibrated) | 190 | 92 | 0.879 | 0.828-0.922 | 0.761 | 0.661-0.844 | 0.867 | 0.784-0.927 | 0.469 | 0.959 | 0.903 |
| Ensemble + age | 190 | 92 | 0.943 | 0.911-0.969 | 0.880 | 0.796-0.939 | 0.816 | 0.725-0.887 | 0.425 | 0.978 | 1.024 |
| Ensemble + features | 190 | 92 | 0.915 | 0.874-0.953 | 0.793 | 0.696-0.871 | 0.867 | 0.784-0.927 | 0.480 | 0.965 | 1.016 |
| Hybrid | 190 | 92 | 0.953 | 0.925-0.975 | 0.880 | 0.796-0.939 | 0.878 | 0.796-0.935 | 0.526 | 0.979 | 1.033 |
| Confidence gate (RSNA) | 190 | 92 | 0.944 | 0.912-0.969 | 0.815 | 0.721-0.889 | 0.898 | 0.820-0.950 | 0.552 | 0.969 | 1.392 |

## Paired AUROC differences (site x IDH stratified bootstrap)

| comparison | delta_auroc | ci | p_le_0 |
|---|---|---|---|
| Hybrid minus Age | 0.066 | 0.026-0.106 | 0.000 |
| Hybrid minus Features | 0.083 | 0.043-0.127 | 0.000 |
| Hybrid minus Features + age | 0.010 | -0.008-0.028 | 0.137 |
| Hybrid minus Ensemble (raw) | 0.074 | 0.038-0.117 | 0.000 |
| Hybrid minus Ensemble (recalibrated) | 0.074 | 0.038-0.117 | 0.000 |
| Hybrid minus Ensemble + age | 0.010 | -0.003-0.023 | 0.058 |
| Hybrid minus Ensemble + features | 0.038 | 0.008-0.071 | 0.005 |
| Hybrid minus Confidence gate (RSNA) | 0.009 | -0.002-0.022 | 0.052 |
| Ensemble + age minus Ensemble (recalibrated) | 0.064 | 0.031-0.103 | 0.000 |
| Confidence gate (RSNA) minus Ensemble (recalibrated) | 0.065 | 0.033-0.103 | 0.000 |

Ensemble (raw) on all 200 labelled reserved patients: AUROC 0.888.

## Subgroups (descriptive)

| subgroup | model | n | idh_mutant | auroc | auroc_ci |
|---|---|---|---|---|---|
| grade 4 | Age | 127 | 31 | 0.900 | 0.844-0.948 |
| grade 4 | Features + age | 127 | 31 | 0.933 | 0.889-0.973 |
| grade 4 | Ensemble (recalibrated) | 127 | 31 | 0.834 | 0.740-0.913 |
| grade 4 | Ensemble + age | 127 | 31 | 0.943 | 0.901-0.976 |
| grade 4 | Hybrid | 127 | 31 | 0.952 | 0.916-0.982 |
| grade 4 | Confidence gate (RSNA) | 127 | 31 | 0.927 | 0.876-0.965 |
| site UCSF | Age | 61 | 35 | 0.916 | 0.830-0.982 |
| site UCSF | Features + age | 61 | 35 | 0.969 | 0.922-0.998 |
| site UCSF | Ensemble (recalibrated) | 61 | 35 | 0.921 | 0.846-0.977 |
| site UCSF | Ensemble + age | 61 | 35 | 0.969 | 0.927-0.996 |
| site UCSF | Hybrid | 61 | 35 | 0.973 | 0.933-0.997 |
| site UCSF | Confidence gate (RSNA) | 61 | 35 | 0.968 | 0.925-0.996 |
| site UPENN | Age | 62 | 5 | 0.821 | 0.667-0.940 |
| site UPENN | Features + age | 62 | 5 | 0.895 | 0.765-0.993 |
| site UPENN | Ensemble (recalibrated) | 62 | 5 | 0.821 | 0.561-0.986 |
| site UPENN | Ensemble + age | 62 | 5 | 0.923 | 0.842-0.986 |
| site UPENN | Hybrid | 62 | 5 | 0.933 | 0.846-0.989 |
| site UPENN | Confidence gate (RSNA) | 62 | 5 | 0.916 | 0.818-0.986 |
| site UTSW | Age | 67 | 52 | 0.824 | 0.633-0.971 |
| site UTSW | Features + age | 67 | 52 | 0.863 | 0.706-0.972 |
| site UTSW | Ensemble (recalibrated) | 67 | 52 | 0.846 | 0.718-0.949 |
| site UTSW | Ensemble + age | 67 | 52 | 0.910 | 0.812-0.977 |
| site UTSW | Hybrid | 67 | 52 | 0.885 | 0.758-0.973 |
| site UTSW | Confidence gate (RSNA) | 67 | 52 | 0.877 | 0.749-0.969 |

## Earlier evaluations on the reserved set

                                                   folder  idh_auroc       date
Downloads\outputs\cv_20260919_050926_120849\balanced_test     0.8617 2026-09-19

- None of the earlier models is part of the paper; disclose them anyway.
- Ensemble members, ensemble weights, combiners and thresholds were all fixed on development data before this run.