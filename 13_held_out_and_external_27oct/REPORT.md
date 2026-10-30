# ISBI GPU experiments (27 Oct 2026)

Run 2026-10-28 05:27 on NVIDIA GeForce RTX 5080 Laptop GPU.

E1 info: {"analysis_patients": 1393, "excluded_no_prediction": 0, "excluded_no_lesion": 0, "max_abs_original_minus_saved_oof": 0.00017638814946008274, "matched_sets": 96, "matched_patients": 358}

## E1. Occlusion: ensemble AUROC by input condition

| condition | n | mutant | auroc | auroc_ci | auroc_age_matched | auroc_age_matched_ci | rho_age_wildtype | rho_age_wildtype_ci | rho_age_mutant | rho_age_mutant_ci |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1393 | 186 | 0.926 | 0.904-0.945 | 0.900 | 0.862-0.930 | -0.088 | -0.14 to -0.03 | -0.178 | -0.32 to -0.04 |
| tumour_ring_only | 1393 | 186 | 0.859 | 0.827-0.888 | 0.838 | 0.790-0.881 | -0.044 | -0.10 to +0.01 | -0.231 | -0.36 to -0.09 |
| lesion_only | 1393 | 186 | 0.738 | 0.695-0.782 | 0.717 | 0.655-0.780 | -0.021 | -0.08 to +0.03 | -0.273 | -0.41 to -0.13 |
| brain_only | 1393 | 186 | 0.677 | 0.636-0.716 | 0.668 | 0.611-0.725 | -0.051 | -0.11 to -0.00 | -0.015 | -0.16 to +0.13 |
| t1c_blank | 1393 | 186 | 0.580 | 0.539-0.623 | 0.586 | 0.530-0.640 | 0.017 | -0.04 to +0.07 | -0.024 | -0.17 to +0.13 |
| flair_blank | 1393 | 186 | 0.913 | 0.890-0.934 | 0.891 | 0.856-0.922 | -0.102 | -0.16 to -0.05 | -0.149 | -0.29 to -0.01 |
| tumour_ring_only_blurred_brain | 1393 | 186 | 0.885 | 0.856-0.913 | 0.858 | 0.808-0.900 | -0.030 | -0.08 to +0.02 | -0.202 | -0.33 to -0.07 |
| brain_only_blurred_tumour | 1393 | 186 | 0.841 | 0.809-0.867 | 0.844 | 0.802-0.885 | -0.113 | -0.17 to -0.06 | -0.077 | -0.22 to +0.07 |

## E1. Paired AUROC difference from the original input (site x class bootstrap; age-matched: cluster bootstrap)

| condition | delta_auroc | ci | delta_auroc_age_matched | ci_age_matched |
|---|---|---|---|---|
| tumour_ring_only | -0.067 | -0.090 to -0.045 | -0.062 | -0.098 to -0.031 |
| lesion_only | -0.188 | -0.229 to -0.148 | -0.183 | -0.247 to -0.122 |
| brain_only | -0.249 | -0.289 to -0.211 | -0.232 | -0.288 to -0.174 |
| t1c_blank | -0.346 | -0.387 to -0.301 | -0.314 | -0.371 to -0.258 |
| flair_blank | -0.013 | -0.025 to -0.002 | -0.009 | -0.026 to +0.006 |
| tumour_ring_only_blurred_brain | -0.041 | -0.058 to -0.024 | -0.042 | -0.071 to -0.016 |
| brain_only_blurred_tumour | -0.086 | -0.105 to -0.066 | -0.056 | -0.087 to -0.025 |

## E1. Models refitted with the occluded ensemble logit (5x5 CV, AUROC)

| condition | Ensemble + features | Ensemble + age | Hybrid |
|---|---|---|---|
| original | 0.933 | 0.955 | 0.961 |
| tumour_ring_only | 0.905 | 0.931 | 0.948 |
| brain_only | 0.903 | 0.904 | 0.949 |
| t1c_blank | 0.884 | 0.892 | 0.940 |
| flair_blank | 0.928 | 0.951 | 0.959 |

## E1. Per-member AUROC by condition

| member | original | tumour_ring_only | lesion_only | brain_only | t1c_blank | flair_blank | tumour_ring_only_blurred_brain | brain_only_blurred_tumour |
|---|---|---|---|---|---|---|---|---|
| v2_mean_concat_seed42 | 0.892 | 0.825 | 0.734 | 0.629 | 0.538 | 0.872 | 0.842 | 0.797 |
| v2_reference_seed42 | 0.897 | 0.804 | 0.657 | 0.658 | 0.604 | 0.871 | 0.830 | 0.798 |
| v2_reference_seed44 | 0.879 | 0.764 | 0.639 | 0.654 | 0.553 | 0.874 | 0.810 | 0.812 |

## E2. ds007045: specificity at frozen thresholds (exact binomial 95% CI)

| group | model | operating_point | threshold | n | false_positives | specificity | spec_ci | mean_predicted_p_mutant | median_predicted_p_mutant |
|---|---|---|---|---|---|---|---|---|---|
| all | Age | youden | 0.185 | 359 | 44 | 0.877 | 0.839-0.910 | 0.087 | 0.038 |
| all | Age | sens90 | 0.059 | 359 | 140 | 0.610 | 0.557-0.661 | 0.087 | 0.038 |
| all | Features | youden | 0.161 | 359 | 52 | 0.855 | 0.814-0.890 | 0.086 | 0.048 |
| all | Features | sens90 | 0.054 | 359 | 160 | 0.554 | 0.501-0.606 | 0.086 | 0.048 |
| all | Features + age | youden | 0.134 | 359 | 44 | 0.877 | 0.839-0.910 | 0.065 | 0.016 |
| all | Features + age | sens90 | 0.072 | 359 | 77 | 0.786 | 0.739-0.827 | 0.065 | 0.016 |
| all | Ensemble (raw) | youden | 0.132 | 359 | 33 | 0.908 | 0.873-0.936 | 0.066 | 0.031 |
| all | Ensemble (raw) | sens90 | 0.063 | 359 | 72 | 0.799 | 0.754-0.840 | 0.066 | 0.031 |
| all | Ensemble (recalibrated) | youden | 0.114 | 359 | 33 | 0.908 | 0.873-0.936 | 0.055 | 0.023 |
| all | Ensemble (recalibrated) | sens90 | 0.047 | 359 | 74 | 0.794 | 0.748-0.835 | 0.055 | 0.023 |
| all | Ensemble + age | youden | 0.108 | 359 | 33 | 0.908 | 0.873-0.936 | 0.044 | 0.011 |
| all | Ensemble + age | sens90 | 0.086 | 359 | 43 | 0.880 | 0.842-0.912 | 0.044 | 0.011 |
| all | Ensemble + features | youden | 0.116 | 359 | 23 | 0.936 | 0.905-0.959 | 0.044 | 0.019 |
| all | Ensemble + features | sens90 | 0.065 | 359 | 44 | 0.877 | 0.839-0.910 | 0.044 | 0.019 |
| all | Hybrid | youden | 0.151 | 359 | 17 | 0.953 | 0.925-0.972 | 0.037 | 0.009 |
| all | Hybrid | sens90 | 0.078 | 359 | 40 | 0.889 | 0.851-0.919 | 0.037 | 0.009 |
| all | Confidence gate (RSNA) | youden | 0.124 | 359 | 23 | 0.936 | 0.905-0.959 | 0.051 | 0.024 |
| all | Confidence gate (RSNA) | sens90 | 0.098 | 359 | 31 | 0.914 | 0.880-0.941 | 0.051 | 0.024 |

## E2. Specificity at the Youden threshold: development and reserved grade-4 IDH-wildtype vs ds007045

| model | threshold | development_grade4_wildtype | reserved_grade4_wildtype | ds007045 |
|---|---|---|---|---|
| Age | 0.185 | 0.894 (n=1146, OOF probabilities) | 0.812 (n=96) | 0.877 (n=359) |
| Features | 0.161 | 0.918 (n=1146, OOF probabilities) | 0.906 (n=96) | 0.855 (n=359) |
| Features + age | 0.134 | 0.909 (n=1146, OOF probabilities) | 0.865 (n=96) | 0.877 (n=359) |
| Ensemble (raw) | 0.132 | 0.884 (n=1146, OOF probabilities) | 0.875 (n=96) | 0.908 (n=359) |
| Ensemble + age | 0.108 | 0.909 (n=1146, OOF probabilities) | 0.823 (n=96) | 0.908 (n=359) |
| Ensemble + features | 0.116 | 0.908 (n=1146, OOF probabilities) | 0.875 (n=96) | 0.936 (n=359) |
| Hybrid | 0.151 | 0.940 (n=1146, OOF probabilities) | 0.885 (n=96) | 0.953 (n=359) |

## E2. Specificity by age band, field strength and centre (Youden threshold)

| group | model | operating_point | threshold | n | false_positives | specificity | spec_ci | mean_predicted_p_mutant | median_predicted_p_mutant |
|---|---|---|---|---|---|---|---|---|---|
| age_band: 40-54 | Features + age | youden | 0.134 | 82 | 28 | 0.659 | 0.546-0.760 | 0.142 | 0.079 |
| age_band: 40-54 | Ensemble (raw) | youden | 0.132 | 82 | 13 | 0.841 | 0.744-0.913 | 0.096 | 0.032 |
| age_band: 40-54 | Hybrid | youden | 0.151 | 82 | 11 | 0.866 | 0.773-0.931 | 0.087 | 0.033 |
| age_band: <40 | Features + age | youden | 0.134 | 13 | 12 | 0.077 | 0.002-0.360 | 0.442 | 0.416 |
| age_band: <40 | Ensemble (raw) | youden | 0.132 | 13 | 3 | 0.769 | 0.462-0.950 | 0.157 | 0.035 |
| age_band: <40 | Hybrid | youden | 0.151 | 13 | 6 | 0.538 | 0.251-0.808 | 0.267 | 0.145 |
| age_band: >=55 | Features + age | youden | 0.134 | 264 | 4 | 0.985 | 0.962-0.996 | 0.023 | 0.010 |
| age_band: >=55 | Ensemble (raw) | youden | 0.132 | 264 | 17 | 0.936 | 0.899-0.962 | 0.051 | 0.031 |
| age_band: >=55 | Hybrid | youden | 0.151 | 264 | 0 | 1.000 | 0.986-1.000 | 0.011 | 0.006 |
| field: 1 T | Features + age | youden | 0.134 | 1 | 1 | 0.000 | 0.000-0.975 | 0.635 | 0.635 |
| field: 1 T | Ensemble (raw) | youden | 0.132 | 1 | 0 | 1.000 | 0.025-1.000 | 0.063 | 0.063 |
| field: 1 T | Hybrid | youden | 0.151 | 1 | 0 | 1.000 | 0.025-1.000 | 0.128 | 0.128 |
| field: 1.5 T | Features + age | youden | 0.134 | 114 | 13 | 0.886 | 0.813-0.938 | 0.051 | 0.016 |
| field: 1.5 T | Ensemble (raw) | youden | 0.132 | 114 | 13 | 0.886 | 0.813-0.938 | 0.068 | 0.034 |
| field: 1.5 T | Hybrid | youden | 0.151 | 114 | 5 | 0.956 | 0.901-0.986 | 0.030 | 0.008 |
| field: 3 T | Features + age | youden | 0.134 | 244 | 30 | 0.877 | 0.829-0.915 | 0.069 | 0.016 |
| field: 3 T | Ensemble (raw) | youden | 0.132 | 244 | 20 | 0.918 | 0.876-0.949 | 0.065 | 0.030 |
| field: 3 T | Hybrid | youden | 0.151 | 244 | 12 | 0.951 | 0.916-0.974 | 0.041 | 0.010 |
| centre_group: BO | Features + age | youden | 0.134 | 171 | 28 | 0.836 | 0.772-0.888 | 0.083 | 0.025 |
| centre_group: BO | Ensemble (raw) | youden | 0.132 | 171 | 18 | 0.895 | 0.839-0.936 | 0.072 | 0.033 |
| centre_group: BO | Hybrid | youden | 0.151 | 171 | 11 | 0.936 | 0.888-0.967 | 0.045 | 0.011 |
| centre_group: KA | Features + age | youden | 0.134 | 107 | 7 | 0.935 | 0.870-0.973 | 0.048 | 0.013 |
| centre_group: KA | Ensemble (raw) | youden | 0.132 | 107 | 6 | 0.944 | 0.882-0.979 | 0.050 | 0.028 |
| centre_group: KA | Hybrid | youden | 0.151 | 107 | 3 | 0.972 | 0.920-0.994 | 0.026 | 0.007 |
| centre_group: NSK | Features + age | youden | 0.134 | 39 | 4 | 0.897 | 0.758-0.971 | 0.035 | 0.008 |
| centre_group: NSK | Ensemble (raw) | youden | 0.132 | 39 | 2 | 0.949 | 0.827-0.994 | 0.063 | 0.030 |
| centre_group: NSK | Hybrid | youden | 0.151 | 39 | 1 | 0.974 | 0.865-0.999 | 0.037 | 0.008 |
| centre_group: other (<20 each) | Features + age | youden | 0.134 | 42 | 5 | 0.881 | 0.744-0.960 | 0.065 | 0.017 |
| centre_group: other (<20 each) | Ensemble (raw) | youden | 0.132 | 42 | 7 | 0.833 | 0.686-0.930 | 0.084 | 0.033 |
| centre_group: other (<20 each) | Hybrid | youden | 0.151 | 42 | 2 | 0.952 | 0.838-0.994 | 0.038 | 0.014 |

## E2. Sensitivity analysis: features z-scored within centre

| group | model | operating_point | threshold | n | false_positives | specificity | spec_ci | mean_predicted_p_mutant | median_predicted_p_mutant |
|---|---|---|---|---|---|---|---|---|---|
| all | Age | youden | 0.185 | 359 | 44 | 0.877 | 0.839-0.910 | 0.087 | 0.038 |
| all | Features | youden | 0.161 | 359 | 43 | 0.880 | 0.842-0.912 | 0.086 | 0.048 |
| all | Features + age | youden | 0.134 | 359 | 42 | 0.883 | 0.845-0.914 | 0.065 | 0.018 |
| all | Ensemble (raw) | youden | 0.132 | 359 | 33 | 0.908 | 0.873-0.936 | 0.066 | 0.031 |
| all | Ensemble (recalibrated) | youden | 0.114 | 359 | 33 | 0.908 | 0.873-0.936 | 0.055 | 0.023 |
| all | Ensemble + age | youden | 0.108 | 359 | 33 | 0.908 | 0.873-0.936 | 0.044 | 0.011 |
| all | Ensemble + features | youden | 0.116 | 359 | 24 | 0.933 | 0.902-0.957 | 0.045 | 0.019 |
| all | Hybrid | youden | 0.151 | 359 | 16 | 0.955 | 0.929-0.974 | 0.038 | 0.010 |
| all | Confidence gate (RSNA) | youden | 0.124 | 359 | 20 | 0.944 | 0.915-0.966 | 0.051 | 0.024 |

## Leave-UCSF-out, 3 retrained member(s): AUROC on UCSF

| model | n | mutant | auroc | ci |
|---|---|---|---|---|
| Ensemble (raw, retrained) | 433 | 68 | 0.916 | 0.881-0.946 |
| Ensemble (pooled CV, site seen in training) | 433 | 68 | 0.947 | 0.914-0.973 |
| Ensemble (recalibrated) | 433 | 68 | 0.916 | 0.881-0.946 |
| Features + age | 433 | 68 | 0.952 | 0.925-0.974 |
| Ensemble + age | 433 | 68 | 0.961 | 0.939-0.980 |
| Ensemble + features | 433 | 68 | 0.934 | 0.900-0.964 |
| Hybrid | 433 | 68 | 0.964 | 0.941-0.984 |

## Leave-UCSF-out: paired differences

| comparison | delta | ci |
|---|---|---|
| Hybrid minus Features + age | 0.013 | -0.002 to +0.031 |
| Hybrid minus Ensemble + age | 0.003 | -0.010 to +0.017 |
| Hybrid minus Ensemble (recalibrated) | 0.048 | +0.023 to +0.076 |
| Features + age minus Ensemble (recalibrated) | 0.035 | +0.002 to +0.071 |
| Ensemble + features minus Features + age | -0.017 | -0.045 to +0.010 |

## Leave-UCSF-out: each retrained member on UCSF

| member_run | n | auroc |
|---|---|---|
| loso_UCSF | 433 | 0.910 |
| loso_UCSF_CA44 | 433 | 0.922 |
| loso_UCSF_MC42 | 433 | 0.890 |

## Leave-UTSW-out, 1 retrained member(s): AUROC on UTSW

| model | n | mutant | auroc | ci |
|---|---|---|---|---|
| Ensemble (raw, retrained) | 514 | 108 | 0.826 | 0.780-0.871 |
| Ensemble (pooled CV, site seen in training) | 514 | 108 | 0.908 | 0.878-0.937 |
| Ensemble (recalibrated) | 514 | 108 | 0.826 | 0.780-0.871 |
| Features + age | 514 | 108 | 0.918 | 0.889-0.943 |
| Ensemble + age | 514 | 108 | 0.913 | 0.881-0.941 |
| Ensemble + features | 514 | 108 | 0.882 | 0.847-0.916 |
| Hybrid | 514 | 108 | 0.931 | 0.905-0.954 |

## Leave-UTSW-out: paired differences

| comparison | delta | ci |
|---|---|---|
| Hybrid minus Features + age | 0.013 | +0.005 to +0.023 |
| Hybrid minus Ensemble + age | 0.017 | +0.004 to +0.034 |
| Hybrid minus Ensemble (recalibrated) | 0.105 | +0.062 to +0.151 |
| Features + age minus Ensemble (recalibrated) | 0.092 | +0.044 to +0.143 |
| Ensemble + features minus Features + age | -0.036 | -0.065 to -0.004 |

## Leave-UTSW-out: each retrained member on UTSW

| member_run | n | auroc |
|---|---|---|
| loso_UTSW_CA42 | 514 | 0.826 |

## Notes

- E1 occluded inputs are out of distribution: a sensitivity analysis, not a causal map. The blurred conditions test the fill.
- E2: confirm in Filimonova et al. 2026 that every patient is IDH-wildtype before calling the false-positive rate a specificity.
- E2 features are z-scored within the external cohort (as for MU and RHUH); thresholds and coefficients are frozen.
- E4/E5: architecture and recipe were chosen in cross-validation that included the held-out site; only the weights are site-naive.