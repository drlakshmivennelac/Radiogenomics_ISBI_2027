# Confidence gating for the IDH hybrid

Run 2026-10-21 16:48 | 1393 patients, 186 IDH-mutant | 5x5 nested CV. DL arm = ensemble; ML arm = features + age. Better single arm: A · ML arm (features + age).

## Performance (95% CI)

| strategy | AUROC | AP | Brier | ECE | Slope | Sens. | Spec. | PPV | F1 | MCC | Spec. @90% sens. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A · ML arm (features + age) | 0.940 (0.923-0.956) | 0.790 (0.741-0.841) | 0.054 (0.046-0.061) | 0.009 | 0.992 (0.890-1.129) | 0.833 (0.780-0.887) | 0.886 (0.868-0.904) | 0.529 (0.489-0.575) | 0.647 (0.607-0.689) | 0.600 (0.551-0.650) | 0.804 (0.782-0.827) |
| B · DL arm (ensemble) | 0.925 (0.902-0.944) | 0.765 (0.713-0.811) | 0.058 (0.051-0.066) | 0.012 | 1.011 (0.905-1.135) | 0.823 (0.763-0.876) | 0.866 (0.846-0.886) | 0.486 (0.447-0.527) | 0.611 (0.570-0.650) | 0.560 (0.509-0.606) | 0.750 (0.726-0.775) |
| C · Fixed blend | 0.954 (0.937-0.969) | 0.848 (0.808-0.889) | 0.047 (0.041-0.053) | 0.035 | 1.412 (1.254-1.635) | 0.849 (0.796-0.903) | 0.924 (0.909-0.939) | 0.632 (0.589-0.684) | 0.725 (0.685-0.767) | 0.685 (0.639-0.736) | 0.857 (0.837-0.876) |
| D1 · Confidence gate (RSNA) | 0.956 (0.940-0.971) | 0.858 (0.819-0.895) | 0.046 (0.039-0.052) | 0.026 | 1.360 (1.205-1.583) | 0.860 (0.812-0.909) | 0.929 (0.914-0.942) | 0.650 (0.604-0.698) | 0.741 (0.700-0.780) | 0.704 (0.656-0.749) | 0.866 (0.846-0.885) |
| D2 · Gate, threshold-centred | 0.957 (0.942-0.971) | 0.850 (0.808-0.890) | 0.044 (0.038-0.051) | 0.020 | 1.171 (1.056-1.330) | 0.860 (0.812-0.909) | 0.909 (0.891-0.925) | 0.593 (0.548-0.641) | 0.702 (0.662-0.743) | 0.662 (0.616-0.709) | 0.865 (0.844-0.884) |
| E · Stacked | 0.958 (0.943-0.972) | 0.859 (0.821-0.896) | 0.043 (0.036-0.050) | 0.018 | 1.013 (0.903-1.166) | 0.871 (0.823-0.919) | 0.930 (0.915-0.943) | 0.656 (0.610-0.704) | 0.748 (0.710-0.787) | 0.713 (0.668-0.757) | 0.863 (0.843-0.882) |
| H · Full hybrid | 0.961 (0.946-0.974) | 0.866 (0.830-0.902) | 0.042 (0.035-0.049) | 0.016 | 0.991 (0.883-1.143) | 0.866 (0.812-0.914) | 0.922 (0.907-0.937) | 0.631 (0.588-0.681) | 0.730 (0.691-0.771) | 0.693 (0.647-0.740) | 0.878 (0.859-0.897) |

## Paired differences (95% CI)

| comparison | pre-declared | AUROC | AP | Brier | Spec. | F1 |
|---|---|---|---|---|---|---|
| D1 · Confidence gate (RSNA)  minus  A · ML arm (features + age) | True | +0.016 (+0.007 to +0.026) | +0.066 (+0.034 to +0.102) | -0.008 (-0.013 to -0.003) | +0.043 (+0.027 to +0.059) | +0.094 (+0.058 to +0.129) |
| D1 · Confidence gate (RSNA)  minus  H · Full hybrid | True | -0.005 (-0.010 to +0.001) | -0.009 (-0.020 to +0.003) | +0.004 (+0.001 to +0.006) | +0.007 (-0.005 to +0.017) | +0.010 (-0.017 to +0.038) |
| D2 · Gate, threshold-centred  minus  D1 · Confidence gate (RSNA) | True | +0.001 (-0.003 to +0.005) | -0.007 (-0.028 to +0.010) | -0.001 (-0.004 to +0.002) | -0.020 (-0.031 to -0.010) | -0.039 (-0.065 to -0.013) |
| D1 · Confidence gate (RSNA)  minus  B · DL arm (ensemble) | False | +0.032 (+0.019 to +0.046) | +0.093 (+0.064 to +0.125) | -0.013 (-0.018 to -0.008) | +0.063 (+0.046 to +0.080) | +0.130 (+0.094 to +0.166) |
| D2 · Gate, threshold-centred  minus  H · Full hybrid | False | -0.004 (-0.010 to +0.001) | -0.016 (-0.039 to +0.002) | +0.003 (-0.000 to +0.005) | -0.013 (-0.025 to -0.002) | -0.028 (-0.052 to -0.005) |
| E · Stacked  minus  H · Full hybrid | False | -0.003 (-0.007 to +0.002) | -0.007 (-0.017 to +0.002) | +0.001 (-0.000 to +0.003) | +0.008 (-0.002 to +0.018) | +0.018 (-0.005 to +0.040) |
| C · Fixed blend  minus  D1 · Confidence gate (RSNA) | False | -0.003 (-0.005 to -0.000) | -0.010 (-0.019 to +0.001) | +0.001 (-0.000 to +0.003) | -0.005 (-0.017 to +0.007) | -0.015 (-0.045 to +0.014) |

## Gate weights by group

| group | n | mutant | D1 mean w_ML | D1 % patients led by DL (w_ML<0.5) | D2 mean w_ML | D2 % led by DL | ML-arm errors / DL-arm errors |
|---|---|---|---|---|---|---|---|
| all: all | 1393 | 186 | 0.5 | 36.3 | 0.56 | 33.9 | 169 / 195 |
| grade: grade 4 | 1185 | 39 | 0.5 | 33.0 | 0.57 | 31.9 | 114 / 147 |
| grade: grade 2-3 | 205 | 147 | 0.48 | 55.6 | 0.51 | 44.9 | 55 / 47 |
| grade: unknown | 3 | 0 | 0.55 | 33.3 | 0.41 | 66.7 | 0 / 1 |
| age: >=55 | 950 | 29 | 0.52 | 22.1 | 0.61 | 22.6 | 50 / 115 |
| age: <40 | 148 | 97 | 0.47 | 59.5 | 0.55 | 42.6 | 42 / 25 |
| age: 40-54 | 295 | 60 | 0.46 | 70.5 | 0.43 | 65.8 | 77 / 55 |
| enhancement: enhancing | 1203 | 75 | 0.5 | 33.1 | 0.57 | 31.7 | 115 / 146 |
| enhancement: non-enhancing | 190 | 111 | 0.49 | 56.8 | 0.51 | 47.9 | 54 / 49 |

## Subgroup AUROC

| group | n | mutant | A · ML arm (features + age) | B · DL arm (ensemble) | D1 · Confidence gate (RSNA) | D2 · Gate, threshold-centred | E · Stacked | H · Full hybrid |
|---|---|---|---|---|---|---|---|---|
| grade: grade 2-3 | 205 | 147 | 0.789 | 0.813 | 0.84 | 0.847 | 0.844 | 0.861 |
| grade: grade 4 | 1185 | 39 | 0.931 | 0.875 | 0.944 | 0.938 | 0.944 | 0.954 |
| age: 40-54 | 295 | 60 | 0.86 | 0.887 | 0.895 | 0.895 | 0.896 | 0.896 |
| age: <40 | 148 | 97 | 0.868 | 0.923 | 0.939 | 0.922 | 0.932 | 0.933 |
| age: >=55 | 950 | 29 | 0.882 | 0.873 | 0.903 | 0.919 | 0.915 | 0.919 |
| enhancement: enhancing | 1203 | 75 | 0.896 | 0.876 | 0.921 | 0.924 | 0.927 | 0.93 |
| enhancement: non-enhancing | 190 | 111 | 0.909 | 0.879 | 0.948 | 0.948 | 0.951 | 0.953 |
| site: UCSF | 433 | 68 | 0.958 | 0.946 | 0.973 | 0.97 | 0.973 | 0.974 |
| site: UPENN | 446 | 10 | 0.925 | 0.86 | 0.936 | 0.937 | 0.937 | 0.936 |
| site: UTSW | 514 | 108 | 0.918 | 0.907 | 0.939 | 0.941 | 0.942 | 0.946 |

## Age-matched AUROC

| strategy | AUROC, all | AUROC, age-matched |
|---|---|---|
| A · ML arm (features + age) | 0.94 | 0.793 (0.753-0.835) |
| B · DL arm (ensemble) | 0.925 | 0.897 (0.858-0.928) |
| C · Fixed blend | 0.954 | 0.870 (0.833-0.908) |
| D1 · Confidence gate (RSNA) | 0.956 | 0.884 (0.846-0.921) |
| D2 · Gate, threshold-centred | 0.957 | 0.884 (0.846-0.921) |
| E · Stacked | 0.958 | 0.882 (0.844-0.918) |
| H · Full hybrid | 0.961 | 0.868 (0.829-0.909) |

## Deferral by each strategy's own confidence

| strategy | error rate, no deferral | error rate, 20% deferred | AURC (50-100% coverage) |
|---|---|---|---|
| H · Full hybrid | 0.085 | 0.021 | 0.0239 |
| D1 · Confidence gate (RSNA) | 0.08 | 0.041 | 0.0422 |
| D2 · Gate, threshold-centred | 0.098 | 0.021 | 0.0278 |
| B · DL arm (ensemble) | 0.14 | 0.081 | 0.0779 |

## Notes

- Pre-declared: D1 vs the better single arm, D1 vs H, D2 vs D1. Everything else is exploratory.
- D1 is exactly the RSNA rule (distance from 0.5). D2 measures distance from each arm's training-fold Youden threshold on the log-odds scale.
- The fixed blend weight and the stacked model are fitted on inner-CV predictions of the training folds only.
- ECE intervals are not reported (biased under resampling).