# External validation of the frozen IDH hybrid models

Run: 2026-10-21 19:01

Frozen development thresholds and coefficients; imaging features z-scored within each external cohort.

| cohort | analysis | model | n | idh_mutant | auroc | auroc_ci | brier | ece | sensitivity | sensitivity_exact_ci | specificity | specificity_exact_ci | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RHUH-GBM | treatment-naive | Age | 38 | 4 | 0.743 | 0.487-1.000 | 0.089 | 0.086 | 0.250 | 0.01-0.81 | 0.971 | 0.85-1.00 | 0.333 |
| RHUH-GBM | treatment-naive | Features | 38 | 4 | 0.551 | 0.162-1.000 | 0.088 | 0.095 | 0.250 | 0.01-0.81 | 0.941 | 0.80-0.99 | 0.286 |
| RHUH-GBM | treatment-naive | Features + age | 38 | 4 | 0.699 | 0.279-1.000 | 0.091 | 0.079 | 0.250 | 0.01-0.81 | 1.000 | 0.90-1.00 | 0.400 |
| RHUH-GBM | treatment-naive | Ensemble (raw) | 38 | 4 | 0.544 | 0.314-0.784 | 0.098 | 0.075 | 0.000 | 0.00-0.60 | 0.941 | 0.80-0.99 | 0.000 |
| RHUH-GBM | treatment-naive | Ensemble (recalibrated) | 38 | 4 | 0.544 | 0.297-0.784 | 0.099 | 0.073 | 0.000 | 0.00-0.60 | 0.971 | 0.85-1.00 | 0.000 |
| RHUH-GBM | treatment-naive | Ensemble + age | 38 | 4 | 0.691 | 0.434-0.917 | 0.099 | 0.081 | 0.000 | 0.00-0.60 | 1.000 | 0.90-1.00 | 0.000 |
| RHUH-GBM | treatment-naive | Ensemble + features | 38 | 4 | 0.544 | 0.171-1.000 | 0.092 | 0.074 | 0.250 | 0.01-0.81 | 0.971 | 0.85-1.00 | 0.333 |
| RHUH-GBM | treatment-naive | Hybrid | 38 | 4 | 0.831 | 0.540-1.000 | 0.099 | 0.087 | 0.000 | 0.00-0.60 | 1.000 | 0.90-1.00 | 0.000 |
| MU-Glioma-Post | earliest scan per patient | Age | 189 | 28 | 0.934 | 0.893-0.970 | 0.074 | 0.048 | 0.857 | 0.67-0.96 | 0.882 | 0.82-0.93 | 0.676 |
| MU-Glioma-Post | earliest scan per patient | Features | 188 | 27 | 0.611 | 0.460-0.755 | 0.123 | 0.078 | 0.407 | 0.22-0.61 | 0.807 | 0.74-0.87 | 0.319 |
| MU-Glioma-Post | earliest scan per patient | Features + age | 188 | 27 | 0.867 | 0.793-0.932 | 0.082 | 0.049 | 0.667 | 0.46-0.83 | 0.826 | 0.76-0.88 | 0.493 |
| MU-Glioma-Post | earliest scan per patient | Ensemble (raw) | 189 | 28 | 0.627 | 0.505-0.740 | 0.149 | 0.116 | 0.571 | 0.37-0.76 | 0.671 | 0.59-0.74 | 0.330 |
| MU-Glioma-Post | earliest scan per patient | Ensemble (recalibrated) | 189 | 28 | 0.627 | 0.510-0.747 | 0.149 | 0.111 | 0.536 | 0.34-0.72 | 0.683 | 0.61-0.75 | 0.319 |
| MU-Glioma-Post | earliest scan per patient | Ensemble + age | 189 | 28 | 0.888 | 0.824-0.941 | 0.088 | 0.038 | 0.786 | 0.59-0.92 | 0.789 | 0.72-0.85 | 0.524 |
| MU-Glioma-Post | earliest scan per patient | Ensemble + features | 188 | 27 | 0.624 | 0.494-0.743 | 0.141 | 0.123 | 0.407 | 0.22-0.61 | 0.702 | 0.62-0.77 | 0.256 |
| MU-Glioma-Post | earliest scan per patient | Hybrid | 188 | 27 | 0.870 | 0.795-0.934 | 0.091 | 0.056 | 0.704 | 0.50-0.86 | 0.832 | 0.77-0.89 | 0.521 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Age | 554 | 95 | 0.930 | 0.881-0.971 | 0.081 | 0.048 | 0.863 | 0.78-0.93 | 0.865 | 0.83-0.89 | 0.686 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Features | 551 | 93 | 0.688 | 0.608-0.766 | 0.130 | 0.071 | 0.495 | 0.39-0.60 | 0.836 | 0.80-0.87 | 0.430 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Features + age | 551 | 93 | 0.904 | 0.863-0.941 | 0.080 | 0.040 | 0.785 | 0.69-0.86 | 0.847 | 0.81-0.88 | 0.619 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Ensemble (raw) | 554 | 95 | 0.711 | 0.590-0.812 | 0.136 | 0.081 | 0.621 | 0.52-0.72 | 0.723 | 0.68-0.76 | 0.420 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Ensemble (recalibrated) | 554 | 95 | 0.711 | 0.593-0.815 | 0.136 | 0.078 | 0.600 | 0.49-0.70 | 0.732 | 0.69-0.77 | 0.415 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Ensemble + age | 554 | 95 | 0.914 | 0.857-0.957 | 0.082 | 0.028 | 0.832 | 0.74-0.90 | 0.810 | 0.77-0.85 | 0.605 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Ensemble + features | 551 | 93 | 0.714 | 0.620-0.794 | 0.130 | 0.092 | 0.548 | 0.44-0.65 | 0.760 | 0.72-0.80 | 0.402 |
| MU-Glioma-Post | all scans (patient-cluster bootstrap) | Hybrid | 551 | 93 | 0.909 | 0.863-0.948 | 0.083 | 0.031 | 0.774 | 0.68-0.85 | 0.869 | 0.83-0.90 | 0.640 |

## Feature medians (raw)

     cohort  vol_whole  enhancing_frac_whole  no_enhancement  edema_ratio_log  mismatch_index  nt2_nonenh_median  nflair_nonenh_median
development     77.476                 0.186             0.0            0.798          -0.085              1.711                 1.784
       RHUH    112.318                 0.208             0.0            0.730           0.046              1.731                 1.590
         MU     54.153                 0.123             0.0            1.782          -0.024              1.503                 1.516

- RHUH has very few IDH-mutant patients: treat sensitivity as descriptive.
- MU is post-treatment imaging, outside the pre-operative intended use.