# v8 recomputations (29 Oct 2026)

Requested by the Stage 2 audit of v7. Script: `code/29oct/recompute_v8.py` (run as `python recompute_v8.py results/ results/14_v8_recomputations_29oct/`). Inputs are the package's own per-patient files: `03/cv_predictions.csv`, `09/gating_cv_predictions.csv`, `01/features.csv`, `02/frozen_models.json`, `13/e2_ds007045/frozen_model_predictions.csv`. Seed 20261021, 2,000 bootstraps. The published age matching (96 sets, 358 patients) is reproduced exactly before anything is computed.

## S1. Age-matched sample: AUROC within matched sets

Each mutant is compared only with its own age-matched, same-site wildtype controls; CIs from a set-level bootstrap. The v7 table pooled all 358 patients, which compares patients across sets and lets age differences between sets back in.

| model                |   within-set AUROC | ci          |   pooled AUROC in matched sample (v7) |
|:---------------------|-------------------:|:------------|--------------------------------------:|
| Age                  |              0.412 | 0.339-0.488 |                                 0.533 |
| Features             |              0.878 | 0.824-0.926 |                                 0.878 |
| Features + age       |              0.859 | 0.801-0.911 |                                 0.793 |
| Ensemble (raw)       |              0.908 | 0.867-0.947 |                                 0.9   |
| Ensemble + age       |              0.908 | 0.868-0.947 |                                 0.835 |
| Ensemble + features  |              0.924 | 0.879-0.962 |                                 0.917 |
| Hybrid               |              0.924 | 0.878-0.965 |                                 0.868 |
| Confidence gate (D1) |              0.912 | 0.863-0.956 |                                 0.884 |

| comparison                          |   delta | ci               |
|:------------------------------------|--------:|:-----------------|
| Features + age minus Ensemble (raw) |  -0.05  | -0.099 to +0.000 |
| Ensemble (raw) minus Features       |   0.031 | -0.016 to +0.075 |
| Ensemble + features minus Features  |   0.046 | +0.012 to +0.082 |
| Hybrid minus Features + age         |   0.065 | +0.027 to +0.105 |
| Hybrid minus Ensemble + age         |   0.015 | -0.015 to +0.047 |

## S2. Pooled-CV subgroups with paired CIs (site x class stratified bootstrap within the subgroup)

| subgroup      |    n |   mutant |   Features + age |   Ensemble (raw) |   Ensemble + age |   Hybrid | Ensemble minus Features + age   |
|:--------------|-----:|---------:|-----------------:|-----------------:|-----------------:|---------:|:--------------------------------|
| age <40       |  148 |       97 |            0.868 |            0.923 |            0.92  |    0.933 | +0.055 (-0.005 to +0.118)       |
| grade 2-3     |  205 |      147 |            0.789 |            0.814 |            0.862 |    0.861 | +0.025 (-0.047 to +0.090)       |
| grade 4       | 1185 |       39 |            0.931 |            0.878 |            0.95  |    0.954 | -0.053 (-0.108 to -0.004)       |
| non-enhancing |  190 |      111 |            0.909 |            0.879 |            0.94  |    0.953 | -0.030 (-0.093 to +0.029)       |

## S3. ds007045: exact McNemar tests at the frozen thresholds

| operating point   | model          |   false positives |   specificity |
|:------------------|:---------------|------------------:|--------------:|
| threshold_youden  | Age            |                44 |         0.877 |
| threshold_youden  | Features + age |                44 |         0.877 |
| threshold_youden  | Ensemble (raw) |                33 |         0.908 |
| threshold_youden  | Hybrid         |                17 |         0.953 |
| threshold_sens90  | Age            |               140 |         0.61  |
| threshold_sens90  | Features + age |                77 |         0.786 |
| threshold_sens90  | Ensemble (raw) |                72 |         0.799 |
| threshold_sens90  | Hybrid         |                40 |         0.889 |

| operating point   | comparison                       |   FP only by first |   FP only by second |   exact McNemar p |
|:------------------|:---------------------------------|-------------------:|--------------------:|------------------:|
| threshold_youden  | Hybrid vs Features + age         |                  2 |                  29 |           4.6e-07 |
| threshold_youden  | Ensemble (raw) vs Features + age |                 19 |                  30 |           0.15    |
| threshold_youden  | Hybrid vs Ensemble (raw)         |                  5 |                  21 |           0.0025  |
| threshold_sens90  | Hybrid vs Features + age         |                  4 |                  41 |           9.3e-09 |
| threshold_sens90  | Ensemble (raw) vs Features + age |                 44 |                  49 |           0.68    |
| threshold_sens90  | Hybrid vs Ensemble (raw)         |                 14 |                  46 |           4.2e-05 |

| group            |   n |   Age FP |   Features + age FP |   Ensemble FP |   Hybrid FP |
|:-----------------|----:|---------:|--------------------:|--------------:|------------:|
| ds007045 age <40 |  13 |       13 |                  12 |             3 |           6 |

## S4. Ensemble and age within IDH class

| class        |   spearman rho(ensemble, age) |
|:-------------|------------------------------:|
| IDH-wildtype |                        -0.088 |
| IDH-mutant   |                        -0.179 |

## S5. Calibration slope on the reserved set (frozen models)

Logistic recalibration slope on the predicted log-odds; the slope is unaffected by the 50/50 case-control design, unlike the Brier score and calibration-in-the-large. Site x class stratified bootstrap.

| model                  |   calibration slope | ci        |
|:-----------------------|--------------------:|:----------|
| Features + age         |                1.06 | 0.80-1.52 |
| Ensemble (raw)         |                0.97 | 0.75-1.32 |
| Ensemble + age         |                1.02 | 0.83-1.36 |
| Hybrid                 |                1.03 | 0.82-1.41 |
| Confidence gate (RSNA) |                1.39 | 1.08-2.01 |

## S6. ds007045 without the subjects added in dataset v2.0.0

The data descriptor (Filimonova et al., Sci Data 13:1213, 2026) gives IDH-wildtype GBM (WHO CNS5) as an inclusion criterion for its 337 patients. Dataset v2.0.0 (OpenNeuro CHANGES, 2026-09-04) added Karlsruhe subjects KA103, KA117, KA119–128, KA130–136 and KA138–152 and removed six others. Its README says only "histologically confirmed primary glioblastoma", and participants.tsv has no IDH column. 32 of the added subjects are among our 359.

| operating point   | model          | subset                          |   n |   false positives |   specificity % | ci        |
|:------------------|:---------------|:--------------------------------|----:|------------------:|----------------:|:----------|
| threshold_youden  | Features + age | all 359                         | 359 |                44 |            87.7 | 83.9-91.0 |
| threshold_youden  | Features + age | original descriptor cohort only | 327 |                41 |            87.5 | 83.4-90.9 |
| threshold_youden  | Features + age | added in v2.0.0 only            |  32 |                 3 |            90.6 | 75.0-98.0 |
| threshold_youden  | Ensemble (raw) | all 359                         | 359 |                33 |            90.8 | 87.3-93.6 |
| threshold_youden  | Ensemble (raw) | original descriptor cohort only | 327 |                31 |            90.5 | 86.8-93.5 |
| threshold_youden  | Ensemble (raw) | added in v2.0.0 only            |  32 |                 2 |            93.8 | 79.2-99.2 |
| threshold_youden  | Hybrid         | all 359                         | 359 |                17 |            95.3 | 92.5-97.2 |
| threshold_youden  | Hybrid         | original descriptor cohort only | 327 |                15 |            95.4 | 92.5-97.4 |
| threshold_youden  | Hybrid         | added in v2.0.0 only            |  32 |                 2 |            93.8 | 79.2-99.2 |
| threshold_sens90  | Features + age | all 359                         | 359 |                77 |            78.6 | 73.9-82.7 |
| threshold_sens90  | Features + age | original descriptor cohort only | 327 |                74 |            77.4 | 72.4-81.8 |
| threshold_sens90  | Features + age | added in v2.0.0 only            |  32 |                 3 |            90.6 | 75.0-98.0 |
| threshold_sens90  | Ensemble (raw) | all 359                         | 359 |                72 |            79.9 | 75.4-84.0 |
| threshold_sens90  | Ensemble (raw) | original descriptor cohort only | 327 |                69 |            78.9 | 74.1-83.2 |
| threshold_sens90  | Ensemble (raw) | added in v2.0.0 only            |  32 |                 3 |            90.6 | 75.0-98.0 |
| threshold_sens90  | Hybrid         | all 359                         | 359 |                40 |            88.9 | 85.1-91.9 |
| threshold_sens90  | Hybrid         | original descriptor cohort only | 327 |                38 |            88.4 | 84.4-91.6 |
| threshold_sens90  | Hybrid         | added in v2.0.0 only            |  32 |                 2 |            93.8 | 79.2-99.2 |

## S7. Unseen-site features + age without per-site z-scoring of the held-out site

Features are standardised by the pipeline's scaler fitted on the training sites only, so the held-out site gets no label-free adaptation. Requested by the Stage 2 review of v8. The networks are unaffected; the ensemble scored 0.916 (UCSF) and 0.826 (UTSW).

| held-out site   | feature standardisation       |   Features + age AUROC | ci          |
|:----------------|:------------------------------|-----------------------:|:------------|
| UCSF            | own-site z (as published)     |                  0.952 | 0.925-0.974 |
| UCSF            | training-site statistics only |                  0.948 | 0.920-0.972 |
| UTSW            | own-site z (as published)     |                  0.918 | 0.889-0.943 |
| UTSW            | training-site statistics only |                  0.91  | 0.881-0.937 |

## S8. Reserved-set sensitivity at the frozen thresholds (all and grade-4 mutants)

| operating point   | model          | mutants   | detected   |   sensitivity % | ci        |
|:------------------|:---------------|:----------|:-----------|----------------:|:----------|
| threshold_youden  | Features + age | all       | 84/92      |            91.3 | 83.6-96.2 |
| threshold_youden  | Features + age | grade 4   | 27/31      |            87.1 | 70.2-96.4 |
| threshold_youden  | Ensemble (raw) | all       | 70/92      |            76.1 | 66.1-84.4 |
| threshold_youden  | Ensemble (raw) | grade 4   | 21/31      |            67.7 | 48.6-83.3 |
| threshold_youden  | Hybrid         | all       | 81/92      |            88   | 79.6-93.9 |
| threshold_youden  | Hybrid         | grade 4   | 26/31      |            83.9 | 66.3-94.5 |
| threshold_sens90  | Features + age | all       | 87/92      |            94.6 | 87.8-98.2 |
| threshold_sens90  | Features + age | grade 4   | 29/31      |            93.5 | 78.6-99.2 |
| threshold_sens90  | Ensemble (raw) | all       | 75/92      |            81.5 | 72.1-88.9 |
| threshold_sens90  | Ensemble (raw) | grade 4   | 22/31      |            71   | 52.0-85.8 |
| threshold_sens90  | Hybrid         | all       | 85/92      |            92.4 | 84.9-96.9 |
| threshold_sens90  | Hybrid         | grade 4   | 29/31      |            93.5 | 78.6-99.2 |
