# What the ensemble adds beyond age and radiology

Run 2026-10-21 21:05 | 1393 patients, 186 IDH-mutant | internal pooled-site cross-validation.

Cross-validated AUROC (reproduced from the hybrid notebook): Age 0.889, Features 0.883, Features + age 0.940, Ensemble (raw) 0.926, Ensemble (recalibrated) 0.925, Ensemble + age 0.955, Ensemble + features 0.933, Hybrid 0.961

## A1. Correlation of each component with age (Spearman, 95% CI)

| population | rho, ensemble logit | rho, features score |
|---|---|---|
| all patients | -0.306 (-0.349 to -0.266) | -0.284 (-0.327 to -0.238) |
| IDH-wildtype only | -0.088 (-0.142 to -0.034) | -0.090 (-0.144 to -0.039) |
| IDH-mutant only | -0.179 (-0.323 to -0.039) | -0.231 (-0.362 to -0.089) |
| within site x IDH class (pooled) | -0.094 (-0.145 to -0.044) | -0.116 (-0.167 to -0.067) |

## A2. Share of the ensemble output explained by age, site and features (cross-validated R²)

| predictors | all | wildtype only | mutant only |
|---|---|---|---|
| age | 0.191 | 0.003 | 0.012 |
| site | 0.055 | 0.015 | 0.06 |
| handcrafted features | 0.457 | 0.22 | 0.24 |
| age + features | 0.491 | 0.222 | 0.224 |
| age + features + site | 0.51 | 0.25 | 0.241 |

## A3. AUROC within age bands (95% CI)

| age band | n | mutant | Age | Features | Ensemble (raw) | Features + age | Hybrid |
|---|---|---|---|---|---|---|---|
| <40 | 148 | 97 | 0.552 (0.451 to 0.652) | 0.881 (0.820 to 0.935) | 0.923 (0.878 to 0.961) | 0.868 (0.804 to 0.925) | 0.933 (0.894 to 0.968) |
| 40-54 | 295 | 60 | 0.731 (0.662 to 0.807) | 0.841 (0.777 to 0.898) | 0.889 (0.842 to 0.931) | 0.860 (0.796 to 0.916) | 0.896 (0.843 to 0.942) |
| >=55 | 950 | 29 | 0.716 (0.647 to 0.783) | 0.868 (0.817 to 0.914) | 0.877 (0.810 to 0.926) | 0.882 (0.831 to 0.929) | 0.919 (0.860 to 0.966) |

## A4. AUROC in an age-matched sample (cluster bootstrap over matched sets)

| model | AUROC, age-matched (95% CI) | AUROC, all patients |
|---|---|---|
| Age | 0.533 (0.516 to 0.549) | 0.889 |
| Features | 0.878 (0.835 to 0.916) | 0.883 |
| Ensemble (raw) | 0.900 (0.862 to 0.930) | 0.926 |
| Features + age | 0.793 (0.753 to 0.835) | 0.94 |
| Ensemble + age | 0.835 (0.794 to 0.877) | 0.955 |
| Hybrid | 0.868 (0.829 to 0.909) | 0.961 |

96 of 186 mutants matched to 262 wildtype patients (same site, |age difference| <= 2 y, mean difference 0.43 y). Age AUROC near 0.5 confirms that age carries no information in this sample.

## A5. Information unique to each component

| component | AUROC on its own | AUROC of what is left (95% CI) | likelihood-ratio chi2 (df) | p |
|---|---|---|---|---|
| ensemble, beyond age + features | 0.926 | 0.699 (0.644 to 0.748) | 107.3 (1) | 3.8e-25 |
| features, beyond ensemble + age | 0.883 | 0.562 (0.512 to 0.611) | 38.1 (8) | 7.1e-06 |
| age (younger = mutant), beyond ensemble + features | 0.89 | 0.702 (0.661 to 0.741) | 112.1 (1) | 3.4e-26 |

## B1. Deferral strategies (AURC: mean hybrid error over 50-100% coverage; lower is better)

| strategy | AURC | AUROC for flagging hybrid errors |
|---|---|---|
| random deferral (expected) | 0.0854 |  |
| low hybrid confidence | 0.0355 | 0.845 |
| component conflict | 0.0469 | 0.77 |
| combined (mean rank) | 0.039 | 0.819 |

## B1b. Paired bootstrap differences (negative = first strategy better)

| comparison | AURC difference (95% CI) |
|---|---|
| combined (mean rank) minus low hybrid confidence | 0.0036 (0.0006 to 0.0062) |
| component conflict minus low hybrid confidence | 0.0115 (0.0060 to 0.0166) |

## B2. Operating points (hybrid at its nested Youden threshold)

| strategy | deferral | deferred | hybrid errors captured | mutants deferred | retained accuracy | retained sensitivity | retained specificity | retained PPV |
|---|---|---|---|---|---|---|---|---|
| no deferral | 0% | 0 | 0/119 | 0 | 0.915 | 0.866 | 0.922 | 0.631 |
| low hybrid confidence | 10% | 139 | 51/119 (43%) | 21 | 0.946 | 0.915 | 0.95 | 0.737 |
| component conflict | 10% | 139 | 33/119 (28%) | 22 | 0.931 | 0.866 | 0.941 | 0.689 |
| combined (mean rank) | 10% | 139 | 45/119 (38%) | 14 | 0.941 | 0.878 | 0.951 | 0.74 |
| low hybrid confidence | 20% | 279 | 82/119 (69%) | 43 | 0.967 | 0.944 | 0.97 | 0.823 |
| component conflict | 20% | 279 | 67/119 (56%) | 37 | 0.953 | 0.913 | 0.96 | 0.777 |
| combined (mean rank) | 20% | 279 | 74/119 (62%) | 44 | 0.96 | 0.944 | 0.962 | 0.784 |
| low hybrid confidence | 30% | 418 | 93/119 (78%) | 54 | 0.973 | 0.962 | 0.975 | 0.858 |
| component conflict | 30% | 418 | 83/119 (70%) | 59 | 0.963 | 0.945 | 0.966 | 0.805 |
| combined (mean rank) | 30% | 418 | 93/119 (78%) | 54 | 0.973 | 0.955 | 0.976 | 0.863 |

## B3. Joint model of hybrid errors

| predictor of a hybrid error (per SD) | odds ratio (95% CI) |
|---|---|
| low hybrid confidence | 7.94 (4.67 to 14.74) |
| component conflict | 0.75 (0.47 to 1.11) |

## B3c. Binary agreement of the components versus hybrid correctness

| binary calls | hybrid right | hybrid wrong |
|---|---|---|
| components agree | 1095 | 62 |
| components disagree | 179 | 57 |

## C1-C4. Pre-declared attribution tests (Mann-Whitney, Holm)

| test | group 1 median (IQR) | group 2 median (IQR) | P(group 1 > group 2) | p | hypothesis | Holm p |
|---|---|---|---|---|---|---|
| C1 FLAIR enrichment, non-enhancing lesion: mutant vs wildtype | 0.285 (0.160-1.532), n=62 | 0.536 (0.190-1.586), n=66 | 0.455 | 0.3842 | mutant higher | 1.0000 |
| C2 T1c signed density, enhancing tumour: mutant vs wildtype | -3.945 (-7.452--1.696), n=40 | -6.857 (-11.878--3.620), n=58 | 0.619 | 0.0472 | mutant higher | 0.1890 |
| C3 attribution outside the lesion: errors vs correct | 0.922 (0.870-0.948), n=65 | 0.922 (0.867-0.956), n=66 | 0.499 | 0.9835 | errors higher | 1.0000 |
| C4 FLAIR enrichment, non-enhancing lesion: FN vs TP | 0.322 (0.193-1.400), n=32 | 0.227 (0.138-2.282), n=30 | 0.522 | 0.7728 | FN lower | 1.0000 |

## C5. Attribution by outcome, median (IQR) (exploratory)

| outcome | n | FLAIR share, non-enh. lesion | T1c share, enhancing | share outside lesion | FLAIR share of all attribution | FLAIR top-10% in oedema | T1c top-10% in enhancing |
|---|---|---|---|---|---|---|---|
| TP | 33 | 0.02 (0.01-0.12) | 0.00 (0.00-0.00) | 0.94 (0.92-0.96) | 0.17 (0.06-0.29) | 0.00 (0.00-0.25) | 0.00 (0.00-0.00) |
| FN | 32 | 0.03 (0.01-0.11) | 0.03 (0.01-0.05) | 0.89 (0.82-0.92) | 0.15 (0.09-0.26) | 0.01 (0.00-0.13) | 0.00 (0.00-0.00) |
| TN | 33 | 0.05 (0.02-0.07) | 0.05 (0.03-0.12) | 0.87 (0.83-0.92) | 0.23 (0.10-0.45) | 0.03 (0.00-0.10) | 0.00 (0.00-0.01) |
| FP | 33 | 0.02 (0.01-0.06) | 0.01 (0.00-0.02) | 0.94 (0.92-0.97) | 0.13 (0.06-0.22) | 0.01 (0.00-0.18) | 0.00 (0.00-0.00) |

## C6. Attribution versus handcrafted features (exploratory)

| attribution metric | handcrafted feature | n | Spearman rho | p | question |
|---|---|---|---|---|---|
| FLAIR_enrich_nonenh_lesion | mismatch_index | 128 | 0.003 | 0.9730 | network uses non-enhancing FLAIR where T2-FLAIR mismatch is strong? |
| FLAIR_share_nonenh_lesion | enhancing_frac_whole | 128 | -0.033 | 0.7111 | less non-enhancing attention when the tumour enhances? |
| T1c_share_enhancing | enhancing_frac_whole | 128 | 0.885 | 0.0000 | T1c attention follows enhancement? |
| outside_lesion_share | log_vol_whole | 128 | -0.398 | 0.0000 | small lesions push attribution outside? |

## D1. What the representation encodes (cross-validated R²; IDH as AUROC, site as balanced accuracy)

| target | v2_reference_seed42 | v2_reference_seed44 | v2_mean_concat_seed42 | untrained control | trained members (mean) | gain over untrained |
|---|---|---|---|---|---|---|
| log_vol_whole | 0.071 | -0.076 | 0.118 | 0.392 | 0.038 | -0.354 |
| log_vol_enhancing | 0.391 | 0.338 | 0.308 | 0.382 | 0.346 | -0.036 |
| enhancing_frac_whole | 0.291 | 0.277 | 0.241 | 0.215 | 0.27 | 0.055 |
| no_enhancement | 0.291 | 0.31 | 0.291 | 0.121 | 0.297 | 0.176 |
| necrosis_frac | -0.051 | -0.042 | -0.024 | -0.02 | -0.039 | -0.019 |
| edema_ratio_log | 0.193 | 0.214 | 0.142 | 0.106 | 0.183 | 0.077 |
| mismatch_index | 0.009 | -0.012 | 0.01 | 0.012 | 0.003 | -0.01 |
| mismatch_frac | 0.003 | -0.009 | -0.01 | -0.006 | -0.005 | 0.001 |
| age | 0.162 | 0.144 | 0.15 | 0.07 | 0.152 | 0.082 |
| IDH (AUROC) | 0.901 | 0.881 | 0.89 | 0.795 | 0.891 | 0.096 |
| site (balanced accuracy) | 0.476 | 0.491 | 0.49 | 0.624 | 0.486 | -0.138 |

## D2. Feature probes in IDH-wildtype patients only (cross-validated R²)

| target | v2_reference_seed42 | v2_reference_seed44 | v2_mean_concat_seed42 | untrained control | trained members (mean) | gain over untrained |
|---|---|---|---|---|---|---|
| log_vol_whole | 0.071 | -0.076 | 0.082 | 0.379 | 0.026 | -0.353 |
| log_vol_enhancing | 0.249 | 0.174 | 0.202 | 0.353 | 0.208 | -0.145 |
| enhancing_frac_whole | 0.167 | 0.177 | 0.14 | 0.179 | 0.161 | -0.018 |
| no_enhancement | 0.024 | 0.034 | -0.039 | 0.065 | 0.006 | -0.059 |
| necrosis_frac | -0.05 | 0.012 | -0.056 | -0.009 | -0.031 | -0.023 |
| edema_ratio_log | 0.104 | 0.101 | 0.066 | 0.085 | 0.09 | 0.005 |
| mismatch_index | -0.016 | -0.034 | -0.015 | 0.025 | -0.022 | -0.047 |
| mismatch_frac | -0.043 | -0.013 | -0.035 | -0.01 | -0.031 | -0.021 |
| age | -0.012 | -0.027 | -0.008 | 0.008 | -0.015 | -0.023 |

## D3. Alignment of probe directions with the IDH head (|cosine|)

| target | v2_reference_seed42 | v2_reference_seed44 | v2_mean_concat_seed42 | untrained control |
|---|---|---|---|---|
| log_vol_whole | 0.047 | 0.145 | 0.037 | 0.046 |
| log_vol_enhancing | 0.035 | 0.018 | 0.079 | 0.047 |
| enhancing_frac_whole | 0.057 | 0.043 | 0.042 | 0.052 |
| no_enhancement | 0.247 | 0.289 | 0.312 | 0.03 |
| necrosis_frac | 0.354 | 0.144 | 0.203 | 0.048 |
| edema_ratio_log | 0.178 | 0.152 | 0.118 | 0.051 |
| mismatch_index | 0.442 | 0.412 | 0.398 | 0.028 |
| mismatch_frac | 0.355 | 0.398 | 0.377 | 0.049 |
| age | 0.45 | 0.336 | 0.394 | 0.039 |

## Notes

- Pre-declared: A1 within-class correlation, A4 age-matched AUROC, C1-C4. Part B and everything else is exploratory.
- Deferral scores use only pre-operative inputs (images, segmentation, age).
- The XRAI sample is outcome-enriched (33 per outcome): group comparisons are valid, frequencies are not.
- Probe R² shows linear decodability, not use by the IDH head; D3 tests alignment with the head directly.
- Likelihood-ratio tests in A5 are in-sample and descriptive.