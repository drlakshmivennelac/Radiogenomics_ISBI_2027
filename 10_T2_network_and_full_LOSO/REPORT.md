# T2+FLAIR network and full-pipeline leave-UCSF-out

Analysed 22 Oct 2026 from existing runs on Tara's machine (no new training). Script: `analysis_script.py`; full printout: `raw_output.txt`.

## 1. Does a network that reads T2 add anything?

Runs `Downloads/T2-Flair_RadGenome/*_joint_seed42_matched01`: same 1,435 development patients and same five folds as the
paper's members, with the T1c channel replaced by T2 (inputs T2 + FLAIR). Three fusion designs.

| T2+FLAIR network (out-of-fold, 1,393 analysis patients) | AUROC |
|---|---|
| Cross-attention | 0.756 |
| Pooled (mean-concat) | 0.784 |
| T2 - FLAIR difference | 0.734 |
| Mean of the three, recalibrated | 0.805 |

| Comparison (repeated 5x5 CV, paired stratified bootstrap) | AUROC | Difference (95% CI) |
|---|---|---|
| Ensemble (paper) / as four-member ensemble with one T2 network | 0.926 / 0.920-0.924 | slightly lower, CIs include 0 |
| Ensemble + T2 network vs ensemble | 0.930 vs 0.925 | +0.005 (-0.002 to +0.012) |
| Features + age + T2 network vs features + age | 0.950 vs 0.940 | +0.010 (+0.002 to +0.018) |
| **Hybrid + T2 network vs hybrid** | **0.962 vs 0.961** | **+0.001 (-0.002 to +0.004)** |

Reading: a T2-reading network is much weaker than the T1c+FLAIR members and adds nothing once the hybrid has the
ensemble, the handcrafted mismatch features and age. This answers "the network never sees T2".

## 2. Full-pipeline leave-UCSF-out

Run `clinical_experiments/loso_UCSF`: one CA42-type member retrained with inner 5-fold CV on UTSW + UPENN only
(checkpoint selection on inner validation), scored on all UCSF patients. The combiner was fitted on UTSW + UPENN using that
run's own inner out-of-fold logits, so no model or combiner saw UCSF labels.

| Evaluated on UCSF (n = 433, 68 mutant) | AUROC (95% CI) |
|---|---|
| CA42 member, UCSF in training (pooled-CV logits) | 0.911 |
| CA42 member, retrained without UCSF | 0.910 (0.872-0.943) |
| Features + age, fitted on UTSW + UPENN | 0.952 (0.927-0.974) |
| Member + age | 0.952 (0.923-0.976) |
| **Hybrid (member + features + age), fully site-held-out** | **0.962 (0.939-0.982)** |

Hybrid minus features + age: +0.011 (-0.002 to +0.025). Caveats: one member, not the three-member ensemble; one site
only (the UTSW run did not finish); UCSF has the most grade 2-3 tumours; architecture and recipe were chosen on CV that
included UCSF.

## 3. The 200 reserved patients

`balanced_holdout_ids.json` (100 mutant / 100 wildtype; UCSF 61, UTSW 76, UPENN 63). Checked:
- absent from every v2 fold manifest (1,435 development rows), the ensemble OOF predictions and the hybrid's 1,393;
- the v2 and T2 training configs name this file as the frozen holdout;
- used once before: `cv_20260919_050926_120849/balanced_test` (v1 model, 19 Sep; IDH AUROC 0.862). That model is not in the paper.
- **but** the exploratory v3/v4 runs (20-21 Sep; `holdout_path: None`) put 199 of the 200 into their cross-validation (1,622 patients).
  The paper's v2 members (23 Sep) excluded them again. So no checkpoint, ensemble choice, combiner or threshold used them,
  but they contributed to CV results that compared training settings (e.g. joint vs IDH-only head). Disclose this; it is
  indirect and weak, but the reserved test is not pristine.

So the reserved set is a valid one-time test of the frozen pipeline: `IDH_Reserved_Test.ipynb` (results in
`outputs/idh_reserved_test/` once run). 190 of the 200 have complete features and age (92 mutant).

## 4. Joint IDH/MGMT head vs IDH-only head

Matched runs (same single cross-attention architecture, tumour crop, seed 42, same 1,622 patients and folds):

| Run | Joint: IDH AUROC | IDH-only: IDH AUROC | IDH-only minus joint (95% CI) |
|---|---|---|---|
| v3 (`v3_mask_only_01`) | 0.865 | 0.864 | -0.001 (-0.023 to +0.019) |
| v4 (`v4_mask_only_tmax25_01`, short cosine schedule) | 0.888 | 0.829 | -0.059 (-0.088 to -0.031) |

MGMT-only (v3) gave MGMT AUROC 0.597 vs 0.560 joint (950 patients). The joint head costs IDH nothing and in one setting
helped, which justifies keeping it. Pooled OOF AUROCs here are lower than the paper's members because these are single
earlier-recipe runs. A fresh IDH-only run on the v2 recipe is possible (about 4 h GPU for 5 folds) but is not needed for the paper.
