# E1b. Shape control for the occlusion analysis (29 Oct 2026)

Transcribed from the executed notebook `code/29oct/E1b_Shape_Control_29Oct.ipynb`, which ran on Tara's machine with an RTX 5080 (51.8 min). The helper module `isbi_e1b_shape_control.py` and the per-patient CSVs are on Tara's machine and are not in this package.

**Reproduction checks (passed):** original vs saved OOF, max |diff| 0.0006 (0 of 4,296 above 0.01); brain_only vs E1, max |diff| 0.0001.

**Pre-declared verdict:** R3, mixed. Geometry explains part of the brain-only signal and tissue explains part of it. The two silhouette constants disagree by more than 0.05, so the silhouette result is fill-sensitive.

| Score (1,393 patients, 186 mutant) | AUROC (95% CI) | Age-matched |
|---|---|---|
| original | 0.926 (0.904–0.945) | 0.900 |
| tumour_ring_only (E1) | 0.859 (0.827–0.888) | 0.838 |
| tumour_silhouette | 0.525 (0.481–0.568) | 0.500 |
| brain_only (lesion + margin occluded, background fill) | 0.677 (0.636–0.716) | 0.668 |
| brain_only_inpainted | 0.678 (0.637–0.716) | 0.650 |
| silhouette (+1 SD) | 0.580 (0.542–0.620) | 0.571 |
| silhouette_neg (−1 SD) | 0.523 (0.477–0.566) | 0.554 |
| Lesion geometry, 11 features (CV) | 0.792 (0.756–0.825) | 0.726 |
| Geometry + brain_only logit (CV) | 0.807 (0.771–0.839) | 0.759 |
| Geometry + silhouette + brain_only logits (CV) | 0.806 (0.770–0.839) | 0.759 |

| Paired difference | Δ AUROC (95% CI) |
|---|---|
| silhouette − brain_only | −0.098 (−0.136 to −0.057) |
| inpainted − brain_only | +0.001 (−0.032 to +0.034) |
| inpainted − silhouette | +0.098 (+0.051 to +0.143) |
| tumour_silhouette − tumour_ring_only | −0.333 (−0.377 to −0.291) |
| geometry + silhouette + brain_only − geometry + silhouette | +0.014 (+0.003 to +0.024) |
| brain_only − lesion geometry | −0.114 (−0.161 to −0.070) |

CV R² of the brain_only logit from geometry: 0.19. Median hole share of the brain in the crop: 0.30.

**In the paper (v8):**
- **Methods:** one sentence on the shape controls.
- **Results:** tumour silhouette 0.525; occluded-region silhouette 0.580 (0.523 with the opposite sign); inpainted 0.678; geometry 0.792; tumour-occluded output +0.014 (0.003–0.024) beyond geometry and silhouette.
- **Discussion:** within the tumour and margin, the signal is image content rather than outline. The signal that survives occluding the tumour is not the outline of the hole either, but it overlaps largely with lesion geometry.
- **Limitations:** "inpainting leaves mass effect" replaces "fills still convey lesion size, shape and location".
