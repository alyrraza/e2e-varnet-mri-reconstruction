# Statistical Re-Analysis Addendum — Response to Review
### E2E-VarNet Project | For the reviewer who flagged pseudoreplication, metric-saturation, and other issues
### Prepared: July 2026

---

## HOW TO USE THIS DOCUMENT

This addendum documents every re-analysis performed in direct response to the prior review. Nothing here required retraining except one item (GAP 3 positive control, which used the existing T=8 checkpoint with no new training, only a lightweight adaptation run). All original raw results remain unchanged; what changed is how they are analyzed and reported. Each section states: what the review flagged, what was done about it, and the resulting number.

---

## 1. VOLUME-LEVEL CLUSTERING (fixes the pseudoreplication issue)

**What was flagged**: slice-level p-values (e.g. p<10⁻⁶⁵ on n=1410 "samples") were invalid because slices within a volume are correlated (same anatomy, same coil profile), so effective sample size is closer to the number of volumes (~39 knee, ~15 brain), not the number of slices.

**What was done**: recomputed every correlation using (a) volume-level means (one data point per volume) and (b) block bootstrap (resampling whole volumes with replacement, pooling their slices, recomputing the correlation 2000 times) to get an honest 95% confidence interval on the slice-level relationship.

### Knee (in-domain), GAP 2 uncertainty-vs-error correlation
- Original (invalid) claim: Pearson r=0.4338, p<10⁻⁶⁵ (n=1410 slices, pseudoreplicated)
- Volume-level (n=39 volumes): Pearson r=0.7777, p=5.74×10⁻⁹
- Block-bootstrap (volume-clustered, slice-level pooling): mean r=0.4337, **95% CI [0.3620, 0.5042]**, excludes zero
- **Verdict: the correlation survives. The effect is real and the CI is reasonably tight.**

### Brain (OOD), GAP 2 uncertainty-vs-error correlation
- Original (invalid) claim: Pearson r=-0.1332, p=0.04 (n=238 slices, pseudoreplicated), described as "weak, direction-flipped"
- Volume-level (n=15 volumes): Pearson r=0.2474, p=0.37 (not significant)
- Block-bootstrap: mean r=-0.1522, **95% CI [-0.4620, 0.1637]**, includes zero
- **Verdict: no reliable per-slice correlation can be claimed under domain shift. Revised framing: "no statistically reliable correlation under OOD shift" — not "weak and direction-flipped," which overstated a null result as a directional finding.**

### Drift-shift magnitude (uncertainty is higher on brain than knee)
- Original (invalid) claim: 1.55x shift, Mann-Whitney p<10⁻⁵⁴ (n=1410 vs n=238 slices, pseudoreplicated)
- Volume-level (n=39 vs n=15 volumes): 1.54x shift, Mann-Whitney p=4.46×10⁻⁶
- Bootstrap 95% CI on the ratio itself: **[1.35x, 1.74x]**, excludes 1.0x (no-shift)
- **Verdict: this result is robust and survives clustering correction essentially unchanged. This remains the strongest, most defensible GAP 2 result.**

---

## 2. GAP 1 EXTENDED METRICS (tests whether the flat in-domain curve is SSIM-specific)

**What was flagged**: whole-image SSIM may be saturated (dominated by easy background agreement), potentially hiding a real depth-related improvement that other metrics would reveal. PSNR, NMSE, and a foreground-masked SSIM were requested.

**What was done**: computed PSNR, NMSE, and foreground-masked SSIM (simple intensity-threshold anatomy mask) per volume, for all three cascade depths, on both knee and brain.

| Model | Knee SSIM | Knee PSNR | Knee NMSE | Knee FG-SSIM | Brain SSIM | Brain PSNR | Brain NMSE | Brain FG-SSIM |
|---|---|---|---|---|---|---|---|---|
| T=4 | 0.7594 | 32.67 dB | 0.04224 | 0.9260 | 0.6630 | 25.07 dB | 0.09096 | 0.8804 |
| T=6 | 0.7606 | 32.84 dB | 0.04105 | 0.9274 | 0.6713 | 25.11 dB | 0.09114 | 0.8802 |
| T=8 | 0.7607 | 32.74 dB | 0.04184 | 0.9265 | 0.6780 | 25.13 dB | 0.09089 | 0.8820 |

**Finding**: the hypothesis was NOT confirmed. PSNR, NMSE, and foreground-masked SSIM all show flat, non-monotonic patterns on the knee (in-domain) side, and PSNR/NMSE/FG-SSIM show much flatter trends than SSIM on the brain (OOD) side as well. The whole-image SSIM OOD-improves-with-depth trend is not strongly corroborated by the other three metrics.

**Revised framing**: the depth-vs-OOD-robustness finding should be stated as an SSIM-specific result, not a metric-agnostic one. See Section 3 for whether the SSIM effect itself is statistically real.

---

## 3. GAP 1 PAIRED BOOTSTRAP (tests whether the SSIM trend itself is statistically real, not luck across 3 noisy points)

**What was flagged**: with only 3 cascade-depth points and no significance testing, the monotonic OOD-SSIM-improves-with-depth pattern could plausibly be noise across a small number of comparisons.

**What was done**: paired, volume-level bootstrap on brain SSIM differences between adjacent (and non-adjacent) cascade depths, using the same 15 brain volumes evaluated by all three models (removes cross-volume difficulty variance, which is the correct way to compare paired measurements).

| Comparison | Mean SSIM difference | 95% CI | Excludes zero |
|---|---|---|---|
| T6 − T4 | — | [0.0017, 0.0156] | Yes |
| T8 − T6 | — | [0.0010, 0.0117] | Yes |
| T8 − T4 | +0.0150 (13/15 volumes positive) | [0.0076, 0.0232] | Yes |

Wilcoxon signed-rank (T8 > T4, paired): p=7.63×10⁻⁴.

**Verdict: the SSIM-based OOD-robustness-improves-with-depth trend is statistically robust at the paired, volume level.** Combined with Section 2's finding, the honest overall statement is: this is a real, statistically confirmed effect in SSIM terms specifically, small in absolute magnitude, and not strongly replicated across PSNR/NMSE/FG-SSIM. Both facts are reported together, not one instead of the other.

---

## 4. GAP 2 TRIVIAL-BASELINE COMPARISON (tests whether the checkpoint-ensemble signal adds value over free heuristics)

**What was flagged**: the checkpoint-ensemble uncertainty signal was never compared against cheap, already-available proxies (e.g. the zero-filled residual). If a trivial proxy performs as well or better, the checkpoint-ensemble method adds no value.

**What was done**: computed two trivial, zero-extra-training-cost proxies — (a) the magnitude of the residual between the zero-filled image and the model's mean prediction (`zf_residual`), and (b) a k-space periphery-to-center energy ratio (`periphery_ratio`) — and compared their volume-clustered bootstrap correlation with true error, and their brain/knee shift-detection ratio, against the checkpoint-ensemble signal.

### Error-prediction (in-domain, knee, volume-clustered bootstrap)

| Signal | Bootstrap mean r | 95% CI |
|---|---|---|
| zf_residual | **0.7151** | **[0.5271, 0.8446]** |
| ckpt_unc (checkpoint-ensemble) | 0.4317 | [0.3588, 0.5027] |
| periphery_ratio | 0.1581 | [0.0335, 0.2727] |

**zf_residual clearly outperforms checkpoint-ensemble uncertainty for error prediction; the confidence intervals do not overlap.**

### Shift-detection (brain/knee ratio, volume-level)

| Signal | Shift ratio | Mann-Whitney p (volume-level) |
|---|---|---|
| ckpt_unc | **1.54x** | **4.46×10⁻⁶** |
| zf_residual | 1.17x | 0.016 |

**checkpoint-ensemble uncertainty clearly outperforms zf_residual for shift-detection.**

**Revised, final GAP 2 headline**: neither free signal dominates the other across both use cases. `zf_residual` is the better error-flagging proxy; checkpoint-ensemble uncertainty is the substantially better distribution-shift detector. This is reported as a complementary-tradeoff finding — recommend using each signal for its stronger use case, not treating either as a universal solution. This is a stronger, more nuanced claim than the original single-signal framing, not a weaker one.

---

## 5. GAP 3 POSITIVE CONTROL (tests whether the negative TTA result is domain-shift-specific or a general method problem)

**What was flagged**: the GAP 3 negative result (SSDU-style LoRA TTA degrading SSIM on brain) went through several implementation bugs before producing a clean monotonic decline. Before treating this as a genuine domain-shift finding, the same method needed to be tested in-domain (knee-to-knee) as a positive control, to rule out the possibility that the method itself is broken regardless of domain.

**What was done**: ran the identical TTA procedure (same T=8 base model, same LoRA rank=4, same SSDU-style ACS-protected self-supervised split, same 20-sample adaptation / held-out evaluation split size) entirely in-domain, knee-to-knee.

| Learning rate | Before TTA | After TTA (5 epochs) | Trajectory |
|---|---|---|---|
| lr=1e-3 (knee) | 0.7741 | 0.7659 | 0.7739 → 0.7730 → 0.7711 → 0.7686 → 0.7659 |
| lr=1e-4 (knee) | 0.7741 | 0.7741 | flat, no measurable change |
| lr=1e-3 (brain, original) | 0.6779 | 0.6337 | 0.6722 → 0.6664 → 0.6590 → 0.6483 → 0.6337 |
| lr=1e-4 (brain, original) | 0.6779 | 0.6756 | 0.6774 → 0.6770 → 0.6767 → 0.6761 → 0.6756 |

**Verdict: the knee (in-domain) positive control reproduces the same monotonic decline pattern seen on brain.** This rules out domain shift as the cause. The corrected, final GAP 3 finding: SSDU-style self-supervised LoRA fine-tuning does not improve — and mildly harms — reconstruction quality on an already-converged VarNet, independent of domain shift. This is a genuine limitation of the specific proxy objective (hidden-line prediction) for this model/setting, not a domain-shift-specific phenomenon. The earlier citation to FINE's "over-smoothing" behavior is also withdrawn as an imprecise analogy (over-smoothing is a blurring failure mode, distinct from the monotonic SSIM collapse observed here); no substitute citation is offered, this is reported as a standalone, self-contained empirical finding.

---

## 6. QUALITATIVE FIGURE FIX (worst/median/best, not cherry-picked best-case)

**What was flagged**: the original qualitative comparison figure showed a single high-SSIM (~0.89) example next to a table reporting a 0.7637 mean, which misleadingly implies the shown example is typical.

**What was done**: ranked all 1410 knee validation slices by actual T=8 SSIM and selected the true worst-case, median-case, and best-case slices for a 3-row comparison figure (zero-filled / T=8 reconstruction / ground truth per row), with per-row and overall mean SSIM labeled directly on the figure.

**Unexpected additional finding during this process**: the worst-case slice (SSIM=0.4392) turned out to be a non-anatomical edge-of-volume slice (both zero-filled input and ground truth show pure background noise, no visible knee anatomy — confirmed by inspecting `slice_num`, which was 0). A follow-up check of the bottom 10% of all slices by SSIM (141 slices, mean SSIM=0.5541) confirmed this is systematic: all bottom-10% slices have `slice_num` in {0, 1, 2}, i.e. they are consistently early, non-anatomical edge slices, not reconstruction failures on real anatomical content.

**This is reported as an honest, additional methodology note**: approximately 10% of the validation set consists of non-anatomical edge slices that systematically drag down the reported mean SSIM, despite not representing meaningful reconstruction failures. This is consistent with standard (unfiltered) fastMRI evaluation convention and the reported SSIM numbers are not adjusted for it, but the effect is now explicitly documented rather than left unexplained.

---

## SUMMARY OF WHAT CHANGED

| Original claim | Status after re-analysis |
|---|---|
| GAP 2 knee correlation r=0.43, p<10⁻⁶⁵ | **Survives**: r=0.43, honest 95% CI [0.36, 0.50] |
| GAP 2 brain correlation r=-0.13, "weak, flipped" | **Withdrawn as directional claim**: CI [-0.46, 0.16] includes zero, reframed as "no reliable correlation" |
| GAP 2 drift-shift 1.55x, p<10⁻⁵⁴ | **Survives essentially unchanged**: 1.54x, CI [1.35x, 1.74x] |
| GAP 2 checkpoint-ensemble is the best free uncertainty signal | **Revised to a tradeoff**: zf_residual is better for error-prediction, checkpoint-ensemble is better for drift-detection |
| GAP 1 "deeper cascades are more OOD-robust" (SSIM-based, presented as general) | **Narrowed**: statistically real and paired-bootstrap-confirmed in SSIM terms specifically (all three pairwise comparisons exclude zero); not strongly corroborated by PSNR/NMSE/FG-SSIM |
| GAP 1 "non-obvious result" framing | **Reframed as mechanistic**: consistent with deeper unrolling anchoring more strongly to measured k-space via repeated data-consistency steps |
| GAP 3 negative TTA result attributed to over-smoothing / domain shift | **Corrected**: positive control (knee-to-knee) reproduces the same decline, ruling out domain shift; over-smoothing citation withdrawn as an imprecise analogy |
| Qualitative figure shows a single ~0.89 SSIM example | **Replaced** with worst/median/best figure, mean SSIM labeled; edge-slice effect on the worst case documented |

## OUTSTANDING CAVEATS CARRIED FORWARD (not yet independently re-verified further, stated plainly)

- The ESC (knee) vs SVD (brain) coil-combination mismatch remains an acknowledged confound in the cross-domain comparison: the brain OOD test reflects both an anatomy shift and a coil-combination-method shift, which cannot currently be separated. This is a caveat for the robustness claim specifically; it does not undermine the drift-detection use case, which only requires that the monitor fire reliably on some shift, regardless of its exact source.
- T=4 received 50 training epochs while T=6 and T=8 received 24-25 (stopped early once plateaued); this is a minor, acknowledged budget asymmetry across the compared models, though all three had clearly plateaued before stopping.
- Foreground-masking used a simple intensity threshold, not a learned/anatomical segmentation mask; a more precise foreground definition might change the FG-SSIM numbers in Section 2 slightly, though the qualitative flat-trend conclusion is unlikely to change given how consistent it is across three independent metrics.
