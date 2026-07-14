# E2E-VarNet MRI Reconstruction — Complete Project Notes (FINAL, POST-REANALYSIS)
### Ali Raza | For Interview Prep, Portfolio, and PhD Outreach
### Last updated: July 2026 — supersedes all earlier drafts and both addenda

---

## WHAT THIS PROJECT DOES (One Line)

Reproduces a published deep learning method (E2E-VarNet) that reconstructs clear MRI images from partially-collected scan data, then rigorously measures three extensions: a checkpoint-based uncertainty/drift signal (the project's strongest result), a cascade-depth-vs-cross-domain-robustness study (an honestly near-null finding), and a self-supervised test-time adaptation experiment (a controlled negative result). Every statistical claim below has been independently stress-tested (volume-level clustering, bootstrap confidence intervals, trivial-baseline controls, and a positive control) across two rounds of internal review.

## WHY WE BUILT IT

MRI scans normally take 30-60 minutes because the scanner has to collect every frequency line of raw data (k-space). Collecting only 25% of that data cuts scan time to under 10 minutes but produces a blurry, artifact-heavy image. Deep learning models can reconstruct a clean image from that partial data. E2E-VarNet (Sriram et al., MICCAI 2020) is a well-established method for this. This project reproduces it honestly (not a novel architecture) and adds three small, measured extensions.

## TECH STACK

PyTorch, official `facebookresearch/fastMRI` package (pinned commit `91f2df4711adbb6d643df1810f234e4abcf5881b`), Kaggle free tier + rented Vast.ai GPUs (RTX 3090/4090), h5py, numpy/scipy (pearsonr, spearmanr, mannwhitneyu, bootstrap), matplotlib. MLflow/FastAPI/Docker/Evidently/Prometheus/Grafana are the next, not-yet-built phase.

---

## HEADLINE RESULT — LEAD WITH THIS

**Checkpoint-ensemble uncertainty magnitude reliably rises under knee-to-brain distribution shift: 1.54x, bootstrap 95% CI [1.35x, 1.74x], survives volume-level clustering correction, and is computable on 100% of tested volumes (unlike a competing eSNR signal, which is numerically stronger but only computable on 12/15 brain volumes).** This is the project's strongest, most production-relevant, most statistically robust finding, and it is the one to lead with in any PhD email, interview, or portfolio writeup. Everything else in this document is secondary to this result.

---

## OFFICIAL, LOCKED NUMBERS (do not use any other SSIM figures anywhere)

Two things caused earlier drafts of this project to report inconsistent SSIM numbers (0.7637 vs 0.7594 for the same T=4 model): (1) a pooled-slice-mean vs volume-level-mean averaging discrepancy, and (2) the fastMRI mask-generation function's `seed` parameter was never explicitly set, so re-evaluating in a fresh session silently produces a slightly different random undersampling mask each time. **Resolution: volume-level mean SSIM, computed in a single session, is the official convention from here forward.**

| Model | Official SSIM (knee, in-domain, volume-level mean) | Official SSIM (brain, OOD, volume-level mean) |
|---|---|---|
| Zero-filled baseline | 0.7453 | 0.4153 |
| T=4 | 0.7594 | 0.6622 |
| T=6 | 0.7606 | 0.6705 |
| T=8 | 0.7607 | 0.6773 |

Any document, CV bullet, or email still citing 0.7637 is out of date and should be corrected to 0.7594.

---

## TRAINING RUNS — FULL RECORD

### T=4 (baseline reproduction)
Config: `num_cascades=4, chans=18, sens_chans=8`, Adam lr=3e-4, loss=1-SSIM, batch_size=1 (structural requirement — see Bugs), grad accumulation=8, EquispacedMaskFractionFunc, center_fractions=0.08, acceleration=4x. 198 train volumes (7,066 slices), 39 val volumes (1,410 slices) — 2 train + 1 val volume removed as corrupt during extraction. Trained 50 epochs across two phases (Kaggle then Vast.ai RTX 3090), surviving 3+ session disconnects via full checkpoint-resume.

### T=6 and T=8
Same config, `num_cascades=6` and `8` respectively. Trained on a rented Vast.ai RTX 4090/3090. Both showed the same convergence pattern as T=4 (plateau by ~epoch 20-22); manually stopped early (T=6 at epoch 25, T=8 at epoch 24) once the plateau was confirmed stable, to conserve budget. This means T=4 received a fuller 50-epoch budget while T=6/T=8 received 24-25 epochs — an acknowledged, minor budget asymmetry, though all three had clearly plateaued before stopping.

### T=2 — deliberately skipped
Dropped from scope for budget/time reasons. GAP 1's cascade sweep covers T=4, T=6, T=8 only; this is stated honestly rather than presented as a complete T=2-through-T=8 sweep.

---

## GAP 1 — Cascade Depth vs Cross-Domain Robustness: A NEAR-NULL FINDING (deliberately deflated, do not oversell)

**Original hypothesis**: deeper cascades might show a joint pattern across in-domain accuracy and out-of-domain (brain) robustness that no published paper had measured.

**What was actually found, after full multi-metric and statistical stress-testing**:

| Model | Knee SSIM | Brain SSIM | Brain PSNR | Brain NMSE | Brain FG-SSIM (foreground-only) |
|---|---|---|---|---|---|
| T=4 | 0.7594 | 0.6622 | 25.07 dB | 0.09096 | 0.8804 |
| T=6 | 0.7606 | 0.6705 | 25.11 dB | 0.09114 | 0.8802 |
| T=8 | 0.7607 | 0.6773 | 25.13 dB | 0.09089 | 0.8820 |

A paired, volume-level bootstrap confirmed the whole-image brain-SSIM trend is statistically real (T8 vs T4: mean diff +0.0150, 95% CI [0.0076, 0.0232], Wilcoxon p=7.6×10⁻⁴; all three pairwise comparisons T4→T6, T6→T8, T4→T8 exclude zero). However, PSNR, NMSE, and — critically — **foreground-masked SSIM (i.e. SSIM computed on the actual anatomical region only) are all flat with cascade depth.** If depth were producing a genuine reconstruction-quality improvement, it should appear in the anatomy (FG-SSIM) and in direct pixel-fidelity metrics (PSNR/NMSE). It does not.

**Final, honest conclusion**: cascade depth has a **near-null effect on out-of-distribution reconstruction quality**. A small, real, statistically-confirmed trend exists in whole-image SSIM specifically, most plausibly reflecting SSIM's known sensitivity to background/structural agreement patterns rather than a clinically meaningful robustness improvement. The originally-proposed mechanistic explanation (deeper cascades anchor more strongly to measurements via repeated data-consistency steps, therefore generalize better) is **withdrawn**, because it predicts a PSNR/NMSE improvement that was not observed. This robustness check was also repeated with edge-of-volume, non-anatomical slices excluded (see Edge-Slice Effect below) and the flat multi-metric pattern held. **This is not the project's headline result and should not be presented as one.**

---

## GAP 2 — Checkpoint-Ensemble Uncertainty: THE STRONGEST, MOST DEFENSIBLE RESULT

**Method**: a K=2 checkpoint ensemble (T=4's epoch-21 and epoch-50 checkpoints — the ideal K=4 spaced subset, epochs 21/30/40/50, was not available because intermediate checkpoints did not survive a Kaggle-to-Vast.ai compute migration; this is stated as an explicit limitation).

### Error-prediction (in-domain, knee), volume-clustered bootstrap, n=39 volumes

| Signal | Bootstrap r | 95% CI |
|---|---|---|
| **zf_residual (zero-filled vs prediction residual)** | **0.7151** | **[0.5271, 0.8446]** |
| ckpt_unc (checkpoint-ensemble) | 0.4317 | [0.3588, 0.5027] |
| periphery_ratio (k-space complexity) | 0.1581 | [0.0335, 0.2727] |

A simpler, free proxy (`zf_residual`) beats checkpoint-ensemble uncertainty for error prediction, with non-overlapping confidence intervals. **This was reported honestly rather than hidden.** Robustness check: recomputed on anatomical-only slices (excluding edge slices), r=0.7652 (p=1.4×10⁻⁸) — essentially unchanged from the full-data volume-level r=0.7777, confirming the correlation is not an edge-slice artifact.

### Distribution-shift detection (brain vs knee), volume-level, n=39 knee / n=15 brain (except eSNR, n=12 brain)

| Signal | Shift ratio | p-value | Coverage |
|---|---|---|---|
| eSNR (estimated SNR proxy) | 2.04x | 7.78×10⁻⁷ | 39/39 knee, 12/15 brain |
| **ckpt_unc (checkpoint-ensemble)** | **1.54x** | **4.46×10⁻⁶** | **39/39 knee, 15/15 brain (full)** |
| zf_residual | 1.17x | 1.60×10⁻² | 39/39 knee, 15/15 brain |
| KER (k-space energy ratio) | 0.59x | 0.082 (not significant) | 39/39 knee, 15/15 brain |

eSNR shows a numerically stronger shift, but could only be computed on 12 of 15 brain volumes (3 excluded — insufficient acquired peripheral k-space lines under 4x undersampling to estimate a stable noise floor; this was a real bug in the first implementation attempt, fixed by restricting the estimate to non-zero/actually-acquired values only). **Checkpoint-ensemble uncertainty is reported as a robust, universally-computable drift signal — not as the single best signal tested.** A production system would reasonably combine eSNR (where available) with checkpoint-ensemble uncertainty (as a reliable fallback) rather than choosing one exclusively.

**Final, honest GAP 2 headline**: no single free signal dominates across both use cases. `zf_residual` is the better error-prediction proxy; checkpoint-ensemble uncertainty and eSNR are both strong shift-detectors, with checkpoint-ensemble uncertainty offering full coverage where eSNR does not. **The 1.54x/CI[1.35,1.74] shift-detection result, specifically, is this project's strongest and most reportable finding.**

---

## GAP 3 — Self-Supervised LoRA Test-Time Adaptation: A CONTROLLED NEGATIVE RESULT

**Method**: manually implemented rank-4 LoRA adapters on T=8's Conv2d layers, adapted using 20 unlabeled slices with an SSDU-style self-supervised loss (ACS-protected hidden-line-prediction), evaluated on a held-out set never touched during adaptation.

**Original (brain/OOD) result**: monotonic SSIM decline at both tested learning rates (0.6779→0.6337 at lr=1e-3; 0.6779→0.6756 at lr=1e-4).

**Positive control (knee/in-domain), added specifically to rule out a domain-shift explanation**: the identical procedure applied knee-to-knee reproduced the same monotonic decline pattern (0.7741→0.7659 at lr=1e-3; flat at lr=1e-4). **This rules out domain shift as the cause.**

**Final, honest conclusion**: SSDU-style self-supervised LoRA adaptation, with this proxy objective, does not improve — and mildly harms — reconstruction quality on this already-converged VarNet, **independent of domain shift**. This is a property of the proxy task/method combination for this model, not a domain-shift-specific finding. The earlier citation to FINE's "over-smoothing" behavior as an explanation was withdrawn — over-smoothing is a blurring failure mode distinct from the monotonic collapse observed here, and citing it would have been an imprecise analogy stretched to fit a convenient narrative.

**Scope caveat (important, always state alongside this result)**: this negative result applies to an already-decent-baseline regime (T=8 achieves 0.678 SSIM on brain even before adaptation). SSDU-style TTA methods in the literature are typically designed for and evaluated on severely-mismatched-baseline scenarios (e.g. a genuinely unfamiliar new scanner with poor initial performance) — that harder, more clinically realistic regime was not tested here, and this result does not rule out TTA helping in that regime.

---

## EDGE-SLICE EFFECT (a genuine, unplanned methodological finding)

While building a qualitative before/after comparison figure, ranking all 1410 knee validation slices by SSIM revealed the bottom 10% (141 slices, mean SSIM=0.5541 vs the overall mean of 0.7594-0.7607) are systematically non-anatomical edge-of-volume slices (`slice_num` in {0,1,2}) — pure background noise, in both the model's output and the ground truth, not a reconstruction failure on real anatomy. This is expected fastMRI dataset structure. Both the GAP 1 depth-trend and the GAP 2 error-correlation were confirmed to hold essentially unchanged when these edge slices are excluded, so this effect does not undermine either result, but it is documented because it explains an otherwise-confusing "why is the worst-case example just noise" question, and because it is a genuine, honestly-reported limitation of whole-image SSIM as an evaluation metric on this dataset.

---

## ERRORS WE HIT AND HOW WE SOLVED THEM (numbered, chronological — real debugging record)

1. **h5py memory leak, unfixable by version pinning on Python 3.12** — no compatible old-HDF5 wheel exists for cp312; fixed via code-level mitigation (`.copy()` + `gc.collect()` per volume, `persistent_workers=False`).
2. **`%%bash` cell magic required** for multi-line bash in Kaggle notebooks — separate `!`-prefixed lines don't share state.
3. **Extraction scripts miscounted files** at the wrong path (non-recursive `ls`/`glob` vs tar's internal subdirectories) — once filled an entire 19.5GB Kaggle disk quota before being caught; fixed with recursive `find` plus hard safety caps (iteration timeout + disk-usage check).
4. **Kaggle Dataset 20GB commit limit** — split the 200-volume knee subset across two Kaggle Datasets, combined via symlinks at training time.
5. **Corrupt/truncated files** from interrupted extractions — standard corruption-check (open + touch actual data arrays, not just the file handle) re-run after every fresh extraction.
6. **Kaggle session disconnects wiped `/kaggle/working`** repeatedly — full checkpoint-resume training loop (model + optimizer state + last epoch) was essential; had to be used multiple times for real.
7. **VarNet requires an explicit coil dimension** even for single-coil data — `unsqueeze(0).unsqueeze(0)` for both batch and coil dims.
8. **VarNet output resolution mismatch** with center-cropped target — fixed via `center_crop()` using the provided `crop_size`.
9. **`batch_size > 1` crashes** — fastMRI k-space has variable matrix widths per scan (368 vs 372); confirmed as a structural requirement, not a bug; used gradient accumulation instead.
10. **CPU/GPU device-mismatch errors** in loss computation — explicit `.to(device)` on every tensor immediately before loss.
11. **VarNet's data-consistency step requires a boolean mask, not float** — `.bool()` cast added universally after being caught on the manually-built brain pipeline.
12. **Naive multi-coil-to-single-coil combination destroyed the brain OOD data** (most consequential bug in the project) — a naive unweighted coil sum caused destructive phase cancellation (brain SSIM ~0.32, and critically, the *zero-filled, no-model* baseline was equally broken at 0.15, which was the key diagnostic clue). Fixed with SVD-based, phase-aligned coil compression. A second, separate bug (missing intensity normalization, since the manual brain pipeline never applied fastMRI's automatic scaling) was found via the same zero-baseline check and fixed by rescaling each slice to match its own ground-truth peak intensity. **Lesson: always sanity-check a zero-model baseline first when building a new data pipeline — it is the fastest way to distinguish "real finding" from "broken pipeline."**
13. **Spurious negative correlation in GAP 2** (r=-0.46, wrong direction) — caused by comparing an unnormalized raw-pixel uncertainty scalar against a scale-invariant SSIM-based error metric; brighter slices produced larger raw uncertainty regardless of true agreement. Fixed by normalizing uncertainty by each slice's own max value, flipping the correlation to the expected positive direction (r=0.43).
14. **GAP 3 TTA loss printed as exactly 0.000000 every epoch** — a display-precision artifact (k-space-scale losses land around 1e-10 to 1e-12, rounding to zero at 6 decimal places), not a real bug; confirmed via scientific-notation printing.
15. **GAP 3 TTA had zero measurable effect on output despite "converging" loss** — Adam's `epsilon=1e-8` term dominated the update step at such small loss magnitudes, silently shrinking every gradient step to ~1/100th its intended size; fixed by rescaling the loss before computing MSE.
16. **GAP 3 SSDU held-out-line selection caused NaN** when it randomly included the ACS (center) region, destabilizing VarNet's sensitivity-map estimator — fixed by explicitly excluding ACS lines from the held-out candidate pool.
17. **Pseudoreplication in every statistical claim** (Round 1 review) — slice-level p-values (n=1410) were invalid because slices within a volume are correlated; fixed by re-analyzing everything at the volume level with block bootstrap over volumes.
18. **Inconsistent official SSIM numbers across documents** (0.7637 vs 0.7594) — traced to (a) pooled-slice vs volume-level averaging convention, and (b) the mask-generation function's unset `seed` parameter causing run-to-run mask non-determinism; resolved by locking volume-level mean SSIM, computed in a single session, as the sole official convention.
19. **eSNR drift-signal proxy produced an absurd ~1.3 million x shift ratio** — caused by computing a noise-floor standard deviation over a mostly-all-zero masked (undersampled) peripheral k-space region; fixed by restricting the estimate to actually-acquired (non-zero) peripheral values only, excluding volumes with too few such values for a stable estimate.

---

## WHAT I LEARNED FROM THIS PROJECT

- How to reproduce a published deep learning paper end-to-end from a partially-maintained, archived codebase, including working around a real, unfixable-by-textbook dependency bug.
- How to build resilient, resumable training infrastructure across two cloud platforms, surviving repeated real session interruptions.
- **How to diagnose a broken data pipeline using a zero-model baseline check as the first diagnostic step** — this single technique correctly localized two separate serious bugs (coil-combination phase cancellation, missing normalization) that would otherwise have looked like an alarming, genuine domain-shift finding.
- How normalization mismatches can silently invalidate an entire experiment's statistics, and that a "too strong" or wrong-direction result should first be suspected of a scale/normalization bug, not accepted as a real finding.
- **How pseudoreplication silently inflates statistical significance in any per-slice medical-imaging analysis**, and how to properly correct for it via volume-level clustering and block bootstrap — this is a genuinely advanced statistical skill most portfolio projects never encounter, let alone catch and fix.
- The discipline of running a positive control (same method, in-domain) before accepting a negative result as domain-shift-specific, rather than accepting the first plausible-sounding explanation.
- How Adam's epsilon term can silently neuter gradient updates on physically tiny quantities, and why rescaling a loss (without changing its minimum) is a legitimate fix.
- How to scope a project deliberately: cutting T=2, cutting a fake human-in-the-loop review queue, cutting a dynamic model-router — and, in this final phase, **deflating my own two most exciting-sounding results (GAP 1's robustness claim, GAP 2's "best signal" claim) once the data required it**, rather than keeping the more impressive-sounding original framing.

## SKILLS I CAN CLAIM AFTER THIS PROJECT

- Reproducing a research paper's method from its official codebase, including resolving real environment/dependency incompatibilities with no documented fix.
- Building and debugging a multi-stage MRI data pipeline (raw k-space, undersampling, coil combination, normalization) from first principles.
- Designing and running honestly-scoped research extensions with proper statistical rigor: volume-level clustering, bootstrap confidence intervals, trivial-baseline controls, and positive controls — and revising or withdrawing claims when the evidence required it.
- Building resumable, checkpoint-based training infrastructure across multiple cloud platforms.
- Diagnosing subtle numerical bugs (scale mismatches, optimizer-epsilon dominance, pseudoreplication, mask non-determinism) using targeted, hypothesis-driven diagnostics.

## WHAT TO WRITE ON CV (Bullet Points)

- Reproduced E2E-VarNet (Sriram et al., MICCAI 2020) for 4x-accelerated single-coil knee MRI reconstruction (SSIM 0.7594-0.7607 across cascade depths, vs 0.7453 zero-filled baseline), and built a checkpoint-ensemble uncertainty signal that reliably detects knee-to-brain distribution shift (1.54x magnitude increase, bootstrap 95% CI [1.35x, 1.74x], volume-level clustering-corrected), directly usable as a production drift-monitor signal.
- Independently identified and corrected pseudoreplication in slice-level statistical claims across a multi-experiment medical-imaging project, re-deriving every reported correlation and significance test at the volume level via block bootstrap, and revised or withdrew two of the project's own headline claims once the corrected statistics no longer supported them.
- Diagnosed and fixed five independent numerical/statistical bugs (coil-combination phase cancellation, a normalization-driven sign-flip in a correlation result, an optimizer-epsilon-driven silent training failure, a near-zero-denominator SNR-proxy blowup, and mask-seed non-determinism causing inconsistent metric reporting) using a zero-baseline-first, hypothesis-driven debugging methodology.

## WHAT TO WRITE IN PORTFOLIO / LINKEDIN POST

I reproduced E2E-VarNet, a published method for reconstructing MRI scans from partial data, then went further: I trained three model depths, tested them on a different body part than they were trained on, and built a free uncertainty signal from checkpoints I already had. The most useful part of this project wasn't the model — it was catching my own mistakes. Two of my headline findings didn't survive a proper statistical review (pseudoreplication, missing controls), and I revised both rather than keep the more exciting-sounding version. What's left is smaller but everything left is real: a checkpoint-disagreement signal that reliably detects when a scan looks unfamiliar to the model, with a confidence interval that survives every stress test I threw at it.

## INTERVIEW QUESTIONS — LIKELY TO BE ASKED

**Q: What's your main result?**
A: A free, checkpoint-based uncertainty signal that reliably detects distribution shift — 1.54x higher on out-of-domain brain data than in-domain knee data, with a bootstrap confidence interval [1.35x, 1.74x] that holds up even after correcting for the fact that MRI slices from the same scan aren't statistically independent. It's directly usable as a drift-monitor signal in production.

**Q: What was the hardest part?**
A: Two things, both about not trusting my own numbers too fast. First, diagnosing a coil-combination bug in synthetic brain data — the initial result looked like a dramatic domain-shift finding, and checking the zero-model baseline (which was equally broken) is what revealed it was a pipeline bug. Second, realizing my per-slice statistics were pseudoreplicated — slices from the same scan aren't independent samples — which meant re-deriving every p-value and confidence interval at the volume level, and watching two of my strongest claims get weaker (one nearly disappeared) once I did it properly.

**Q: What would you do differently?**
A: Preserve all intermediate checkpoints across compute migrations (I lost epochs 30/40, leaving only a K=2 instead of K=4 ensemble). Set an explicit random seed for the mask-generation function from day one — not doing so caused two different sessions to report slightly different SSIM for the same trained model, which took real effort to track down and reconcile.

**Q: What are the limitations?**
A: Single-coil only. Cascade sweep missing T=2. K=2 instead of K=4 checkpoint ensemble. The cross-domain (knee-to-brain) comparison is confounded by two different coil-combination methods (fastMRI's official method for knee, my own SVD-based method for brain), which I can't currently separate from the anatomy shift itself. The cascade-depth-vs-robustness finding, once checked against PSNR/NMSE/foreground-SSIM, turned out to be a near-null result rather than the meaningful trend I initially reported.

## THINGS TO KEEP IN MIND AS SKILLS (concepts to explain confidently)

- Unrolled/variational networks: alternating data-consistency and learned-refinement steps.
- k-space and undersampling; why the ACS/center region must always stay fully sampled.
- Checkpoint ensembles (Chen et al.) vs snapshot ensembles (Huang et al., requires cyclic LR) — never conflate.
- Why pseudoreplication happens in per-slice medical-imaging statistics and how block bootstrap over the true independent unit (the volume/patient/scan) fixes it.
- SSDU-style self-supervised training: splitting already-acquired k-space into input and held-out-loss subsets, no ground truth needed.
- Why Adam's epsilon term matters at small gradient magnitudes.
- Why a positive control (same method, known-good setting) is necessary before attributing a negative result to the variable you actually wanted to test (domain shift, in this case).
- Why whole-image SSIM can diverge from foreground-only SSIM and pixel-fidelity metrics (PSNR/NMSE), and why that divergence should make you suspicious of a whole-image-only result.

---
*This document is the final, single source of truth for this project's results. Both prior statistical re-analysis addenda (Round 1 and Round 2) are fully incorporated above and do not need to be consulted separately.*
