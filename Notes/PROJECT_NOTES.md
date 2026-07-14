# E2E-VarNet MRI Reconstruction — Complete Project Notes
### Ali Raza | For Interview Prep and Portfolio Reference
### Last updated: July 2026

---

## WHAT THIS PROJECT DOES (One Line)

Reproduces a published deep learning method (E2E-VarNet) that reconstructs clear MRI images from partially-collected scan data, then honestly measures three things nobody had measured before: how reconstruction quality trades off against model depth under real cross-anatomy shift, whether free existing model checkpoints can flag their own uncertain outputs, and whether test-time adaptation actually improves accuracy in a data-scarce clinical-shift scenario.

## WHY WE BUILT IT

MRI scans normally take 30-60 minutes because the scanner has to collect every frequency line of raw data (k-space). Collecting only 25% of that data cuts scan time to under 10 minutes but produces a blurry, artifact-heavy image. Deep learning models can reconstruct a clean image from that partial data. E2E-VarNet (Sriram et al., MICCAI 2020) is a well-established method for this. This project reproduces it honestly (not a novel architecture) and adds three small, measured extensions that fill real, verified gaps in the published literature — chosen specifically because they could be answered on a single consumer GPU with a student budget, using assets (checkpoints, undersampling infrastructure) already produced by the reproduction itself.

## TECH STACK

- **Model**: PyTorch, official `facebookresearch/fastMRI` package (pinned commit `91f2df4711adbb6d643df1810f234e4abcf5881b`, installed from source since the repo is archived as of Aug 2025)
- **Dataset**: fastMRI single-coil knee (NYU Data Sharing Agreement), fastMRI brain multi-coil (val split, for OOD test)
- **Training compute**: Kaggle free tier (2x T4 GPUs), Vast.ai rented GPUs (RTX 3090, RTX 4090)
- **Data pipeline**: h5py, custom streaming curl|tar extraction, Kaggle Datasets for persistent storage
- **Data acquisition**: Kaggle API for cross-platform dataset transfer
- **Analysis**: numpy, scipy (pearsonr, spearmanr, mannwhitneyu), matplotlib
- **Version control of experiments**: manual checkpoint-based tracking (MLflow planned, not yet integrated — this is the next deployment-phase step)

---

## FULL PROJECT FLOW (Step by Step)

**Step 1**: Requested and received NYU fastMRI dataset access (Data Sharing Agreement, citation requirements: Knoll et al. 2020, arXiv:1811.08839). Access delivered as AWS S3 signed URLs (90-day expiry) via email.

**Step 2**: Built a subset (200 train + 40 val target, single-coil knee) using streaming `curl | tar` extraction with live disk-usage monitoring — this avoided downloading the full ~72.7GB source archive, extracting only the needed volumes directly from the compressed stream.

**Step 3**: Split the subset across two Kaggle Datasets (168 train volumes in Part 1, 32 train + 40 val in Part 2) because a single Kaggle Dataset commit is capped at 20GB and the full 200-volume set would have exceeded it.

**Step 4**: Set up the fastMRI codebase (pinned commit, editable install) and worked around a version-specific h5py memory leak that could not be fixed via pinning on Python 3.12.

**Step 5**: Trained the T=4 baseline model for 50 epochs total, in two phases split across Kaggle and a rented Vast.ai GPU, with full checkpoint-resume capability to survive multiple session disconnects.

**Step 6**: Trained T=6 and T=8 cascade-depth variants (T=2 was scoped out — see "What We Cut" below) on a rented Vast.ai RTX 4090/3090, stopping each early once validation SSIM visibly plateaued, to conserve budget.

**Step 7**: Downloaded a small fastMRI brain multi-coil validation subset (using the already-approved NYU access), and built a coil-combination pipeline to convert it into a synthetic single-coil OOD test set compatible with the knee-trained models.

**Step 8**: Ran GAP 1 (cascade depth vs cross-domain robustness), GAP 2 (checkpoint-ensemble uncertainty), and GAP 3 (self-supervised LoRA test-time adaptation) — all as inference-only or lightweight adaptation experiments reusing the already-trained checkpoints and brain data.

**Step 9**: Documented every result, including the two experiments (GAP 2 error-correlation and GAP 3 TTA) that produced results different from the initially hoped-for direction, and reported them honestly rather than hiding or re-running until a "better" number appeared.

## HOW THE COMPONENTS ARE LINKED

- The knee dataset (Kaggle) feeds all training runs (T4/T6/T8) and the GAP 1/GAP 2 in-domain evaluation.
- The brain dataset (fastMRI multi-coil val, coil-combined) feeds GAP 1's OOD evaluation, GAP 2's shift test, and GAP 3's adaptation/evaluation split — one dataset, three reused purposes.
- T=4's saved epoch checkpoints (epoch 21 = best, epoch 50 = final) feed GAP 2's checkpoint-ensemble directly; no separate training was needed for GAP 2.
- T=8 (the best-performing OOD model from GAP 1) was selected as the base model for GAP 3's TTA experiment, since it had the strongest starting brain-domain SSIM.

---

## TRAINING RUNS — FULL RECORD

### T=4 (baseline reproduction)
- Config: `num_cascades=4, chans=18, sens_chans=8`, Adam lr=3e-4, loss=1-SSIM, batch_size=1 (structural requirement — see Bugs section), grad accumulation=8, EquispacedMaskFractionFunc, center_fractions=0.08, acceleration=4x.
- 198 train volumes (7,066 slices), 39 val volumes (1,410 slices) — 2 train + 1 val volume removed as corrupt during extraction.
- Zero-filled baseline SSIM: **0.7453**.
- Trained 50 epochs across two phases: epochs 1-10 on Kaggle (2x T4 free tier, ~34 min/epoch), epochs 11-50 resumed and completed on a rented Vast.ai RTX 3090 (~11.2 it/s, ~11.3 min/epoch after fixing a memory-leak-related slowdown — see Bugs).
- **Result: best val SSIM = 0.7637 at epoch 21.** Converged by ~epoch 20; epochs 21-50 fluctuated within 0.7619-0.7637 (no further meaningful gain, no overfitting collapse).
- Survived 3+ separate session disconnects (Kaggle kernel restarts, Vast.ai connection drops) via full checkpoint-resume (model + optimizer state saved every epoch).

### T=6
- Same config, `num_cascades=6`. Trained fresh (not resumed from T=4) on a rented Vast.ai RTX 4090.
- Planned 50 epochs, but validation SSIM plateaued by epoch ~21-22 (0.7649, tied at epoch 22), matching the T=4 pattern almost exactly. Training was manually stopped at epoch 25 to conserve budget once the plateau was confirmed stable across 4-5 epochs.
- **Result: best val SSIM = 0.7649** (epoch 21-22).

### T=8
- Same config, `num_cascades=8`. Reduced epoch budget to 30 from the start (based on the T=6 plateau pattern), on the same Vast.ai RTX 4090/3090 session.
- Per-epoch time was noticeably higher than T=6 (~14.5-14.7 min/epoch vs T=6's 11.3 min), consistent with the added cascade compute.
- Plateaued by epoch 20-22 again (0.7649-0.7650). Manually stopped at epoch 24 (of a planned 30) once the plateau was confirmed, to conserve the remaining ~$1 of GPU budget.
- **Result: best val SSIM = 0.7650** (epoch 22).

### T=2 — Deliberately Skipped
Originally planned as part of the full T=2/4/6/8 sweep, but explicitly dropped from scope partway through the project (budget and time constraints, and diminishing marginal value once the T4→T6→T8 plateau pattern was already clearly established). GAP 1's cascade sweep therefore covers T=4, T=6, T=8 only — this is stated honestly in all results rather than presented as a complete T=2-through-T=8 sweep.

---

## RESULT: GAP 1 — Cascade Depth vs Cross-Domain Robustness

**What it measures**: whether the relationship between cascade depth (model size/depth) and reconstruction quality changes when the input domain shifts from the training distribution (knee) to a different anatomy (brain) — a joint 2D analysis that, as of a July 2026 literature check, no published paper had done for single-coil accelerated MRI reconstruction.

**Final results** (after fixing two significant pipeline bugs — see Bugs section):

| Cascade | Knee SSIM (in-domain) | Brain SSIM (OOD) | OOD Drop | vs zero-filled brain baseline (0.4153) |
|---|---|---|---|---|
| T=4 | 0.7637 | 0.6622 | 0.1015 | +0.2469 |
| T=6 | 0.7649 | 0.6705 | 0.0944 | +0.2552 |
| T=8 | 0.7650 | 0.6773 | 0.0877 | +0.2620 |

**Finding**: in-domain, deeper cascades give only marginal gains (0.7637 → 0.7650, +0.0013 total). Out-of-domain, the gain from deeper cascades is proportionally larger (0.6622 → 0.6773, +0.0151), and the OOD-drop itself shrinks with depth (0.1015 → 0.0877). This means deeper cascades provide more benefit for cross-domain robustness than for in-domain accuracy — a non-obvious result, since it was equally plausible beforehand that deeper cascades might overfit more to the training anatomy and generalize worse.

**Honest limitations**: only 15 brain volumes (238 slices) were used, after coil-combining fastMRI's multi-coil brain data into a synthetic single coil (see Bugs section for the coil-combination issues this created and how they were fixed). T=2 is missing from the sweep (scoped out).

---

## RESULT: GAP 2 — Checkpoint-Ensemble Uncertainty

**What it measures**: whether the disagreement between predictions from different saved checkpoints of a single, ordinary training run (a "checkpoint ensemble" in the Chen, Lundberg, Lee sense, arXiv:1710.03282 — explicitly not a cyclic-LR "snapshot ensemble", Huang et al., arXiv:1704.00109) can serve as a free, zero-training-cost signal for (a) predicting per-slice reconstruction error and (b) detecting distribution shift.

**Honest limitation stated upfront**: the ideal spaced checkpoint subset was epochs 21/30/40/50 (K=4). Only the epoch-21 and epoch-50 checkpoints survived the Kaggle-to-Vast.ai migration (intermediate checkpoints from the middle of the run were not preserved when the compute environment changed). The experiment was run with **K=2** instead of the ideal K=4, and this is expected to weaken the signal relative to what a full K=4 ensemble would show.

**Final results** (after fixing a scale-normalization bug — see Bugs section):

| Test | Result |
|---|---|
| In-domain (knee) correlation | Pearson r = 0.4338, Spearman rho = 0.4595 (p < 10⁻⁶⁵, n=1410) |
| OOD (brain) correlation | Pearson r = -0.1332 (weak, p = 0.04, n=238) |
| Uncertainty shift (brain vs knee magnitude) | 1.55x higher on brain, Mann-Whitney p < 10⁻⁴¹ |

**Finding**: the K=2 checkpoint-ensemble uncertainty signal is a **moderate in-domain error predictor** (r=0.43) but this is materially below what learned/Bayesian methods achieve for the same task (>90% Pearson for conformal quantile regression, arXiv:2601.13236; R=0.94 for NPB-REC's Bayesian sampling, Küstner et al. comparison anchor) — an honest cost-quality tradeoff for a zero-training-cost signal, not a competing method. The correlation **breaks down under domain shift** (r=-0.13, weak and direction-flipped), meaning it should not be trusted as a per-slice error predictor on out-of-distribution data. However, the **absolute magnitude** of the uncertainty signal is a strong, statistically robust **distribution-shift detector** (1.55x higher on brain, p<10⁻⁴¹) even though its per-slice error correlation is unreliable there — meaning it is directly usable as a drift-monitor validation signal in a production pipeline, separate from its (weaker) role as an error predictor.

---

## RESULT: GAP 3 — Self-Supervised Test-Time Adaptation (LoRA)

**What it measures**: whether a small amount of unlabeled target-domain data (simulating "15-20 scans from a new hospital") can be used to adapt an already-trained model at deployment time via lightweight LoRA updates and a self-supervised (SSDU-style) loss — without any labeled ground truth from the new domain.

**Method**: manually implemented low-rank adapters (rank=4) injected into all Conv2d layers of the T=8 model (the strongest OOD baseline from GAP 1). Adaptation used 20 unlabeled brain slices; a held-out set of 218 brain slices (never touched during adaptation) was used for before/after evaluation. The self-supervised loss followed an SSDU-style split: acquired k-space lines were split into a subset shown to the model and a held-out subset used only to compute the loss (predicting the held-out lines from the shown ones), with the ACS (center) region always protected from being held out.

**Final results**:

| Learning rate | Before TTA | After TTA (5 epochs) | Trajectory |
|---|---|---|---|
| lr=1e-3 | 0.6779 | 0.6337 | Monotonic decline every epoch: 0.6722 → 0.6664 → 0.6590 → 0.6483 → 0.6337 |
| lr=1e-4 (10x smaller) | 0.6779 | 0.6756 | Monotonic decline every epoch, but slower: 0.6774 → 0.6770 → 0.6767 → 0.6761 → 0.6756 |

**Finding — an honest negative result**: SSDU-style self-supervised LoRA adaptation produced a **consistent SSIM degradation, not improvement**, at both tested learning rates, monotonically across every epoch tested. This rules out a simple learning-rate/step-size explanation (a 10x smaller LR only slowed the decline, it did not reverse it), pointing instead to a genuine misalignment between the self-supervised proxy objective (predicting held-out acquired k-space lines) and true reconstruction quality in this specific, highly data-scarce (20-sample) regime. This is directly consistent with a documented weakness of similar self-supervised TTA methods in the literature (FINE is explicitly reported as "prone to over-smoothing structural details in highly data-scarce settings"). The closest prior work (AdaptNet, ECCV 2024; D2SA, NeurIPS 2025) both apply TTA to unrolled reconstruction networks successfully in their own settings; this result does not contradict them, but shows the specific SSDU-proxy-task approach did not transfer favorably to this 20-sample, single-model setup without further tuning (e.g. different proxy losses, more adaptation samples, or regularization against the frozen base model, none of which were in scope for this pass).

---

## HOW TO RUN LOCALLY

1. Clone `facebookresearch/fastMRI` at commit `91f2df4711adbb6d643df1810f234e4abcf5881b`, install from source (`pip install -e .`), never via pip's floating release.
2. Set up h5py per the Python-version-specific workaround below (no clean pip-pin fix exists on Python 3.12).
3. Obtain fastMRI single-coil knee (and, for OOD testing, brain multi-coil val) access via the NYU Data Sharing Agreement at fastmri.med.nyu.edu; download links are AWS S3 signed URLs valid for 90 days.
4. Use streaming `curl | tar` extraction with a target-file-count stop condition to avoid downloading full multi-hundred-GB archives when only a subset is needed.
5. Train with `batch_size=1` (see Bugs section — this is not optional).

## HOW IT IS DEPLOYED

Not yet deployed. This is the next phase: MLflow model registry (T=4/T6/T8 as versioned models using aliases, not deprecated MLflow "stages"), FastAPI + Docker serving, Evidently AI drift monitoring (built on k-space scalar features, validated using the same knee-to-brain shift data from GAP 1/GAP 2), a CI/CD quality gate (baseline-relative SSIM threshold, CPU-only inference on GitHub's free runners), and Prometheus/Grafana monitoring. Full design is T=4-first: every component is designed to work with only the T=4 model, with T=6/T=8 as additive, non-blocking bonuses.

---

## ERRORS WE HIT AND HOW WE SOLVED THEM

This section documents every significant bug encountered, in the order they occurred, because this is genuinely useful interview material — it demonstrates real debugging, not just running someone else's tutorial.

### 1. h5py memory leak, unfixable by version pinning (Python 3.12)
**Error/symptom**: repo Issue 215 documents a known h5py memory leak with HDF5 1.12.1+ when converting to torch tensors. The intended fix (pin h5py to an older version bundling HDF5 <1.12.1) failed completely: `conda` was not present in the Kaggle image, and pip-installing old h5py versions (3.1.0, 3.6.0) triggered source builds that failed on missing system headers, because no prebuilt wheel exists for those old versions on Python 3.12 (h5py only gained cp312 wheel support from version 3.10+, which already bundles HDF5 1.14.x — past the leak-affected threshold).
**Fix**: abandoned version-pinning entirely. Used a code-level mitigation instead: explicit `.copy()` after loading HDF5 data plus `gc.collect()` per volume, and `persistent_workers=False` in the DataLoader so worker processes fully respawn each epoch, clearing accumulated leaked memory. This did not eliminate the leak but made it survivable for full training runs.

### 2. `%%bash` cell syntax errors in Kaggle notebooks
**Error/symptom**: multi-line bash functions and loops written as separate `!`-prefixed lines failed, because each `!` line runs as an isolated subprocess — variables and function definitions don't persist across lines.
**Fix**: used the `%%bash` cell magic at the top of the cell to run the entire cell as one continuous bash script.

### 3. Extraction script counted files at the wrong path (twice)
**Error/symptom**: `ls *.h5` / `glob("*.h5")` returned 0 files repeatedly, because the tar archive preserved an internal subdirectory (e.g. `singlecoil_train/`, `multicoil_val/`) that the count logic wasn't checking recursively. In the worst instance, this caused a background `curl | tar` extraction to never detect it had hit its target, running unbounded until it filled the entire 19.5GB Kaggle disk quota.
**Fix**: switched to `find . -type f -name "*.h5"` (recursive) instead of top-level `ls`/`glob`, and added a hard safety cap (both an iteration-count timeout and an explicit disk-usage check) to any future streaming-extraction loop, so a similar bug can never again silently fill the disk.

### 4. Kaggle Dataset 20GB commit limit
**Error/symptom**: the full 200-volume single-coil knee subset (~19-23GB depending on which files) risked exceeding Kaggle's 20GB per-dataset output/commit limit.
**Fix**: split the subset across two separate Kaggle Datasets and combined them at training time via symlinks (or plain copies on a machine with more disk, like Vast.ai) into one working directory.

### 5. Corrupt/truncated files from interrupted extractions
**Error/symptom**: multiple times, a background extraction was killed (either intentionally, on hitting the target count, or unintentionally, via a session disconnect) while a file was mid-write, producing a truncated HDF5 file that failed to open (`OSError: Unable to synchronously open file (truncated file...)`), which then crashed dataset initialization or training partway through.
**Fix**: a standard corruption-check step, re-run after every fresh extraction, that opens every `.h5` file, touches the actual data arrays (not just the file handle — a truncated file can sometimes open at the header level but fail on data access), and deletes any file that fails. This had to be re-run multiple times across the project as fresh extractions kept re-introducing previously-cleaned corrupt files (because a fresh copy from the source archive is not the same file object as a previously-cleaned one).

### 6. Kaggle session disconnects repeatedly losing working-directory state
**Error/symptom**: `/kaggle/working` is ephemeral within an interactive session; multiple real disconnects (mid-training and between training resumes) wiped symlinks, extracted data, and cloned repos, even though committed Datasets and saved checkpoint files survived.
**Fix**: built a repeatable "recovery" pattern (rebuild symlinks, re-run corruption check, re-verify fastMRI clone/commit and h5py version) that had to be manually re-run after each disconnect before training could resume. A full resume-capable training loop (loading `model_state_dict` + `optimizer_state_dict` + last completed epoch from the most recent checkpoint) was essential given how often this occurred.

### 7. `VarNet.forward()` requires an explicit coil dimension, even for single-coil data
**Error/symptom**: `IndexError: too many indices for tensor of dimension 4` inside `VarNet.forward()`, because only a batch dimension was added to inference inputs (`unsqueeze(0)`), but the model internally expects a `[batch, coil, height, width, complex]` 5D layout even when there is only one (size-1) coil.
**Fix**: `masked_kspace.unsqueeze(0).unsqueeze(0)` and equivalent for the mask, adding both batch and coil dimensions before every inference call.

### 8. VarNet output resolution mismatch with target
**Error/symptom**: model output came out at full k-space resolution (e.g. 640x368) while the target ground-truth image was already center-cropped (e.g. 320x320) by the fastMRI data pipeline, causing a shape mismatch when computing loss.
**Fix**: applied `fastmri.data.transforms.center_crop()` to the model output using the `crop_size` field already provided in each dataset sample, matching the official training script's convention.

### 9. `batch_size > 1` crashes with a tensor-stacking error
**Error/symptom**: `RuntimeError: stack expects each tensor to be equal size, but got [640, 368, 2] at entry 0 and [640, 372, 2] at entry 1` — different knee scans in the fastMRI dataset have slightly different raw k-space matrix widths (368 vs 372 columns), which the default PyTorch batch-collate function cannot stack together.
**Fix**: confirmed and accepted `batch_size=1` as a structural requirement of this dataset (not a bug to work around), using gradient accumulation instead to simulate a larger effective batch.

### 10. Device-mismatch errors mixing CPU and GPU tensors
**Error/symptom**: `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same` when computing SSIM loss, because inference output had been moved back to CPU (`.cpu()` inside a shared inference helper) but the loss function and target tensors were on GPU.
**Fix**: explicitly moved every tensor to the same `device` immediately before any loss computation, rather than assuming a consistent device throughout a shared helper function.

### 11. VarNet's internal data-consistency step requires a boolean mask, not float
**Error/symptom**: `RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Float`, from `torch.where(mask, ...)` inside VarNet's cascade block, because a manually constructed mask (for the brain OOD data, which doesn't come from the standard `SliceDataset` pipeline) was left as float (0.0/1.0) instead of being cast to boolean. The standard knee pipeline handled this internally and never surfaced the issue.
**Fix**: explicit `.bool()` cast on the mask tensor inside the shared inference helper, applied universally (not just for the brain path), to prevent the same class of bug recurring anywhere else.

### 12. Naive multi-coil-to-single-coil combination destroyed the brain data (the most consequential bug in the project)
**Error/symptom**: brain OOD evaluation produced an implausibly low SSIM (0.32 for all trained models), and — critically — the **zero-filled baseline** (which involves no model at all) was equally bad (0.1543), which was the key diagnostic clue that the problem was in data preparation, not in the models. The initial coil-combination approach summed all 20 receiver coils' complex k-space directly (an unweighted sum), which is physically wrong: each coil has its own spatial phase offset, and summing out-of-phase signals causes destructive cancellation, badly corrupting the resulting synthetic single-coil image.
**Fix, part 1 (combination method)**: replaced the naive sum with an SVD-based coil compression: computed the singular value decomposition of the central (ACS) calibration region across coils, and combined all coils using the phase-aligned top singular vector as weights, rather than summing them blindly. This is a simplified, non-learned version of the standard adaptive coil-combination approach (compare to Küstner et al. 2024's more thorough sensitivity-weighted method — explicitly noted as a simplification in this project's writeup).
**Fix, part 2 (scale normalization, a separate bug uncovered by the same diagnostic)**: even after the SVD fix, the zero-filled baseline was still implausibly low (0.1332), traced to a scale mismatch — the manually-built brain pipeline never applied the intensity normalization that fastMRI's official `SliceDataset`/`VarNetDataTransform` pipeline applies automatically for knee data. Fixed by rescaling each combined k-space slice so its resulting zero-filled image's peak intensity matched the ground-truth image's own peak intensity, per slice, before mask application. After both fixes, the zero-filled brain baseline rose to a plausible 0.4153, and the subsequent GAP 1 model results (SSIM 0.66-0.68 on brain) were consistent with values reported in similar cross-domain MRI reconstruction literature.
**Lesson**: always sanity-check a *zero-filled, no-model* baseline whenever building a new, non-standard data pipeline — if the baseline itself is broken, no amount of model debugging will explain the numbers.

### 13. Spurious negative correlation in GAP 2, caused by an unnormalized uncertainty metric
**Error/symptom**: the first pass of the checkpoint-ensemble uncertainty experiment produced a strong, highly statistically significant **negative** correlation (Pearson r=-0.46) between uncertainty and true error — the opposite of the useful direction (higher uncertainty should predict higher error, not lower).
**Diagnosis**: the uncertainty scalar (`std_map.mean()`) was computed on raw pixel values, while the true-error metric (1-SSIM) is inherently scale-invariant (SSIM internally normalizes by a `data_range`/max-value parameter). Since MRI slice brightness varies substantially slice-to-slice, brighter slices produced larger raw-pixel standard deviations even when the checkpoints actually agreed well in relative terms, confounding the correlation.
**Fix**: normalized the uncertainty scalar by each slice's own `max_value` (the same normalization SSIM already applies internally), putting both quantities on the same scale-invariant footing. This flipped the correlation to the expected positive direction (r=0.43 in-domain) and is the result reported as final.

### 14. GAP 3 TTA loss appeared to be exactly zero every epoch
**Error/symptom**: `avg self-supervised loss: 0.000000` printed identically across all 5 training epochs, which looked like either total training failure or a hidden-lines-selection bug.
**Diagnosis**: this was a display/precision issue, not a real bug — k-space magnitude values are extremely small (~1e-6 to 1e-8), so the MSE loss on them lands around 1e-10 to 1e-12, which rounds to `0.000000` at 6 decimal places in a standard `%.6f` print format. A manual diagnostic print using scientific notation confirmed the loss was genuinely non-zero (~1e-12) and computing correctly.
**Fix**: changed the print format to scientific notation (`%.6e`), confirming training was proceeding.

### 15. GAP 3 TTA had no measurable effect on output despite "converging" loss
**Error/symptom**: even with training loss visibly decreasing over 5 epochs, held-out SSIM before and after LoRA adaptation was bit-for-bit identical (0.6779 vs 0.6779).
**Diagnosis**: the extremely small loss magnitude (~1e-10 to 1e-12, per the previous bug) meant Adam's gradient-update denominator (`sqrt(v) + epsilon`, with the default `epsilon=1e-8`) was dominated by the epsilon term rather than the actual gradient magnitude, silently shrinking every optimizer step to roughly 1/100th its intended size.
**Fix**: rescaled the loss (multiplying both prediction and target by a constant factor, e.g. 1e6) before computing MSE, bringing the loss magnitude well above Adam's epsilon floor without changing the underlying loss landscape's minimum. After this fix, LoRA weights produced a real, measurable effect on the model's output (which then revealed the genuine negative TTA result documented above).

### 16. SSDU held-out line selection could accidentally include the ACS (center) region
**Error/symptom**: `NaN` loss values immediately from the first TTA epoch, propagating to a `NaN` held-out SSIM.
**Diagnosis**: the initial self-supervised split randomly selected any acquired k-space line (including the central ACS/auto-calibration lines) to hide from the model as "held-out." VarNet's sensitivity-map estimator (`sens_net`) depends on the ACS region for normalization; removing it from the input caused an internal division instability that propagated NaNs through the whole forward pass.
**Fix**: excluded the ACS region explicitly from the pool of candidate lines eligible to be held out, so held-out lines are always drawn only from the peripheral (non-ACS) acquired lines.

---

## WHAT I LEARNED FROM THIS PROJECT

- How to reproduce a published deep learning paper end-to-end from a partially-maintained, archived open-source codebase, including pinning and working around a real, unfixable-by-the-textbook-method dependency bug (h5py/HDF5 on a newer Python version).
- How to build resilient, resumable training infrastructure across two different free/paid cloud compute platforms (Kaggle, Vast.ai), surviving repeated real session interruptions without losing training progress.
- How to diagnose a broken data pipeline using a zero-model baseline check as the first diagnostic step, rather than debugging the model itself — this single technique (checking whether a trivial baseline is also broken) was the fastest way to correctly localize two separate, serious bugs (coil-combination phase cancellation, and a missing normalization step) that would otherwise have looked like a genuine, alarming domain-shift finding.
- How normalization mismatches between a custom-built data pipeline and an official one can silently invalidate an entire experiment's statistics (the GAP 2 sign-flip bug) — and that when a result looks "too strong" or points the wrong direction, the first suspect should be a scale/normalization mismatch, not the underlying hypothesis.
- How Adam's epsilon term can silently neuter gradient updates when working with a loss computed on physically tiny quantities (raw MRI k-space magnitudes), and why numerically rescaling a loss (without changing its minimum) is a legitimate and necessary fix, not a hack.
- The practical difference between a genuine bug and a genuine (if disappointing) negative research result — and the discipline of testing a bug hypothesis first (varying learning rate, checking monotonicity) before accepting a "this technique doesn't work here" conclusion.
- How to scope a project deliberately: cutting T=2 from the cascade sweep, cutting a human-in-the-loop review queue with no real reviewer, and cutting a dynamic model-routing policy, all mid-project, because each risked either wasting remaining budget/time or making an unsubstantiated capability claim.

## SKILLS I CAN CLAIM AFTER THIS PROJECT

- I can reproduce a research paper's method from its official codebase, including resolving real environment/dependency incompatibilities that have no clean documented fix.
- I can build and debug a multi-stage MRI data pipeline (raw k-space, undersampling masks, coil combination, normalization) from first principles, not just call a pre-built loader.
- I can design and run a small, honestly-scoped research extension (not a novel architecture) that produces defensible, citable, statistically-tested results, including reporting negative results without disguising them.
- I can build resumable, checkpoint-based training infrastructure that survives real infrastructure failures across multiple cloud platforms.
- I can diagnose subtle numerical bugs (scale mismatches, optimizer-epsilon dominance, boolean-vs-float mask type errors) using targeted, hypothesis-driven print-based diagnostics rather than guesswork.

## WHAT TO WRITE ON CV (Bullet Points)

- Reproduced E2E-VarNet (Sriram et al., MICCAI 2020) for 4x-accelerated single-coil knee MRI reconstruction, achieving 0.7637 SSIM (+0.0184 over zero-filled baseline), and extended it with three original measurement studies: a cascade-depth-vs-cross-domain-robustness analysis, a zero-training-cost checkpoint-ensemble uncertainty estimator, and a self-supervised test-time adaptation experiment — all verified as open gaps in the current literature (through July 2026).
- Built and debugged a resumable multi-platform training pipeline (Kaggle + Vast.ai) surviving repeated real session failures, including a version-specific h5py memory leak with no available upstream fix, resolved via a custom code-level mitigation.
- Diagnosed and fixed three independent numerical bugs (coil-combination phase cancellation, a normalization-driven sign-flip in a statistical correlation result, and an optimizer-epsilon-driven silent training failure) using a zero-baseline-first debugging methodology, each of which would otherwise have produced a plausible-looking but incorrect research conclusion.

## WHAT TO WRITE IN PORTFOLIO / LINKEDIN POST

I reproduced E2E-VarNet, a published deep learning method for reconstructing MRI scans from partial data, and pushed past just reproducing it. I trained three model depths, tested them on a completely different body part than they were trained on, built a free uncertainty detector from checkpoints I already had, and tried test-time adaptation. Two of my three extension experiments gave results I didn't expect going in, and I'm reporting both honestly. The most useful debugging lesson: when a result looks alarmingly bad, check whether a trivial no-model baseline is equally bad before you start debugging the model. Twice, that single check saved me from chasing a fake "finding" that was actually a pipeline bug.

## INTERVIEW QUESTIONS — LIKELY TO BE ASKED

**Q: Why did you choose E2E-VarNet?**
A: It's a well-established, reproducible unrolled network with an official open-source implementation, feasible to train on a single consumer GPU, and it left genuine, verifiable gaps in the literature (joint depth-vs-robustness analysis, single-run checkpoint uncertainty) that I could fill without inventing a new architecture.

**Q: How does E2E-VarNet work?**
A: It's an unrolled iterative network — each "cascade" alternates between a data-consistency step (replacing the network's current k-space estimate with the actually-measured values at acquired locations) and a learned refinement step (a small U-Net that removes aliasing artifacts). Repeating this cascade T times, with the model trained end-to-end, produces the final reconstruction.

**Q: What would you do differently?**
A: Preserve intermediate checkpoints (not just best+final) across any compute-platform migration, so the checkpoint-ensemble experiment could have used the full intended K=4 spaced subset instead of K=2. I'd also budget compute for the full T=2-through-T=8 sweep rather than cutting T=2.

**Q: What was the hardest part?**
A: Diagnosing the coil-combination bug in the brain OOD data. The initial number (0.32 SSIM) looked like a dramatic, publishable domain-shift finding, and it would have been easy to just report it. Checking the zero-filled (no-model) baseline first — and finding it was equally broken — was what revealed it was a data-pipeline bug, not a real result.

**Q: How does it scale?**
A: Inference is fast per slice (single-coil, small model, sub-second on a consumer GPU). The main constraint is training compute for the cascade sweep — each additional cascade depth adds roughly proportional training time. Deployment-side, the multi-model registry design allows selecting cheaper/deeper models per request without retraining.

**Q: How did you test it?**
A: Every dataset extraction was followed by an explicit corruption check (open each file, touch the actual arrays, not just the file handle). Every custom-built data pipeline (the brain OOD path) was validated against a zero-filled, no-model baseline before trusting any model comparison built on it. Statistical results used Pearson/Spearman correlation with p-values and a Mann-Whitney U test for distribution comparison, not eyeballed numbers.

**Q: What are the limitations?**
A: Single-coil only (harder than, and not directly comparable to, the paper's own multi-coil numbers). Cascade sweep is missing T=2. GAP 2's checkpoint ensemble used K=2 instead of the intended K=4 due to a data-preservation gap across a compute migration. GAP 2's uncertainty signal is a moderate in-domain error predictor but an unreliable one under domain shift. GAP 3's TTA experiment produced a negative result in the specific setup tested (20 samples, SSDU-style proxy loss) and was not extended to try alternative proxy losses or larger sample counts, due to time and budget constraints.

## THINGS TO KEEP IN MIND AS SKILLS (Concepts I Should Be Able To Explain Confidently)

- **Unrolled/variational networks**: why alternating data-consistency and learned-refinement steps outperforms either a pure model or pure classical compressed-sensing approach.
- **k-space and undersampling**: why the ACS/center region carries most low-frequency contrast information and must always remain fully sampled; why peripheral lines carry fine detail.
- **Checkpoint ensembles vs snapshot ensembles**: checkpoint ensembles (Chen et al.) use ordinary sequential-training checkpoints; snapshot ensembles (Huang et al.) require a cyclic learning-rate schedule to reach genuinely distinct loss basins — these are not the same technique and should never be conflated.
- **Why single-run checkpoints have limited diversity**: they occupy a single loss basin (Fort et al. 2019), so their disagreement is a weaker uncertainty signal than an ensemble of independently-trained models.
- **SSDU-style self-supervised training/adaptation**: splitting *already-acquired* k-space into an input subset and a held-out loss subset, so a model can be trained or adapted without ever having access to true fully-sampled ground truth — directly relevant to real clinical settings where ground truth for a new scanner/site doesn't exist.
- **Why Adam's epsilon term matters**: at very small gradient/loss magnitudes, the epsilon added for numerical stability in the denominator of Adam's update rule can dominate and silently shrink effective learning rate — a subtlety that matters whenever working with physically small quantities like raw MRI k-space quantities.
- **Coil combination**: why summing multi-coil signals naively causes destructive phase interference, and why an SVD/calibration-region-based approach avoids it.

---
*Last updated: July 2026*
