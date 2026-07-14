# NEW CHAT KICKOFF — E2E-VarNet MLOps Phase
### Paste this entire document as the first message in a new chat session.

---

## HOW TO USE THIS DOCUMENT (for whoever/whatever reads this first)

This is a continuation of an existing, fully-completed research project. The research phase is DONE and LOCKED — do not re-derive, re-question, or re-open any of the numbers or findings below. This document exists so the MLOps/deployment phase can start immediately with full context, without re-explaining the whole project. Everything in "PART 1" is historical fact, not up for debate. "PART 2" is the actual task for this new chat.

---

# PART 1 — EVERYTHING ALREADY DONE (research phase, complete)

## Project identity
- Reproduction of E2E-VarNet (Sriram et al., MICCAI 2020, arXiv:2004.06688) for 4x-accelerated single-coil knee MRI reconstruction.
- Official fastMRI codebase, pinned commit `91f2df4711adbb6d643df1810f234e4abcf5881b` (repo is archived as of Aug 2025, installed from source).
- Framing, permanent: reproduction + honestly measured extensions, not a novel method. Never compare to the paper's own multi-coil SSIM (0.930).
- Compute used: Kaggle free tier (2x T4) + rented Vast.ai GPUs (RTX 3090/4090), real dollar budget constraints throughout.

## Official, locked results (do not use any other numbers)

| Model | Knee SSIM (in-domain) | Brain SSIM (OOD) |
|---|---|---|
| Zero-filled baseline | 0.7453 | 0.4153 |
| T=4 | 0.7594 | 0.6622 |
| T=6 | 0.7606 | 0.6705 |
| T=8 | 0.7607 | 0.6773 |

(T=2 was deliberately scoped out and does not exist.)

## Three research extensions, all complete, all statistically stress-tested (volume-clustered bootstrap, positive controls, trivial-baseline controls)

**GAP 1 (cascade depth vs cross-domain robustness): NEAR-NULL.** A small, real, paired-bootstrap-confirmed trend exists in whole-image SSIM only; not corroborated by PSNR, NMSE, or foreground-masked SSIM. Not the headline result. Do not oversell this in any MLOps context either — it's a minor, honestly-reported research finding, not a production capability.

**GAP 2 (checkpoint-ensemble uncertainty): THE HEADLINE RESULT, AND THE ONE THIS MLOPS PHASE IS BUILT AROUND.**
- Uses a K=2 checkpoint ensemble (T=4's epoch-21 and epoch-50 checkpoints; the ideal K=4 was not available due to a compute-migration data loss — documented limitation).
- **Drift-detection (the production-relevant use case): uncertainty magnitude rises 1.54x on brain (OOD) vs knee (in-domain), bootstrap 95% CI [1.35x, 1.74x], survives volume-level clustering correction, computable on 100% of tested volumes.** This is the signal to wire into the drift monitor.
- Compared against 3 other free signals: eSNR is numerically stronger for shift-detection (2.04x) but only computable on 12/15 brain volumes (partial coverage); checkpoint-ensemble uncertainty is the most robust/universally-computable of the four.
- Error-prediction (separate use case, less relevant to MLOps): checkpoint-ensemble r=0.43 in-domain, but a simpler zero-filled-residual proxy (`zf_residual`) is actually better (r=0.72) for this specific use case. Reported honestly, not hidden.
- OOD error-prediction: unreliable, do not use for that purpose.

**GAP 3 (self-supervised LoRA test-time adaptation): controlled negative result.** SSDU-style TTA degrades SSIM (doesn't help), confirmed via positive control (same degradation happens in-domain too, so it's not a domain-shift-specific problem, it's a proxy-task/method issue). Scope caveat: only tested on an already-decent baseline; harder mismatched-baseline scenarios untested. Not directly relevant to the MLOps build, but part of the complete research record.

## Real engineering/debugging record (19 numbered bugs total, full list in PROJECT_NOTES_FINAL.md) — highlights relevant context for MLOps work
- h5py memory leak on Python 3.12 with no clean upstream fix, worked around with code-level mitigation (relevant if the MLOps environment also uses h5py for any inference-time data loading).
- Mask-generation function (`EquispacedMaskFractionFunc`) has an optional `seed` parameter that was never set, causing run-to-run non-determinism in evaluation metrics. **For any MLOps evaluation/CI code, always pass an explicit seed.**
- Coil-combination and normalization bugs were both caught by checking a zero-model baseline first — this diagnostic habit should carry into MLOps testing too (e.g. sanity-check any new pipeline component against a trivial/no-op baseline before trusting its output).
- Pseudoreplication was caught and fixed in all statistics (volume-level clustering, block bootstrap) — if the MLOps phase does any further statistical monitoring/alerting design (e.g. setting drift-alert thresholds), this same volume/patient-level-independence principle applies and should not be re-broken.

## Assets available locally (already downloaded from Kaggle, ready to use in this new chat/session)
- Model checkpoints: `best_model.pt` (T4, epoch 21), `checkpoint_epoch_50.pt` (T4, epoch 50), `t6_best_model.pt`, `t8_best_model.pt`.
- Result JSONs: GAP 1, GAP 2, GAP 3 results, edge-slice finding.
- Figures: GAP1/GAP2 summary plot, GAP3 trajectory plot, T4-vs-T6 convergence plot, qualitative comparison figure, bottom-slices check figure.
- Training logs (text files) for T4 (two files, epochs 1-10 and 11-50) and T6 and T8.
- Full Kaggle notebook exported as .ipynb.
- A small (5-10 volume) subset of knee validation `.h5` files, intended for the CI/CD quality-gate (see Part 2).
- `PROJECT_NOTES_FINAL.md` — the complete, authoritative research write-up (attach this file to this new chat alongside this kickoff prompt).

---

# PART 2 — THE ACTUAL TASK: MLOPS PHASE (start here)

## Locked design principle: T=4-FIRST, GRACEFULLY DEGRADING

Every component below must work with ONLY the T=4 model and its checkpoints. T=6/T=8 are additive bonuses (multi-model registry entries), never blockers. If anything is incomplete by the end, the system should still be a complete, working, honestly-documented T=4-only pipeline.

## Component 1 — MLflow

- Backfill: write a script that loops over the four saved checkpoints (T4 epoch 21, T4 epoch 50, T6 best, T8 best) and logs their available metrics (train_loss, val_ssim, num_cascades where present) into MLflow, so the historical training record is captured even though it wasn't tracked live originally.
- Model Registry: register models using **aliases** (`@champion`, `@challenger`), NOT the deprecated MLflow "stages" (Staging/Production) — confirmed deprecated as of MLflow 2.9.0.
- Register T=4 (epoch 21 checkpoint, i.e. `best_model.pt`) as the `champion`.
- Register the two checkpoint-ensemble members (T4 epoch 21, T4 epoch 50) as a second registered model, e.g. `varnet-t4-ensemble-members`.
- If T=6/T=8 are available (they are, in this case), register them as additional versions too.

## Component 2 — Checkpoint-ensemble uncertainty pipeline (production version of the GAP 2 research code)

- Port the existing checkpoint-ensemble uncertainty computation (already built and validated in the research phase) into a clean, reusable inference function: given a k-space slice, run both T4 checkpoints, compute per-pixel std, return a `mean_uncertainty` scalar (normalized by max_value, exactly as in the research code) and a `flagged_for_review` boolean based on a calibrated threshold.
- This is simultaneously the production drift signal AND already-validated research artifact — one piece of code, two purposes.

## Component 3 — FastAPI + Docker

- `/reconstruct` endpoint: accepts a k-space slice, runs T=4 (hardcoded initially, per the T=4-first principle), returns the reconstruction plus the uncertainty scalar and flag boolean from Component 2.
- No human-in-the-loop queue, no reviewer UI, no database — this was explicitly cut from scope in an earlier review (a "clinical review queue" with no real clinician was judged to be a red flag, not a feature). The flag is just a JSON field in the API response.
- Dockerize.

## Component 4 — Evidently AI drift monitoring

- Evidently has no native image support — reduce k-space to scalar features first: k-space energy ratio (KER, already computed in research — was NOT a significant shift signal, 0.59x, p=0.082, so don't rely on it alone), estimated SNR (eSNR, computed in research — strongest shift signal at 2.04x but only reliable on ~80% of volumes, note this coverage caveat in the monitor design), matrix width (a real, cheap metadata feature — 368 vs 372 was actually encountered during training).
- Validate the drift monitor using the already-completed knee-vs-brain evaluation as a labeled positive-drift test case (brain = known OOD, should fire; knee = known in-domain, should not) — this reuses existing data/results, no new experiment needed.
- The checkpoint-ensemble uncertainty scalar (Component 2) is a complementary drift signal (1.54x, universally computable) and should be surfaced alongside the Evidently k-space-feature-based monitor, not replacing it.

## Component 5 — CI/CD quality gate (GitHub Actions)

- GitHub's free-tier runners have no GPU — the gate must run CPU-only inference on the small committed validation subset (5-10 `.h5` files, already prepared and available).
- Quality gate threshold: **baseline-relative, not an absolute hardcoded number** (an earlier draft used a hardcoded `>=0.76` which was flagged as too thin a margin against the actual best result of 0.7607 — use something like `zero_filled_baseline (0.7453) + a defined margin` instead).
- On pass: register the candidate as `challenger` in MLflow. Promotion to `champion` requires a human approval step, never automatic.
- Also run standard lint + pytest on every push.

## Component 6 — Prometheus + Grafana

- `prometheus-fastapi-instrumentator` for RED metrics (Rate, Errors, Duration).
- Custom gauges: uncertainty-score distribution, flag rate over time, drift pass/fail state.
- If T=6/T=8 are wired into the registry, add per-cascade-depth labels so one panel compares all variants; with T=4-only, the same panels work with a single series.

## Explicitly OUT OF SCOPE (do not build, do not suggest re-adding)
- Human-in-the-loop review queue with a reviewer UI and feedback database.
- A policy router that dynamically picks cascade depth per input.
- Auto-retraining from any feedback signal, ever.
- Simulated PACS/DICOM clinical integration — this is a portfolio project, not a clinical deployment; never imply otherwise.

## Build order (suggested, T=4-first, everything after step 2 works with T=4 alone)
1. MLflow backfill + register T=4 as `champion`.
2. Checkpoint-ensemble uncertainty pipeline (Component 2) — do this early since it's simultaneously a research artifact and the production signal.
3. FastAPI + Docker (Component 3).
4. Evidently drift check + knee/brain validation (Component 4).
5. GitHub Actions CI/CD quality gate (Component 5).
6. Prometheus + Grafana (Component 6).
7. Only if time permits: register T=6/T=8, add per-variant dashboard labels.

## Standing rules (carry over from the research phase, unchanged)
- No em dash, no en dash, anywhere.
- No placeholder/fake/mocked code, ever. Anything not fully built and working must be labeled PSEUDOCODE explicitly.
- Every number must be traceable to either this document or an actually-completed run; never invent one.
- Explain every function in plain English, add inline comments, note edge cases (per the original project's code-explanation rule).
- Produce/update a PROJECT_NOTES-style record for the MLOps phase at the end, following the same honest, bug-inclusive, no-oversell style as the research phase's PROJECT_NOTES_FINAL.md.

---

*Attach PROJECT_NOTES_FINAL.md alongside this document when starting the new chat. Everything needed to begin Component 1 (MLflow backfill) is already available in the local assets listed in Part 1.*
