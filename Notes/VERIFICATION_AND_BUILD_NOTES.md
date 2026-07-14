# E2E-VarNet Project — Verification Report and Build-Start Notes
### For Ali Raza | Verified before build begins

---

## PART 1: WHAT I VERIFIED (and it checked out)

The third tool's output is the first one that is honest and usable. I verified the load-bearing claims myself.

### Base paper and code — CONFIRMED REAL
- Paper: End-to-End Variational Networks for Accelerated MRI Reconstruction, Sriram, Zbontar et al., MICCAI 2020, arXiv:2004.06688. Real.
- Official code: github.com/facebookresearch/fastMRI. The VarNet model lives in fastmri/models, and a full training example is in fastmri_examples/varnet/. Real and present.
- License: MIT. You can legally use, modify, and deploy it. Just credit the paper.
- Built-in and ready to use: PyTorch data loaders (SliceDataset), subsampling mask functions, and evaluation metrics (PSNR, SSIM, NMSE) are all in the fastmri package. You do not reimplement these.

### The honest-extension gap — CONFIRMED REAL
- Knee-to-brain degradation is a documented, citeable finding: Darestani, Liu, Heckel, ICML 2022, arXiv:2204.07204 (a knee-trained model does not reconstruct brains well).
- D2SA (Zhang et al., CVPR 2025, arXiv:2503.20815) builds a test-time adaptation framework for exactly this shift, which proves the shift is a real, active research problem, not something you invented.
- So your extension (measure the degradation honestly, do NOT claim to fix it with a fancy method) sits in real, defensible territory.

### The "everything clean is already published" warning — CONFIRMED, still true
Across three tool outputs I checked, every clean single-technique combination was already published in 2024-2025:
- PCFM + GSURE unsupervised flow matching for parallel MRI = UPMRI (arXiv:2512.17493), Luo, Li, Qin, Imperial College.
- Flow-aligned unrolled training = FLAT (arXiv:2512.03020).
- Dose-anatomy CLIP + Mamba diffusion for CT = FoundDiff (arXiv:2508.17299), with public code.
- Conformal uncertainty for MRI = CUTE-MRI (arXiv:2508.14952) and others.
This is why the honest reproduction-plus-extension framing is the correct one. Do not let any tool talk you into a "world-first" claim. It would be false and a professor would catch it.

---

## PART 2: TWO THINGS THE OUTPUT MISSED (these will save you real time)

### Miss 1 — The official repo is ARCHIVED (as of 18 Aug 2025)
- Facebook archived github.com/facebookresearch/fastMRI. It is read-only now. The code works but is no longer maintained.
- What this means for you: this is actually GOOD for reproducibility (the code is frozen and will not shift under you), but you must install a pinned version from source, not a floating `pip install fastmri`, so a future dependency bump does not break your build.
- Action: clone the repo, install from source with `pip install -e .`, and record the exact commit hash in your PROJECT_NOTES.md and requirements.

### Miss 2 — Known h5py memory leak that WILL hit you on Kaggle
- The repo itself documents this (Issue 215): `h5py` installed via `pip` leaks memory when converting to a torch.Tensor with HDF5 1.12.1 or later.
- You are loading HDF5 k-space files in limited RAM on Kaggle. You will hit this.
- Fix: use the conda build of `h5py` 3.6.0 (which uses HDF5 1.10.6), or otherwise pin to an HDF5 version before 1.12.1.
- Action: pin h5py correctly at the start. Do not discover this after a session crashes at 3 hours in.

---

## PART 3: NUMBER-PRECISION RULES (do not get this wrong in your report)

The output was right to flag this. Be precise or a reviewer will ding you.
- The original E2E-VarNet paper (Table 3) reports only ROUNDED knee numbers: 4x SSIM 0.930 / PSNR 40; 8x SSIM 0.890 / PSNR 37. These are MULTI-COIL.
- The crisp numbers sometimes quoted (4x 0.0053 / 39.37 / 0.9236) come from the PromptMR repo re-evaluating on a different self-split test set that OVERLAPPED training data, by the PromptMR authors' own admission. They are "for reference only." Do NOT present them as the paper's own numbers.
- You will train SINGLE-COIL (memory reasons). Single-coil is HARDER than multi-coil. Do NOT compare your single-coil SSIM to the paper's 0.930 multi-coil number as if they are equivalent. State the setting every time you state a number.
- Target ranges the output gave (single-coil knee 4x SSIM roughly 0.72-0.78) are plausible TARGETS, not guarantees. Report your real gap honestly.

---

## PART 4: RECENT arXiv IDs TO TREAT AS UNVERIFIED
Several 2026-dated arXiv IDs appeared across the outputs (e.g. 2601.13236, 2604.11762, 2604.22557, 2512.20330). They are very recent. Their specific numbers should be re-checked before you cite them in a formal writeup. Do not build any core claim on a number you have not personally verified from the source.

---

## PART 5: THE 3 THINGS THAT MUST BE TRUE IN YOUR DEPLOYMENT (learned from the bad outputs)
The earlier tool outputs contained fake code. Yours must not. Non-negotiable:
1. NO DummyModel / DummyVectorField. The FastAPI service must load your REAL trained checkpoint from Hugging Face Hub and run REAL inference.
2. NO hardcoded metrics (no `estimated_psnr = 34.8`). Compute PSNR/SSIM/NMSE with the real fastmri.evaluate functions on the real reconstruction.
3. NO loss against torch.randn. If you show a code skeleton, LABEL it as skeleton, do not dress it up as a working service.
If a professor or an MLOps interviewer clones your repo and runs it, it must actually reconstruct an image. That is the whole point.

---

## PART 6: WEEK-BY-WEEK (from the output, lightly corrected)
- Weeks 1-2: Get fastMRI single-coil knee SUBSET onto Kaggle (about 150-250 train volumes, ~40 val, keep working set under ~20GB). Reproduce a small VarNet (T=4) at 4x. GATE: you must beat zero-filled by several dB PSNR. If not, debug data/masks BEFORE anything else.
- Weeks 3-6: Cascade sweep (T = 2,4,6,8) and the 8x setting. Log everything to MLflow. Produce the honest speed-vs-quality curve. If a 12h session cannot finish T=4 for ~30 epochs, drop to 256x256 crops and fewer volumes before cutting scientific scope.
- Weeks 7-9: Cross-domain test on Calgary-Campinas brain (INFERENCE ONLY). Report the degradation honestly. Do NOT try to fix it with a fancy method. Measuring it honestly IS the contribution.
- Weeks 10-12: Build the deployment stack (FastAPI + Docker + MLflow + Evidently + React), all real.
- Weeks 13-14: Write up, push weights to HF Hub, deploy the Space and the Vercel frontend.
- Escalation: only touch PromptMR-plus if you finish early AND only to run its RELEASED weights for an inference comparison. Do NOT retrain its 32-cascade model on a T4. It needs A100s.

---

## PART 7: HONEST BOTTOM LINE
This is a solid reproduction plus a modest, measured extension. It is NOT a novel method, and you should say exactly that everywhere. That honesty is a feature, not a weakness:
- For a PhD email: it shows you can read a paper, reproduce it under real constraints, and think critically about robustness. That is literally what a first-year PhD student does. It reads as "RA-ready," not "overclaiming undergrad."
- For an MLOps role: a real deployed medical imaging model with tracking, drift monitoring, and CI/CD is exactly the production-maturity signal hiring managers want. The model being a faithful reproduction rather than a novel invention does not matter to them at all.

Use the third tool's output as your build plan. Add the two missed items (archived repo pinning, h5py leak fix) before you start. Verify every number against its real source before it goes in your report.
