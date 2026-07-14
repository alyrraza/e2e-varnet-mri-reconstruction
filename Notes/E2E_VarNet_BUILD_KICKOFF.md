# E2E-VarNet MRI Reconstruction — Build Kickoff Sheet
### Take this into the new build chat. Everything the new session needs is here.

---

## HOW TO OPEN THE NEW CHAT
Say exactly:
"New session — E2E-VarNet MRI Reconstruction. Read PORTFOLIO_PROJECTS_STRATEGY.md from project files."
Then attach this file and the two prior files (GOLD_PROMPT_v2_HonestBuild.md and VERIFICATION_AND_BUILD_NOTES.md) for reference.

---

## WHAT THIS PROJECT IS (one line)
Reproduce E2E-VarNet (Sriram et al., MICCAI 2020) for accelerated MRI reconstruction on Kaggle 2x T4, then add one honest measured extension. This is a reproduction plus a modest extension, NOT a novel method. Say that everywhere.

## THE HONEST FRAMING (never drift from this)
- Base: E2E-VarNet, arXiv:2004.06688, official code github.com/facebookresearch/fastMRI (MIT license).
- Extension: (1) cascade-count efficiency curve under a T4 budget, and (2) knee-to-brain cross-domain degradation, measured honestly, NOT fixed with a fancy method.
- Honesty sentence to keep in the report: "As of 2026, E2E-VarNet does not report a cascade-count efficiency curve on constrained GPUs or a cross-domain robustness evaluation as a contribution. Closest works: Darestani/Liu/Heckel (arXiv:2204.07204), D2SA (arXiv:2503.20815). My contribution is to reproduce and measure, not to claim a new method."

## GOAL WEIGHTING
PhD-email readiness AND MLOps-role readiness, equal. Domain: MRI reconstruction (k-space).

---

## HARD CONSTRAINTS (do not violate)
- Kaggle free tier ONLY: 2x T4, ~16GB per GPU, max 12h/session, ~30-42h/week, ~73GB disk.
- Dataset: fastMRI single-coil knee (SUBSET, keep working set under ~20GB). Cross-domain test: Calgary-Campinas (CC-359) brain, inference only.
- Deployment must be REAL: FastAPI + Docker + docker-compose + MLflow + Evidently + GitHub Actions + React/Vercel + HF Hub for weights.
- NO fake code ever: no DummyModel, no hardcoded metrics, no loss-against-random-noise. If showing a skeleton, label it skeleton.

---

## TWO LANDMINES — HANDLE ON DAY 1 (verified, will bite otherwise)
1. The official repo is ARCHIVED (18 Aug 2025). Install from source with a pinned commit, not a floating pip. Record the commit hash in PROJECT_NOTES.md.
   - `git clone https://github.com/facebookresearch/fastMRI.git`
   - `cd fastMRI && git log -1 --format="%H"`  (record this hash)
   - `pip install -e .`
2. h5py memory leak (repo Issue 215): pip-installed h5py leaks with HDF5 1.12.1+ when converting to torch.Tensor. On Kaggle you WILL hit this loading k-space in limited RAM.
   - Fix: use conda h5py 3.6.0 (HDF5 1.10.6) OR otherwise pin HDF5 < 1.12.1. Set this up before the first training run.

## NUMBER-PRECISION RULES (or a reviewer dings you)
- Paper Table 3 knee numbers are ROUNDED and MULTI-COIL: 4x SSIM 0.930 / PSNR 40; 8x SSIM 0.890 / PSNR 37.
- The crisp numbers (0.0053/39.37/0.9236) are from the PromptMR repo's overlapping-split re-eval, "for reference only." Do NOT present as the paper's own.
- You train SINGLE-COIL (harder than multi-coil). Never compare your single-coil SSIM to the paper's 0.930 multi-coil number as equivalent. State the setting with every number.
- Plausible single-coil knee 4x SSIM TARGET: ~0.72-0.78. It is a target, not a guarantee. Report the real gap.

---

## WEEK-BY-WEEK PLAN
- Weeks 1-2: fastMRI single-coil knee subset onto Kaggle (~150-250 train vols, ~40 val). Reproduce small VarNet (T=4) at 4x. GATE: beat zero-filled by several dB PSNR. If not, debug data/masks first.
- Weeks 3-6: cascade sweep T = 2,4,6,8 and the 8x setting. Log all to MLflow. Produce honest speed-vs-quality curve. If a 12h session cannot finish T=4 for ~30 epochs, drop to 256x256 crops / fewer volumes before cutting scope.
- Weeks 7-9: cross-domain test on Calgary-Campinas brain (INFERENCE ONLY). Report degradation honestly. Do not try to fix it.
- Weeks 10-12: build deployment stack (FastAPI + Docker + MLflow + Evidently + React), all real.
- Weeks 13-14: write up, push weights to HF Hub, deploy Space + Vercel frontend.
- Escalation: touch PromptMR-plus ONLY if early AND only to run released weights for inference comparison. Never retrain its 32-cascade model on a T4.

## STARTING HYPERPARAMETERS (official E2E-VarNet defaults)
Cascades T = 4 (sweep 2,4,6,8) | U-Net channels 18 (fallback 12) | SME U-Net channels 8 | Adam | lr 3e-4 or 1e-3, decay 0.1 at epoch 40 | loss 1 - SSIM | batch 1 | grad accumulation 8-16 | AMP mixed precision | 30-40 epochs | accel 4x (8% ACS) and 8x (4% ACS) | equispaced/random Cartesian mask.

## FALLBACK ORDER IF OOM
(1) fewer cascades (4->2), (2) smaller U-Net channels, (3) crop 256x256, (4) fewer volumes. Single-coil is already the memory-light path.

---

## DELIVERABLES CHECKLIST (from your strategy doc)
- [ ] GitHub repo with all 7 README elements (title, architecture diagram, tech badges, install, run-local, run-docker, results/live-link, paper link)
- [ ] Real deployed service (weights on HF Hub, working inference, real metrics)
- [ ] Cascade efficiency curve figure
- [ ] Cross-domain degradation table (knee vs brain)
- [ ] PROJECT_NOTES.md at end of build chat (all 13 sections)
- [ ] LinkedIn post ONLY after deploy + README done (honest framing, no em/en dash)

## REMEMBER
- No em dash, no en dash, anywhere.
- Every function explained in plain English + inline comments + why-this-over-alternatives.
- Two-tool workflow: use Claude Code (VS Code) for code generation/execution; use the build chat here for strategy, sequencing, Git commands, debugging logic.
