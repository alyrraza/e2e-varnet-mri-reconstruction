# GOLD RESEARCH PROMPT v2 — Honest Build-On-Published-Work Project (MRI Reconstruction)

## READ THIS FRAMING FIRST — IT OVERRIDES YOUR DEFAULT BEHAVIOR

I am NOT asking you to invent a never-before-seen method. I have learned the hard way that every clean single-technique combination in frontier MRI reconstruction (Mamba+diffusion, flow-matching for MRI, LoRA test-time adaptation, conformal uncertainty, knowledge distillation) has ALREADY been published in 2024-2025. If you claim any of these combos is "novel" or "first-ever," you are wrong, and you will damage my credibility with the professors I email.

My actual goal is a SOLID HONEST PROJECT:
- Take ONE published, verifiable, open-source-friendly MRI reconstruction paper as an explicit BASE.
- Reproduce a working version of it on Kaggle 2x T4.
- Add ONE small, honest, well-scoped extension the base paper did NOT do.
- Measure whether that extension actually helps. If it does not help, reporting that honestly is still a valid contribution.
- Cite the base paper as prior work I am building on. NEVER rename their method as my own.

This is enough for a PhD-application email (shows I can read papers, reproduce them, and think critically) AND for an MLOps engineering role (shows I can deploy a real model that actually works). I do not need "world-first." I need "real, working, honestly framed."

## HARD ANTI-PLAGIARISM RULES — NON-NEGOTIABLE

1. Before you propose ANY idea, you MUST search arXiv, Papers With Code, OpenReview, and Google Scholar to check whether that exact combination is already published. If it is, say so explicitly and either (a) pick it as an honest BASE to build on, or (b) move on.

2. You may NOT take a published paper's coined method name (e.g. PCFM, GSURE-based flow matching, FLAT, DA-CLIP, DACB) and present it as my invention. If a technique comes from a paper, name that paper.

3. Every "novelty" claim you make MUST be paired with: "As of [month year], I could not find a paper that does exactly X. The closest works are [paper A], [paper B], which do Y instead." If you cannot write that sentence honestly, do not make the claim.

4. NEVER write placeholder, fake, or mock code that pretends to work. Specifically FORBIDDEN:
   - No DummyModel / DummyVectorField classes that return random noise.
   - No hardcoded fake metrics (e.g. estimated_psnr=34.8 returned without real computation).
   - No loss functions computed against torch.randn (fake self-supervision).
   If you show deployment code, it must call the ACTUAL trained model and compute REAL outputs. If you are only sketching structure, label it clearly as PSEUDOCODE / SKELETON and do not dress it up as a working service.

5. All performance numbers you cite from other papers MUST be traceable to a specific paper and table. Do not write "~36.8 dB" with a tilde and no source. If you cannot verify a number, say "number not verified" instead of inventing one.

6. Do not cite future-dated or non-existent arXiv IDs as established fact. If a paper is very recent (within ~3 months) or you are unsure it exists, flag it as unverified.

## WHO IS ASKING

Final-year BSc Computer Science student, FAST-NUCES Faisalabad, graduating June 2026. AI/ML Engineer with freelance experience since 2023. GitHub: github.com/alyrraza. HuggingFace: alyrraza. Email: mirzaalirazafsd@gmail.com

Already deployed in production: RadGuard (multimodal VLM for chest X-ray error detection, AWS EC2), RetailSense AI (LightGBM forecasting, AWS EC2), Roman Urdu Sentiment (IndicBERT, FastAPI+Docker), YOLOv8 Asset System (Vast.ai + Vercel), B2B Voice Agent (LiveKit + n8n + MCP).

I have NO chest X-ray project need — I already have those. This must be MRI reconstruction.

## MY THEORETICAL FOUNDATION (so you calibrate the technical depth you can assume)

I can explain and implement all of the following from first principles:
- Probability: distributions, PDF/PMF, Bayes, MLE, negative log likelihood, real-vs-model distribution
- Generative models: autoregressive, VAE, GAN, EBM, flow, score-based, diffusion
- VAE: ELBO, reconstruction + KL terms, reparameterization trick (z = mu + sigma*epsilon), posterior collapse, beta-VAE
- Normalizing flows: change of variables, Jacobian determinant, coupling layers, NICE, RealNVP, MAF, IAF
- f-divergence, KL (forward/reverse), Jensen-Shannon, Fisher divergence, Wasserstein distance
- Score matching: score function grad log p(x), denoising score matching, Tweedie's formula, Langevin dynamics
- Diffusion: forward/reverse SDE, noise schedule, annealed Langevin, probability flow ODE, Euler-Maruyama, DDPM
- Flow matching: continuous velocity field, optimal transport paths, rectified flow (conceptually)
- Evaluation: FID, KID, Inception Score, PSNR, SSIM, LPIPS, MMD
- Mamba / state space models (conceptual understanding of linear-complexity sequence modeling)

So: you can propose technically deep work. My limitation is COMPUTE and TIME, not understanding.

## HARD CONSTRAINTS — NON-NEGOTIABLE

Compute:
- Kaggle free tier ONLY. 2x T4 GPUs, ~30GB total VRAM, ~16GB per GPU usable.
- Max 12 hours per session, ~30-42 hours per week.
- ~73GB disk on Kaggle.
- No paid cloud GPU budget. Occasional small AWS credits only.

Dataset:
- Must be publicly downloadable (Kaggle Datasets, official download, or wget). No hospital/IRB data.
- Must fit in Kaggle disk after subsetting. Prefer under 20GB working set.
- Acceptable: fastMRI (knee single-coil or a small multi-coil subset), IXI, Calgary-Campinas-359, BraTS subset. Whichever the BASE paper uses and is actually obtainable.

Deployment (all must be REAL, not mocked):
- Backend: FastAPI + Docker + docker-compose
- Model weights: Hugging Face Hub
- Experiment tracking: MLflow (log real params, real metrics, real artifacts)
- Drift monitoring: Evidently AI (real input distribution monitoring)
- CI/CD: GitHub Actions
- Frontend: React + TypeScript + Vite + Tailwind on Vercel
- Hosting: Hugging Face Spaces (CPU acceptable for demo) or AWS EC2
- Optional: Prometheus + Grafana

Domain: MRI reconstruction (undersampled k-space -> image). NOT CT, NOT CXR, NOT pathology.

Timeline: 8-12 weeks build + 2 weeks deployment. Done before Sept 2026.

## WHAT I WANT YOU TO DO

### PART 0: Base Paper Selection (DO THIS FIRST, SHOW YOUR WORK)

Research and recommend the SINGLE best published MRI reconstruction paper to use as my BASE, judged on:
- Is the method reproducible on Kaggle 2x T4 within 12h sessions? (this is the #1 filter)
- Is there official or reliable open-source code I can start from?
- Is the dataset it uses actually obtainable and small enough?
- Is it recent enough (2023-2025 preferred) to be relevant, but stable enough to be reproducible?
- Does it leave an obvious, small, honest extension gap in its limitations/future-work?

Present your TOP recommendation plus 2 alternatives in a table. For each: paper name, authors, venue/year, arXiv ID, code availability, dataset, why-feasible-on-Kaggle, and the honest extension gap it leaves. Then clearly recommend ONE and justify.

Do NOT pick a paper whose core is itself an unreproducible-on-Kaggle giant (no 8x A100 training runs). Favor unrolled/variational or lightweight score/diffusion methods that fit small GPUs.

### PART 1: The Honest Extension

For the chosen base paper, propose ONE small honest extension. It must be:
- Something the base paper explicitly did NOT do (quote their limitation/future-work if possible).
- Achievable on Kaggle 2x T4 in the remaining time budget.
- Measurable — I can show a number that says whether it helped.

Good honest extensions (examples, pick or improve on these):
- Cross-domain robustness evaluation the base paper skipped (train on knee, test on brain/other scanner, report the degradation honestly).
- A reproducibility study: how does the base method actually perform under 2x T4 constraints vs its paper claims.
- An uncertainty/error correlation study using a SIMPLE method (not a grand conformal framework), honestly reported.
- A lightweight efficiency modification (e.g. fewer cascades / fewer sampling steps) with an honest quality-vs-speed tradeoff curve.

For the extension, write the honesty sentence: "As of [month year], the base paper [X] does not do [extension]. The closest related work is [A], [B]. My contribution is to [do extension] and measure [metric]."

### PART 2: Current State of the Art (Verified)

Brief, VERIFIED landscape of MRI reconstruction 2023-2025:
- Top methods (unrolled: E2E-VarNet, PromptMR family; score/diffusion; Mamba-based; flow-matching). Name authors, venue, arXiv ID.
- Standard datasets and metrics (PSNR, SSIM, NMSE, LPIPS).
- For each number you cite, give the source. Mark anything you cannot verify as unverified.

### PART 3: Documented Gaps (Verified, With Sources)

List 4-5 REAL gaps that actually appear in papers' limitations/future-work sections, each with the paper name and what they actually said. Do not invent gaps. These are context, not all of them are my project.

### PART 4: The Project Design (Base + Extension)

- Tentative honest title (frames it as "building on [base]" or "an evaluation/extension of [base]", NOT a fake new method name).
- 150-word abstract that HONESTLY describes reproducing the base and adding the extension.
- Architecture: describe the base architecture (crediting the paper) and exactly what I add/change. A clear diagram in text is fine.
- Loss function: state the base paper's loss, and any change I make, with the math.
- Dataset setup: exact dataset, exact subset, how to get it on Kaggle, preprocessing steps, undersampling mask setup.
- Baseline to compare against: the base paper's own reported numbers (with source) and/or a simpler baseline (zero-filled, classical CS).
- Metrics: PSNR, SSIM, and one more (LPIPS or NMSE). Realistic target ranges, clearly labeled as targets not guarantees.

### PART 5: Kaggle Training Plan (Realistic)

- Model size estimate (params), image size, batch size, mixed precision, gradient accumulation.
- Epochs and realistic wall-clock estimate per epoch and total, fitting 12h sessions.
- Checkpointing to /kaggle/working/ and resume-on-restart strategy.
- Honest note on what might NOT fit and the fallback (smaller image, single-coil, fewer cascades).
- Full hyperparameter table.

### PART 6: MLOps Deployment (REAL CODE ONLY)

For the actual trained model, provide production deployment. ALL CODE MUST BE REAL:
- FastAPI endpoint that loads the REAL trained model from Hugging Face Hub and runs REAL inference on an uploaded k-space file, returning the REAL reconstruction + REAL computed metrics. If any part is illustrative, label it PSEUDOCODE explicitly.
- Dockerfile and docker-compose.yml (real, buildable).
- MLflow logging code that logs REAL training metrics.
- Evidently drift-monitoring code that monitors REAL input statistics (e.g. k-space energy distribution, SNR) vs the training distribution.
- GitHub Actions CI/CD workflow (lint, test, build, deploy).
- What the React frontend shows: upload undersampled k-space slice -> display zero-filled input, reconstructed output, error/uncertainty map, PSNR/SSIM, inference time.
- One paragraph: why this MLOps setup demonstrates real production maturity to an MLOps hiring manager.

Reminder: NO DummyModel. NO hardcoded metrics. NO loss-against-random-noise. If you catch yourself writing fake code, stop and label it as skeleton/pseudocode instead.

### PART 7: Positioning (Honest)

- LinkedIn hook line (one sentence, one number, no jargon, no em dash, no en dash).
- 3-sentence LinkedIn body (conversational, honest — "I reproduced [base] and extended it to [X]", not "I invented").
- 2-sentence PhD-email hook that is HONEST about building on the base paper. A professor should read it and think "this student can reproduce and extend real work," not "this student is overclaiming."
- 2 CV bullet points (action verb + method + metric + honest framing).
- Which type of professor / lab keywords to target for this specific work.

## WRITING RULES

- No em dash anywhere. No en dash anywhere. Rewrite sentences instead.
- Write like a confident human engineer, not like marketing copy.
- Prefer honest hedging ("this should", "in the paper's setup") over confident overclaiming.

## FINAL INSTRUCTION

Your job is NOT to make me sound impressive with a fake breakthrough. Your job is to find the most reproducible published MRI reconstruction paper for Kaggle 2x T4, design an honest small extension, and give me real deployable code. If at any point the honest answer is "this is a solid reproduction plus a modest extension, not a novel method," say exactly that. That is the outcome I want.

Before you finalize, re-read the ANTI-PLAGIARISM RULES and confirm your proposal does not rename any published method as mine and contains no fake code.

---
*Authored for: Ali Raza | github.com/alyrraza | Honest MRI Reconstruction Project (Base + Extension)*
*Goal weight: PhD-email AND MLOps-role, equal. Domain: MRI reconstruction. Compute: Kaggle 2x T4. Novelty target: solid honest contribution, not world-first.*
