# E2E-VarNet MRI Reconstruction

Reproduction of [E2E-VarNet (Sriram et al., MICCAI 2020)](https://arxiv.org/abs/2004.06688) for 4x-accelerated single-coil knee MRI reconstruction, extended with three honestly-measured experiments: a cascade-depth robustness study, a checkpoint-ensemble uncertainty signal, and a self-supervised test-time adaptation experiment. All results were independently stress-tested with volume-level clustering and bootstrap confidence intervals.

**This is a reproduction plus measured extensions, not a novel method.** The base architecture and training framework come from the official [facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI) codebase.

---

## Live Demo & Deployment

| What | Where |
|---|---|
| **Gradio demo + REST API** | [huggingface.co/spaces/alyrraza/e2e-varnet-api](https://huggingface.co/spaces/alyrraza/e2e-varnet-api) |
| **REST endpoint** | `POST https://alyrraza-e2e-varnet-api.hf.space/reconstruct` |
| **Health check** | `GET https://alyrraza-e2e-varnet-api.hf.space/health` |
| **React frontend** | Deployed on Vercel (Vite + Tailwind, calls REST endpoint) |
| **Model weights** | [huggingface.co/alyrraza/e2e-varnet-mri-reconstruction](https://huggingface.co/alyrraza/e2e-varnet-mri-reconstruction) *(private)* |
| **GitHub** | [github.com/alyrraza/e2e-varnet-mri-reconstruction](https://github.com/alyrraza/e2e-varnet-mri-reconstruction) |

### How inference works end-to-end

```
User uploads fastMRI .h5 file
        ↓
React frontend (Vercel)
        ↓  POST /reconstruct  (multipart/form-data)
HuggingFace Space  —  ZeroGPU, Gradio 5 + FastAPI
        ↓  hf_hub_download (authenticated via HF_TOKEN secret)
Model weights (private HF repo: best_model.pt + checkpoint_epoch_50.pt)
        ↓
VarNet T=4 (K=2 checkpoint ensemble)
  • 4× undersampled k-space → centre-slice reconstruction
  • Per-pixel uncertainty map (std across K=2 checkpoints)
  • Drift flag if uncertainty_scalar > 0.018
        ↓
JSON response: ssim, psnr, uncertainty_scalar, reconstruction_b64, uncertainty_map_b64
        ↓
React renders images + metrics
```

---

## Headline Result

> Checkpoint-ensemble uncertainty magnitude rises **1.54x** when the model sees out-of-distribution brain scans vs in-domain knee scans it was trained on.
> Bootstrap 95% CI: **[1.35x, 1.74x]** | Mann-Whitney p = 4.46e-6 | Works on **100%** of tested volumes.

This signal is free (uses checkpoints already saved during training), universally computable, and wired directly into the production drift monitor.

---

## Architecture

E2E-VarNet is an unrolled iterative network. Each cascade alternates two steps:

```
Input: undersampled k-space (25% of lines acquired, 4x acceleration)

For each cascade (T = 4, 6, or 8):
  ┌─────────────────────────────────────────┐
  │  Data-consistency step                  │
  │  Replace estimated k-space values at    │
  │  acquired locations with real measured  │
  │  values. Enforces physics constraint.   │
  └──────────────────┬──────────────────────┘
                     │
  ┌──────────────────▼──────────────────────┐
  │  Learned refinement step (U-Net)        │
  │  Remove remaining aliasing artifacts    │
  │  from the image domain.                 │
  └─────────────────────────────────────────┘

Output: reconstructed MRI image
```

Config used: `num_cascades=4/6/8`, `chans=18`, `sens_chans=8`, Adam lr=3e-4, loss=1-SSIM, batch=1 (structural requirement of fastMRI dataset), grad accumulation=8.

---

## Official Results

All SSIM values are **volume-level means** computed in a single session with an explicit mask seed. The paper's numbers (SSIM 0.930) are multi-coil and not directly comparable to these single-coil results.

### In-domain (knee) vs Out-of-domain (brain OOD)

| Model | Knee SSIM | Brain SSIM | OOD Drop |
|---|---|---|---|
| Zero-filled baseline | 0.7453 | 0.4153 | -- |
| T=4 (50 epochs) | 0.7594 | 0.6622 | -0.0972 |
| T=6 (25 epochs) | 0.7606 | 0.6705 | -0.0901 |
| T=8 (24 epochs) | **0.7607** | **0.6773** | -0.0834 |

Brain data: fastMRI multi-coil brain val, coil-combined via SVD to synthetic single-coil.

### Convergence

![Convergence T4 vs T6](figures/convergence_t4_t6_fair.png)

---

## GAP 1 — Cascade Depth vs Cross-Domain Robustness

**Finding: near-null.** A small, real trend exists in whole-image SSIM (deeper = slightly better OOD), confirmed by paired volume-level bootstrap. But PSNR, NMSE, and foreground-masked SSIM are flat. The SSIM trend is not corroborated by pixel-fidelity metrics.

| Comparison | Brain SSIM diff | 95% CI | Excludes zero? |
|---|---|---|---|
| T6 vs T4 | small positive | [0.0017, 0.0156] | Yes |
| T8 vs T6 | small positive | [0.0010, 0.0117] | Yes |
| T8 vs T4 | +0.0150 | [0.0076, 0.0232] | Yes |

**Honest conclusion:** statistically real in SSIM terms specifically. Not a clinically meaningful robustness improvement. Do not cite as a general depth-robustness result.

![GAP 1 and GAP 2 Summary](figures/gap1_gap2_summary.png)

---

## GAP 2 — Checkpoint-Ensemble Uncertainty (Headline Result)

A K=2 checkpoint ensemble (T=4 epoch-21 best + epoch-50 final) produces a per-pixel uncertainty map. The normalised scalar `mean(std_map) / max_value` is used as a drift signal.

**Why K=2 and not K=4:** the ideal spaced subset (epochs 21/30/40/50) was not available because intermediate checkpoints did not survive a Kaggle-to-Vast.ai compute migration. This is a documented limitation.

### Error prediction (in-domain, knee)

| Signal | Bootstrap r | 95% CI |
|---|---|---|
| zf_residual (zero-filled residual) | **0.715** | [0.527, 0.845] |
| ckpt_unc (checkpoint ensemble) | 0.432 | [0.359, 0.503] |
| periphery_ratio (k-space energy) | 0.158 | [0.034, 0.273] |

`zf_residual` outperforms checkpoint-ensemble for error prediction. Reported honestly.

### Distribution-shift detection (brain vs knee)

| Signal | Shift ratio | p-value | Coverage |
|---|---|---|---|
| eSNR proxy | 2.04x | 7.78e-7 | 12/15 brain volumes |
| **ckpt_unc** | **1.54x** | **4.46e-6** | **15/15 brain volumes** |
| zf_residual | 1.17x | 0.016 | 15/15 brain volumes |
| KER (k-space energy ratio) | 0.59x | 0.082 (n.s.) | 15/15 brain volumes |

eSNR is numerically stronger but only computable on 80% of volumes. Checkpoint-ensemble uncertainty is the recommended production drift signal: universally computable, strong, and directly wired into the Evidently monitor.

---

## GAP 3 — Self-Supervised LoRA Test-Time Adaptation

**Finding: controlled negative result.** SSDU-style LoRA adaptation (rank=4, 20 unlabeled brain slices, ACS-protected self-supervised split) degraded SSIM at both tested learning rates.

| LR | Before TTA | After TTA (5 epochs) |
|---|---|---|
| 1e-3 | 0.6779 | 0.6337 (monotonic decline) |
| 1e-4 | 0.6779 | 0.6756 (slow decline) |

A positive control (same procedure knee-to-knee) reproduced the same monotonic decline pattern, ruling out domain shift as the cause. This is a property of the SSDU proxy task on an already-converged model, not a domain-shift-specific failure.

**Scope caveat:** SSDU-style TTA is designed for severely mismatched baselines. This experiment used T=8 which already achieves 0.678 SSIM on brain before adaptation. The harder regime (genuinely unfamiliar scanner, poor initial performance) was not tested.

![GAP 3 TTA Trajectories](figures/gap3_summary.png)

---

## Qualitative Results

Before/After comparison (zero-filled input vs T=8 reconstruction vs ground truth):

![Qualitative comparison](figures/qualitative_comparison_honest.png)

*Worst, median, and best-case slices by SSIM. Bottom 10% of slices are edge-of-volume (slice_num 0-2) with no anatomical content -- expected fastMRI dataset structure, not reconstruction failures.*

![Bottom slices check](figures/bottom_slices_check.png)

---

## MLOps Stack

```
Training (Kaggle / Vast.ai GPU)
  └── fastMRI single-coil knee, 50 epochs, T=4/6/8 cascades
  └── Checkpoints → private HuggingFace model repo

Serving
  ├── HuggingFace Space (alyrraza/e2e-varnet-api)
  │     ├── Gradio 5 UI  at  /
  │     ├── POST /reconstruct  (REST, called by frontend)
  │     └── GET  /health
  ├── ZeroGPU (free-tier GPU acquired per request)
  └── K=2 checkpoint ensemble: best_model.pt (epoch-21) + checkpoint_epoch_50.pt

Frontend
  └── React + Vite + Tailwind  →  Vercel
        └── calls VITE_API_URL=/reconstruct

Monitoring / CI
  ├── Uncertainty scalar > 0.018  →  OOD drift flag in response JSON
  ├── Evidently AI  →  k-space feature drift monitor (offline)
  ├── Prometheus + Grafana  →  RED metrics + uncertainty gauge (local docker)
  └── GitHub Actions  →  ruff lint + CPU quality gate (SSIM threshold)
```

---

## Run Locally with Docker

```bash
git clone https://github.com/alyrraza/e2e-varnet-mri-reconstruction.git
cd e2e-varnet-mri-reconstruction

# Checkpoints must be downloaded from HuggingFace Hub first
python -c "
from huggingface_hub import hf_hub_download
import pathlib
ckpt_dir = pathlib.Path('checkpoints')
ckpt_dir.mkdir(exist_ok=True)
for f in ['best_model.pt', 'checkpoint_epoch_50.pt', 't6_best_model.pt', 't8_best_model.pt']:
    hf_hub_download(repo_id='alyrraza/e2e-varnet-mri-reconstruction', filename=f, local_dir='checkpoints')
"

docker compose up --build
# API live at http://localhost:8000/docs
# MLflow UI at http://localhost:5000
# Grafana at http://localhost:3000
```

---

## Run Quality Gate Tests (CPU, no GPU needed)

```bash
pip install -r mlops/requirements.txt
pytest mlops/tests/test_quality_gate.py -v
```

Threshold: `zero_filled_SSIM (0.7453) + 0.005 = 0.7503` (baseline-relative, not hardcoded).

---

## Project Structure

```
e2e-varnet-mri-reconstruction/          ← GitHub repo
├── frontend/                           ← React app (deployed on Vercel)
│   ├── src/
│   │   ├── api/client.ts               # calls HF Space /reconstruct
│   │   ├── components/                 # UploadZone, MetricsBar, ImagePanel
│   │   └── App.tsx
│   └── vite.config.ts
├── mlops/
│   ├── scripts/backfill_mlflow.py      # MLflow backfill
│   ├── src/
│   │   ├── uncertainty.py              # K=2 checkpoint ensemble
│   │   ├── api/main.py                 # local FastAPI (for docker)
│   │   └── monitoring/drift_check.py   # Evidently drift monitor
│   ├── tests/test_quality_gate.py      # CI quality gate
│   ├── prometheus.yml
│   └── grafana/
├── figures/                            # Result plots from research phase
├── results/                            # Locked JSON results (GAP 1/2/3)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/quality_gate.yml

HuggingFace Space (alyrraza/e2e-varnet-api)  ← separate HF repo
└── app.py                              # Gradio + FastAPI, ZeroGPU inference

HuggingFace Model (alyrraza/e2e-varnet-mri-reconstruction)  ← private
├── best_model.pt                       # T=4 epoch-21, best val SSIM (41 MB)
└── checkpoint_epoch_50.pt             # T=4 epoch-50, final (124 MB)
```

---

## Known Limitations

- Single-coil only (paper reports multi-coil; numbers are not comparable)
- T=2 cascade was scoped out; sweep covers T=4, T=6, T=8
- K=2 checkpoint ensemble instead of ideal K=4 (intermediate checkpoints lost in compute migration)
- Cross-domain comparison is confounded by two different coil-combination methods (fastMRI official for knee, SVD for brain)
- GAP 1 depth-robustness trend exists in SSIM only, not in PSNR/NMSE/FG-SSIM
- GAP 3 TTA tested only in already-decent-baseline regime

---

## Citation

```bibtex
@inproceedings{sriram2020endtoend,
  title={End-to-End Variational Networks for Accelerated MRI Reconstruction},
  author={Sriram, Anuroop and Zbontar, Jure and Murrell, Tullie and Defazio, Aaron
          and Zitnick, C. Lawrence and Yakubova, Nafissa and Knoll, Florian
          and Johnson, Patricia},
  booktitle={MICCAI},
  year={2020}
}
```

Dataset: [fastMRI (Knoll et al., 2020)](https://arxiv.org/abs/1811.08839)
Weights: [huggingface.co/alyrraza/e2e-varnet-mri-reconstruction](https://huggingface.co/alyrraza/e2e-varnet-mri-reconstruction)

---

## Related Work

- Darestani, Liu, Heckel — robustness of MRI reconstruction (ICML 2022, arXiv:2204.07204)
- D2SA — test-time adaptation for MRI (arXiv:2503.20815)
- Chen, Lundberg, Lee — checkpoint ensembles (arXiv:1710.03282)
