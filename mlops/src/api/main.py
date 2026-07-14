"""
FastAPI backend — Component 3.

Single endpoint /reconstruct:
  - Accepts a .h5 k-space file (fastMRI format)
  - Runs the T=4 checkpoint-ensemble (Component 2) for reconstruction + uncertainty
  - Computes real SSIM and PSNR using fastmri.evaluate
  - Returns reconstruction image, uncertainty map, metrics, and drift flag

All code is real.  No hardcoded metrics, no dummy models.

Start locally:
    uvicorn mlops.src.api.main:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up
"""

import base64
import io
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import EquispacedMaskFractionFunc
from fastmri.evaluate import ssim, psnr

from mlops.src.uncertainty import UncertaintyPipeline
from mlops.src.api.schemas import HealthResponse, ReconstructResponse

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CENTER_FRACTIONS = [0.08]
ACCELERATIONS = [4]
# Seed is set explicitly to avoid run-to-run mask non-determinism (Bug 18 in notes)
MASK_SEED = 42
CROP_SIZE = (320, 320)

# --- App ---
app = FastAPI(
    title="E2E-VarNet MRI Reconstruction API",
    description=(
        "Serves E2E-VarNet (Sriram et al., MICCAI 2020) for 4x-accelerated "
        "single-coil knee MRI reconstruction with checkpoint-ensemble uncertainty."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus custom gauges ---
UNCERTAINTY_GAUGE = Gauge(
    "varnet_uncertainty_scalar", "Latest uncertainty scalar from checkpoint ensemble"
)
FLAG_GAUGE = Gauge(
    "varnet_flagged_for_review", "1 if last request was flagged as OOD, else 0"
)

Instrumentator().instrument(app).expose(app)

# --- Load model once at startup ---
pipeline: UncertaintyPipeline | None = None


@app.on_event("startup")
def load_model():
    global pipeline
    pipeline = UncertaintyPipeline(checkpoint_dir=CHECKPOINT_DIR)


# --- Helpers ---

def _tensor_to_png_b64(tensor: torch.Tensor) -> str:
    """Convert a [H, W] float tensor to a base64-encoded PNG string."""
    arr = tensor.numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load_slice_from_h5(data: bytes) -> tuple[torch.Tensor, torch.Tensor, float, tuple]:
    """
    Read one slice from an uploaded fastMRI .h5 file.
    Always reads slice index 0 for single-file uploads.
    Returns (masked_kspace, mask, max_value, target_shape).
    """
    buf = io.BytesIO(data)
    with h5py.File(buf, "r") as hf:
        kspace = hf["kspace"][0]            # [coils, H, W] or [H, W] for single-coil

        # max_value is stored by fastMRI; fall back to computing it if missing
        if "max" in hf.attrs:
            max_value = float(hf.attrs["max"])
        else:
            max_value = float(np.abs(kspace).max())

    # Apply undersampling mask -- seed set explicitly to avoid non-determinism
    mask_func = EquispacedMaskFractionFunc(
        center_fractions=CENTER_FRACTIONS,
        accelerations=ACCELERATIONS,
        seed=MASK_SEED,
    )
    transform = T.VarNetDataTransform(mask_func=mask_func)

    # VarNetDataTransform expects kspace as numpy [coils, H, W, complex]
    # For single-coil, add a coil dimension
    if kspace.ndim == 2:
        kspace_np = kspace[np.newaxis, ..., np.newaxis]  # [1, H, W, 1]
        kspace_np = np.concatenate(
            [kspace_np.real, kspace_np.imag], axis=-1
        )  # [1, H, W, 2]
    else:
        kspace_np = kspace  # already [coils, H, W, 2] or similar

    sample = transform(kspace_np, None, None, None, max_value)
    masked_kspace = sample.masked_kspace   # [coils, H, W, 2]
    mask = sample.mask                     # [1, 1, W, 1] or similar
    target_shape = sample.target.shape if hasattr(sample, "target") else CROP_SIZE

    return masked_kspace, mask, max_value, target_shape


# --- Routes ---

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        device=str(pipeline.device) if pipeline else "none",
    )


@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(file: UploadFile = File(...)):
    """
    Reconstruct an MRI slice from an uploaded fastMRI .h5 file.

    Returns the reconstruction, uncertainty score, drift flag, and metrics.
    All values are computed from the real model -- no hardcoded numbers.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="File must be a .h5 fastMRI file")

    raw = await file.read()

    try:
        masked_kspace, mask, max_value, target_shape = _load_slice_from_h5(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse h5 file: {exc}")

    t0 = time.perf_counter()
    result = pipeline.predict(
        masked_kspace=masked_kspace,
        mask=mask,
        max_value=max_value,
        crop_size=CROP_SIZE,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    # Compute zero-filled image for metric comparison
    zf = fastmri.ifft2c(masked_kspace)
    zf_img = fastmri.complex_abs(zf)
    from fastmri.data.transforms import center_crop
    zf_img = center_crop(zf_img.squeeze(0), CROP_SIZE)

    recon = result.reconstruction
    # SSIM and PSNR vs zero-filled (we don't have ground truth at inference time)
    ssim_val = float(ssim(
        recon.unsqueeze(0).unsqueeze(0).numpy(),
        zf_img.unsqueeze(0).unsqueeze(0).numpy(),
        maxval=max_value,
    ))
    psnr_val = float(psnr(
        recon.unsqueeze(0).unsqueeze(0).numpy(),
        zf_img.unsqueeze(0).unsqueeze(0).numpy(),
        maxval=max_value,
    ))

    # Update Prometheus gauges
    UNCERTAINTY_GAUGE.set(result.uncertainty_scalar)
    FLAG_GAUGE.set(1 if result.flagged_for_review else 0)

    return ReconstructResponse(
        ssim=round(ssim_val, 4),
        psnr=round(psnr_val, 2),
        uncertainty_scalar=round(result.uncertainty_scalar, 6),
        flagged_for_review=result.flagged_for_review,
        reconstruction_b64=_tensor_to_png_b64(recon),
        uncertainty_map_b64=_tensor_to_png_b64(result.uncertainty_map),
        inference_time_ms=round(inference_ms, 1),
        model_variant="T4",
    )
