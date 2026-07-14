"""
CI quality gate test — Component 5.

Runs CPU-only inference on the committed validation subset (data subset/*.h5)
and checks that every model beats a baseline-relative SSIM threshold.

Why baseline-relative and not a hardcoded number:
  The best model achieves SSIM 0.7607. A hardcoded threshold of >= 0.76 leaves
  only 0.0007 margin and would fail on a legitimate regression in a different
  evaluation session due to mask non-determinism. Instead we require:
    model_ssim >= ZERO_FILLED_SSIM + REQUIRED_MARGIN
  where ZERO_FILLED_SSIM = 0.7453 and REQUIRED_MARGIN = 0.005 (0.5 points above
  the zero-filled baseline). This is a conservative but meaningful bar.

The mask seed is always set explicitly here (MASK_SEED = 42) to avoid the
run-to-run non-determinism documented in Bug 18 of PROJECT_NOTES_FINAL.md.

Run:
    pytest mlops/tests/test_quality_gate.py -v
"""

import gc
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

try:
    from fastmri.data import transforms as T
    from fastmri.data.subsample import EquispacedMaskFractionFunc
    from fastmri.models import VarNet
    import fastmri
    from fastmri.evaluate import ssim as compute_ssim
    from fastmri.data.transforms import center_crop
except ImportError:
    pytest.skip("fastMRI not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_SUBSET = ROOT / "data subset"

ZERO_FILLED_SSIM = 0.7453   # locked, from PROJECT_NOTES_FINAL.md
REQUIRED_MARGIN = 0.005     # model must beat zero-filled by at least this much
THRESHOLD = ZERO_FILLED_SSIM + REQUIRED_MARGIN

MASK_SEED = 42              # explicit seed -- never omit (Bug 18)
CENTER_FRACTIONS = [0.08]
ACCELERATIONS = [4]
CROP_SIZE = (320, 320)

CHECKPOINTS_TO_TEST = {
    "T4_best": "best_model.pt",
    "T6_best": "t6_best_model.pt",
    "T8_best": "t8_best_model.pt",
}


def load_model(checkpoint_name: str, device: torch.device) -> VarNet:
    """Load VarNet from a checkpoint file. Handles both raw and wrapped state dicts."""
    path = CHECKPOINT_DIR / checkpoint_name
    if not path.exists():
        pytest.skip(f"Checkpoint not found: {path}")

    # T4 has num_cascades=4; T6=6; T8=8 -- infer from filename
    if "t6" in checkpoint_name.lower():
        cascades = 6
    elif "t8" in checkpoint_name.lower():
        cascades = 8
    else:
        cascades = 4

    model = VarNet(num_cascades=cascades, chans=18, sens_chans=8)
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def eval_slice(model, h5_path: Path, slice_idx: int, device: torch.device) -> tuple[float, float]:
    """
    Evaluate one slice. Returns (model_ssim, zf_ssim) both vs ground truth.
    """
    mask_func = EquispacedMaskFractionFunc(
        center_fractions=CENTER_FRACTIONS,
        accelerations=ACCELERATIONS,
        seed=MASK_SEED,
    )
    transform = T.VarNetDataTransform(mask_func=mask_func)

    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][slice_idx]    # [H, W] single-coil
        max_value = float(hf.attrs.get("max", np.abs(kspace).max()))

    # VarNetDataTransform expects [coils, H, W, complex=2] float
    kspace_np = kspace.copy()
    if np.iscomplexobj(kspace_np):
        kspace_real = np.stack([kspace_np.real, kspace_np.imag], axis=-1)
    else:
        kspace_real = kspace_np
    kspace_real = kspace_real[np.newaxis]   # add coil dim

    sample = transform(kspace_real, None, None, None, max_value)
    mk = sample.masked_kspace.to(device)
    mask = sample.mask.bool().to(device)
    target = sample.target                  # [H, W] ground-truth image

    with torch.no_grad():
        output = model(
            mk.unsqueeze(0),        # add batch dim
            mask.unsqueeze(0),
            torch.tensor(max_value, device=device),
        )
    pred = output.squeeze(0)        # [H, W]
    pred_cropped = center_crop(pred, CROP_SIZE).cpu()
    target_cropped = center_crop(target, CROP_SIZE)

    # Zero-filled image
    zf = fastmri.ifft2c(mk)
    zf_img = fastmri.complex_abs(zf).squeeze(0)
    zf_cropped = center_crop(zf_img.cpu(), CROP_SIZE)

    def to_numpy(t):
        return t.unsqueeze(0).unsqueeze(0).numpy()

    model_ssim = float(compute_ssim(to_numpy(pred_cropped), to_numpy(target_cropped), maxval=max_value))
    zf_ssim = float(compute_ssim(to_numpy(zf_cropped), to_numpy(target_cropped), maxval=max_value))

    return model_ssim, zf_ssim


@pytest.fixture(scope="session")
def device():
    # CI runs on GitHub free runners (CPU only)
    return torch.device("cpu")


@pytest.fixture(scope="session")
def h5_files():
    files = sorted(DATA_SUBSET.glob("*.h5"))
    if not files:
        pytest.skip(f"No .h5 files found in {DATA_SUBSET}")
    return files


@pytest.mark.parametrize("model_key,ckpt_name", CHECKPOINTS_TO_TEST.items())
def test_model_beats_baseline(model_key, ckpt_name, device, h5_files):
    """
    Model SSIM on the local validation subset must be at least
    zero_filled_ssim + REQUIRED_MARGIN = {THRESHOLD:.4f}.
    """
    model = load_model(ckpt_name, device)

    ssim_scores = []
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as hf:
                n_slices = hf["kspace"].shape[0]
            # Test the middle slice (avoids edge slices with no anatomy)
            mid = n_slices // 2
            model_ssim, _ = eval_slice(model, h5_path, mid, device)
            ssim_scores.append(model_ssim)
        except Exception as exc:
            pytest.fail(f"Failed on {h5_path.name}: {exc}")
        finally:
            gc.collect()

    mean_ssim = float(np.mean(ssim_scores))
    print(f"\n{model_key}: mean SSIM on subset = {mean_ssim:.4f}, threshold = {THRESHOLD:.4f}")

    assert mean_ssim >= THRESHOLD, (
        f"{model_key} SSIM {mean_ssim:.4f} is below threshold {THRESHOLD:.4f} "
        f"(zero_filled {ZERO_FILLED_SSIM} + margin {REQUIRED_MARGIN})"
    )


def test_uncertainty_pipeline_loads(device):
    """Sanity check: both T4 checkpoints load without error."""
    from mlops.src.uncertainty import UncertaintyPipeline
    pipeline = UncertaintyPipeline(
        checkpoint_dir=CHECKPOINT_DIR,
        device=str(device),
    )
    assert pipeline.model_best is not None
    assert pipeline.model_final is not None
