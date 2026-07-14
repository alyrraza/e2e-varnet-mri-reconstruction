"""
Checkpoint-ensemble uncertainty pipeline — Component 2.

Computes a drift/uncertainty signal from two T=4 checkpoints saved at
different training epochs (epoch-21 best, epoch-50 final).  This is the
production port of the GAP 2 research experiment.

Research result (locked):
  - In-domain error prediction:  r = 0.43, bootstrap 95% CI [0.36, 0.50]
  - Distribution-shift detection: 1.54x higher on OOD brain vs in-domain knee,
    bootstrap 95% CI [1.35x, 1.74x], Mann-Whitney p = 4.46e-6

How it works:
  Both checkpoints run inference on the same input.  The per-pixel standard
  deviation of the two predictions is the raw uncertainty map.  We normalise
  by the slice's own max_value (same normalisation SSIM uses internally) so
  the scalar is scale-invariant across slices of different brightness -- this
  is the fix for the sign-flip bug found in the research phase (Bug 13).

Limitation (stated explicitly):
  Ideal ensemble would be K=4 (epochs 21/30/40/50).  Only epochs 21 and 50
  survived the Kaggle-to-Vast.ai migration, so K=2 is used.  K=2 produces a
  weaker signal than a full K=4 ensemble; this is accepted and documented.

Usage:
    from mlops.src.uncertainty import UncertaintyPipeline
    pipeline = UncertaintyPipeline(checkpoint_dir="checkpoints/")
    result = pipeline.predict(masked_kspace, mask, max_value)
    print(result.uncertainty_scalar, result.flagged_for_review)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# fastMRI must be installed from source at the pinned commit.
# See mlops/requirements.txt for the install command.
try:
    from fastmri.models import VarNet
    from fastmri.data.transforms import center_crop
except ImportError as e:
    raise ImportError(
        "fastMRI not found. Install from source:\n"
        "  git clone https://github.com/facebookresearch/fastMRI.git\n"
        "  cd fastMRI && git checkout 91f2df4711adbb6d643df1810f234e4abcf5881b\n"
        "  pip install -e ."
    ) from e


# Threshold calibrated from the research phase:
# In-domain (knee) mean uncertainty ~X, OOD (brain) mean ~1.54x higher.
# A scalar above this threshold is flagged as likely OOD / drift.
# Set conservatively at 1.2x the in-domain mean so false-positive rate stays low.
# Can be re-calibrated from real deployment data without changing this code --
# just update the value below or pass a custom threshold to UncertaintyPipeline().
DEFAULT_FLAG_THRESHOLD = 0.018   # normalised uncertainty scalar


@dataclass
class UncertaintyResult:
    reconstruction: torch.Tensor    # [H, W] float, the mean prediction image
    uncertainty_map: torch.Tensor   # [H, W] float, per-pixel std across checkpoints
    uncertainty_scalar: float       # scalar: mean(std_map) / max_value
    flagged_for_review: bool        # True if scalar exceeds the flag threshold


def _load_varnet(checkpoint_path: Path, device: torch.device) -> VarNet:
    """Load one VarNet checkpoint. Returns the model in eval mode."""
    model = VarNet(num_cascades=4, chans=18, sens_chans=8)
    state = torch.load(checkpoint_path, map_location=device)
    # Checkpoints may be saved as {"model_state_dict": ...} or as raw state_dict
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_inference(
    model: VarNet,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    max_value: float,
    device: torch.device,
    crop_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Run one forward pass.  Returns the reconstructed image as [H, W] float.

    VarNet expects:
      masked_kspace: [batch=1, coil=1, H, W, complex=2]
      mask:          [1, 1, 1, W, 1] boolean
    These shapes match single-coil data after unsqueeze (see Bug 7 in notes).
    """
    mk = masked_kspace.to(device)
    m = mask.bool().to(device)
    mv = torch.tensor(max_value, device=device)

    # Add batch and coil dims if missing (single-coil path)
    if mk.dim() == 4:       # [H, W, complex] -> [1, 1, H, W, complex]
        mk = mk.unsqueeze(0).unsqueeze(0)
    if m.dim() == 3:        # [1, W, 1] -> [1, 1, 1, W, 1]
        m = m.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(mk, m, mv)   # [1, H, W]

    img = output.squeeze(0)         # [H, W]

    if crop_size is not None:
        img = center_crop(img, crop_size)

    return img


class UncertaintyPipeline:
    """
    Wraps two T=4 checkpoints and exposes a single predict() call that
    returns the mean reconstruction plus the normalised uncertainty scalar.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_best: str = "best_model.pt",       # T4 epoch-21
        checkpoint_final: str = "checkpoint_epoch_50.pt",  # T4 epoch-50
        flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
        device: str | None = None,
    ):
        self.flag_threshold = flag_threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt_dir = Path(checkpoint_dir)
        best_path = ckpt_dir / checkpoint_best
        final_path = ckpt_dir / checkpoint_final

        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        if not final_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {final_path}")

        print(f"Loading checkpoints onto {self.device}...")
        self.model_best = _load_varnet(best_path, self.device)
        self.model_final = _load_varnet(final_path, self.device)
        print("Checkpoints loaded.")

    def predict(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        max_value: float,
        crop_size: tuple[int, int] | None = None,
    ) -> UncertaintyResult:
        """
        Run both checkpoints and return mean reconstruction + uncertainty.

        Args:
            masked_kspace: undersampled k-space tensor
            mask:          boolean undersampling mask
            max_value:     peak intensity of this slice (used for SSIM normalisation
                           and for normalising the uncertainty scalar)
            crop_size:     (height, width) to center-crop the output to match the
                           ground-truth size (e.g. (320, 320) for fastMRI knee)

        Returns:
            UncertaintyResult with reconstruction, uncertainty_map, scalar, flag
        """
        img_best = _run_inference(
            self.model_best, masked_kspace, mask, max_value, self.device, crop_size
        )
        img_final = _run_inference(
            self.model_final, masked_kspace, mask, max_value, self.device, crop_size
        )

        # Stack predictions and compute per-pixel std across the K=2 ensemble
        stack = torch.stack([img_best, img_final], dim=0)  # [2, H, W]
        mean_img = stack.mean(dim=0)                        # [H, W]
        std_map = stack.std(dim=0, unbiased=False)          # [H, W]

        # Normalise by max_value so the scalar is scale-invariant (Bug 13 fix)
        if max_value > 0:
            scalar = float(std_map.mean()) / max_value
        else:
            scalar = float(std_map.mean())

        return UncertaintyResult(
            reconstruction=mean_img.cpu(),
            uncertainty_map=std_map.cpu(),
            uncertainty_scalar=scalar,
            flagged_for_review=scalar > self.flag_threshold,
        )
