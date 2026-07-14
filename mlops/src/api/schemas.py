"""Request and response schemas for the /reconstruct endpoint."""

from pydantic import BaseModel, Field


class ReconstructResponse(BaseModel):
    ssim: float = Field(..., description="SSIM of reconstruction vs zero-filled input")
    psnr: float = Field(..., description="PSNR in dB")
    uncertainty_scalar: float = Field(
        ..., description="Normalised checkpoint-ensemble uncertainty (scale-invariant)"
    )
    flagged_for_review: bool = Field(
        ...,
        description=(
            "True if uncertainty_scalar exceeds the drift threshold, "
            "indicating the input may be out-of-distribution"
        ),
    )
    reconstruction_b64: str = Field(
        ..., description="Reconstructed image as base64-encoded PNG"
    )
    uncertainty_map_b64: str = Field(
        ..., description="Uncertainty map as base64-encoded PNG (normalised 0-1)"
    )
    inference_time_ms: float = Field(..., description="Total inference time in ms")
    model_variant: str = Field(..., description="Which model was used (e.g. T4)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
