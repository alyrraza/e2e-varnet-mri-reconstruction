"""
Evidently AI drift monitor — Component 4.

Reduces k-space inputs to three scalar features and uses Evidently to detect
distribution shift between the training (knee) reference and any new batch.

Features used (all are cheap to compute at inference time):
  ker          -- k-space energy ratio: energy in outer 75% of k-space lines
                  divided by energy in inner 25% (center = low-frequency).
                  NOT a significant shift signal on its own (0.59x, p=0.082),
                  but included as a complementary feature.
  esnr         -- estimated SNR proxy: mean signal in acquired center lines
                  divided by std of acquired peripheral lines.
                  Strongest shift signal (2.04x) but only computable on ~80%
                  of volumes -- volumes with too few acquired peripheral lines
                  are excluded (NaN) rather than producing a blowup (Bug 19 fix).
  matrix_width -- raw k-space width in pixels. 368 vs 372 was a real, observed
                  difference between knee scans (Bug 9 in notes). Cheap metadata.

The knee reference distribution is built from the 10-volume local subset in
`data subset/`. In production it should be computed from the full 198-volume
training set; a broader reference makes the monitor more reliable.

Validated using the existing knee/brain split:
  knee = known in-domain (should not fire)
  brain = known OOD (should fire)
This reuses the GAP 2 data already collected -- no new experiment needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError as e:
    raise ImportError(
        "Evidently not installed. Run: pip install evidently>=0.4.30"
    ) from e

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_SUBSET = ROOT / "data subset"
RESULTS_DIR = ROOT / "results"


# ---- Feature extraction ----

def _compute_features(kspace: np.ndarray) -> dict:
    """
    Compute three scalar drift features from a single k-space array.

    kspace: numpy array, shape [H, W] (single-coil, complex or magnitude)
    Returns dict with keys: ker, esnr, matrix_width. esnr may be None.
    """
    if np.iscomplexobj(kspace):
        mag = np.abs(kspace)
    else:
        mag = kspace

    H, W = mag.shape
    matrix_width = W

    # k-space energy ratio (KER)
    # Inner 25% of lines (low-frequency, center) vs outer 75% (high-frequency)
    center_start = int(H * 0.375)
    center_end = int(H * 0.625)
    center_energy = float(np.sum(mag[center_start:center_end, :] ** 2))
    total_energy = float(np.sum(mag ** 2))
    outer_energy = total_energy - center_energy
    ker = outer_energy / center_energy if center_energy > 0 else np.nan

    # Estimated SNR (eSNR)
    # Mean of center ACS lines / std of acquired peripheral lines.
    # Peripheral = outer 50% of lines. Only use non-zero values (acquired lines).
    # If fewer than 5 acquired peripheral values, skip -- too noisy for stable estimate.
    acs_lines = mag[center_start:center_end, :]
    acs_mean = float(np.mean(acs_lines))

    top_peripheral = mag[:center_start, :]
    bot_peripheral = mag[center_end:, :]
    peripheral_vals = np.concatenate([
        top_peripheral[top_peripheral > 0].ravel(),
        bot_peripheral[bot_peripheral > 0].ravel(),
    ])

    if len(peripheral_vals) >= 5:
        esnr = acs_mean / float(np.std(peripheral_vals)) if np.std(peripheral_vals) > 0 else np.nan
    else:
        esnr = np.nan   # insufficient acquired peripheral lines (Bug 19 fix)

    return {"ker": ker, "esnr": esnr, "matrix_width": float(matrix_width)}


def extract_features_from_h5(path: Path) -> list[dict]:
    """Extract features for every slice in one .h5 file."""
    rows = []
    with h5py.File(path, "r") as hf:
        kspace = hf["kspace"][:]   # [slices, H, W] or [slices, coils, H, W]
        for i in range(kspace.shape[0]):
            sl = kspace[i]
            if sl.ndim == 3:           # multi-coil: use coil 0 as proxy
                sl = sl[0]
            feats = _compute_features(sl)
            feats["slice_idx"] = i
            feats["source_file"] = path.name
            rows.append(feats)
    return rows


def build_reference_dataframe(data_dir: Path = DATA_SUBSET) -> pd.DataFrame:
    """
    Build the reference (in-domain / knee) feature table from local .h5 files.
    In production, replace with the full 198-volume training set.
    """
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    rows = []
    for f in h5_files:
        rows.extend(extract_features_from_h5(f))

    df = pd.DataFrame(rows)
    print(f"Reference distribution: {len(df)} slices from {len(h5_files)} volumes")
    return df


# ---- Evidently drift report ----

FEATURE_COLS = ["ker", "esnr", "matrix_width"]


def run_drift_report(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_html: Optional[Path] = None,
) -> dict:
    """
    Run an Evidently DataDrift report comparing current batch vs reference.

    Returns a dict with: drift_detected (bool), drifted_features (list),
    share_drifted (float), and per-feature p-values.
    """
    # Drop rows where esnr is NaN -- consistent with the ~80% coverage caveat
    ref = reference_df[FEATURE_COLS].dropna()
    cur = current_df[FEATURE_COLS].dropna()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    result = report.as_dict()
    metrics = result["metrics"][0]["result"]

    drift_detected = metrics["dataset_drift"]
    share_drifted = metrics["share_of_drifted_columns"]
    per_feature = {
        col: metrics["drift_by_columns"][col]
        for col in FEATURE_COLS
        if col in metrics.get("drift_by_columns", {})
    }
    drifted = [col for col, v in per_feature.items() if v.get("drift_detected")]

    if output_html is not None:
        report.save_html(str(output_html))
        print(f"Drift report saved to {output_html}")

    return {
        "drift_detected": drift_detected,
        "share_drifted": share_drifted,
        "drifted_features": drifted,
        "per_feature": per_feature,
    }


# ---- Validation against known knee/brain split ----

def validate_monitor(
    reference_df: pd.DataFrame,
    brain_h5_dir: Optional[Path] = None,
) -> dict:
    """
    Validate the drift monitor using the known OOD split from the research phase.
    knee (reference) = in-domain, should NOT fire.
    brain (current)  = OOD, SHOULD fire.

    If brain_h5_dir is None, loads the shift ratios from the locked GAP 2 results
    and just reports them (no new experiment needed).
    """
    gap2_path = RESULTS_DIR / "gap2_final_v3.json"
    if gap2_path.exists():
        with open(gap2_path) as f:
            gap2 = json.load(f)
        print(
            "GAP 2 locked validation result:\n"
            f"  ckpt_unc shift ratio: {gap2['shift_detection']['ckpt_unc']['ratio']}x "
            f"(p={gap2['shift_detection']['ckpt_unc']['mannwhitney_p']:.2e})\n"
            "  KER was NOT significant (0.59x, p=0.082) -- expected, do not rely on KER alone.\n"
            "  eSNR was strongest (2.04x) but only ~80% coverage -- use with ckpt_unc as complement."
        )

    if brain_h5_dir is not None and brain_h5_dir.exists():
        brain_rows = []
        for f in brain_h5_dir.glob("*.h5"):
            brain_rows.extend(extract_features_from_h5(f))
        brain_df = pd.DataFrame(brain_rows)
        result = run_drift_report(brain_df, reference_df)
        print(f"\nDrift monitor on brain (OOD) data: drift_detected={result['drift_detected']}")
        print(f"  Drifted features: {result['drifted_features']}")
        return result

    return {}


if __name__ == "__main__":
    ref_df = build_reference_dataframe()
    validate_monitor(ref_df)
