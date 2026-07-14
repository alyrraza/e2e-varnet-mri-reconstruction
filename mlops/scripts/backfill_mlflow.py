"""
MLflow backfill script — Component 1.

Logs historical training metrics for T4/T6/T8 models into MLflow and
registers each checkpoint in the Model Registry using aliases (not the
deprecated Staging/Production stages, which were removed in MLflow 2.9.0).

Run once from the repo root:
    python mlops/scripts/backfill_mlflow.py

MLflow UI to inspect results:
    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db
"""

import json
import sys
from pathlib import Path

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
MLFLOW_DB = ROOT / "mlops" / "mlflow.db"

# Official locked numbers from PROJECT_NOTES_FINAL.md.
# Do NOT read knee_ssim from gap1_results.json directly -- that file stores
# the deprecated 0.7637 for T4 (old pooled-slice average). Volume-level
# mean SSIM computed in a single session is the sole official convention.
MODELS = {
    "T4": {
        "num_cascades": 4,
        "chans": 18,
        "sens_chans": 8,
        "lr": 3e-4,
        "epochs_trained": 50,
        "best_epoch": 21,
        "knee_ssim": 0.7594,
        "brain_ssim": 0.6622,
        "checkpoint_best": "best_model.pt",       # epoch 21
        "checkpoint_final": "checkpoint_epoch_50.pt",  # epoch 50
    },
    "T6": {
        "num_cascades": 6,
        "chans": 18,
        "sens_chans": 8,
        "lr": 3e-4,
        "epochs_trained": 25,   # stopped early; plateau confirmed at epoch 21-22
        "best_epoch": 21,
        "knee_ssim": 0.7606,
        "brain_ssim": 0.6705,
        "checkpoint_best": "t6_best_model.pt",
        "checkpoint_final": None,
    },
    "T8": {
        "num_cascades": 8,
        "chans": 18,
        "sens_chans": 8,
        "lr": 3e-4,
        "epochs_trained": 24,   # stopped early; plateau confirmed at epoch 20-22
        "best_epoch": 22,
        "knee_ssim": 0.7607,
        "brain_ssim": 0.6773,
        "checkpoint_best": "t8_best_model.pt",
        "checkpoint_final": None,
    },
}

ZERO_FILLED = {"knee_ssim": 0.7453, "brain_ssim": 0.4153}

# GAP 2 headline numbers (volume-clustered bootstrap, checkpoint-ensemble K=2)
GAP2 = {
    "shift_ratio": 1.54,
    "shift_ci_low": 1.35,
    "shift_ci_high": 1.74,
    "shift_mannwhitney_p": 4.46e-6,
    "error_pred_r_indomain": 0.4317,
    "error_pred_ci_low": 0.3588,
    "error_pred_ci_high": 0.5027,
}

EXPERIMENT_NAME = "e2e-varnet-mri-reconstruction"


def load_gap2_extended():
    """Load trivial-baseline comparison from gap2_final_v3.json."""
    path = RESULTS / "gap2_final_v3.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def log_model_run(client, experiment_id, name, info):
    """Create one MLflow run with all params + metrics for a model variant."""
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"backfill-{name.lower()}",
        tags={"source": "backfill", "model_variant": name},
    ) as run:
        mlflow.log_params({
            "num_cascades": info["num_cascades"],
            "chans": info["chans"],
            "sens_chans": info["sens_chans"],
            "lr": info["lr"],
            "batch_size": 1,                   # structural requirement of fastMRI dataset
            "grad_accum_steps": 8,             # effective batch = 8
            "loss": "1-SSIM",
            "optimizer": "Adam",
            "acceleration": "4x",
            "center_fractions": 0.08,
            "mask_type": "EquispacedMaskFractionFunc",
            "dataset": "fastMRI_single_coil_knee",
            "fastmri_commit": "91f2df4711adbb6d643df1810f234e4abcf5881b",
            "epochs_trained": info["epochs_trained"],
            "best_epoch": info["best_epoch"],
            "ssim_averaging": "volume_level_mean",  # official convention, locked
        })

        mlflow.log_metrics({
            "knee_ssim": info["knee_ssim"],
            "brain_ssim_ood": info["brain_ssim"],
            "zero_filled_knee_ssim": ZERO_FILLED["knee_ssim"],
            "zero_filled_brain_ssim": ZERO_FILLED["brain_ssim"],
            "knee_gain_over_zf": round(info["knee_ssim"] - ZERO_FILLED["knee_ssim"], 4),
            "brain_gain_over_zf": round(info["brain_ssim"] - ZERO_FILLED["brain_ssim"], 4),
        })

        # GAP 2 metrics are specific to T4 (ensemble uses T4's two checkpoints)
        if name == "T4":
            mlflow.log_metrics({
                "ckpt_unc_shift_ratio": GAP2["shift_ratio"],
                "ckpt_unc_shift_ci_low": GAP2["shift_ci_low"],
                "ckpt_unc_shift_ci_high": GAP2["shift_ci_high"],
                "ckpt_unc_shift_p": GAP2["shift_mannwhitney_p"],
                "ckpt_unc_error_pred_r": GAP2["error_pred_r_indomain"],
                "ckpt_unc_error_pred_ci_low": GAP2["error_pred_ci_low"],
                "ckpt_unc_error_pred_ci_high": GAP2["error_pred_ci_high"],
            })

            gap2_ext = load_gap2_extended()
            if gap2_ext:
                ep = gap2_ext.get("error_prediction", {})
                sd = gap2_ext.get("shift_detection", {})
                if ep:
                    mlflow.log_metrics({
                        "zf_residual_error_pred_r": ep.get("zf_residual", {}).get("r", 0),
                        "periphery_ratio_error_pred_r": ep.get("periphery_ratio", {}).get("r", 0),
                    })
                if sd:
                    mlflow.log_metrics({
                        "zf_residual_shift_ratio": sd.get("zf_residual", {}).get("ratio", 0),
                    })

        # Log checkpoint file as a raw artifact so it is traceable via this run
        ckpt_best = CHECKPOINTS / info["checkpoint_best"]
        if ckpt_best.exists():
            mlflow.log_artifact(str(ckpt_best), artifact_path="checkpoints")
        else:
            print(f"  WARNING: {ckpt_best} not found, skipping artifact log")

        if info.get("checkpoint_final"):
            ckpt_final = CHECKPOINTS / info["checkpoint_final"]
            if ckpt_final.exists():
                mlflow.log_artifact(str(ckpt_final), artifact_path="checkpoints")

        print(f"  Logged run {run.info.run_id} for {name}")
        return run.info.run_id


def register_checkpoint(client, run_id, model_name, checkpoint_filename, alias=None):
    """Register one checkpoint file in the MLflow Model Registry."""
    ckpt_path = CHECKPOINTS / checkpoint_filename
    if not ckpt_path.exists():
        print(f"  SKIP: {ckpt_path} not found")
        return

    # Create the registered model if it does not exist yet
    try:
        client.create_registered_model(
            model_name,
            tags={"framework": "pytorch", "architecture": "E2E-VarNet"},
            description=(
                "E2E-VarNet (Sriram et al., MICCAI 2020) trained on fastMRI "
                "single-coil knee, 4x acceleration. "
                "Official SSIM numbers are volume-level means from a single session."
            ),
        )
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    # Source URI points to the artifact logged in the run
    source = f"runs:/{run_id}/checkpoints/{checkpoint_filename}"
    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        description=f"Checkpoint file: {checkpoint_filename}",
    )

    if alias:
        client.set_registered_model_alias(model_name, alias, mv.version)
        print(f"  Registered {model_name} v{mv.version} @{alias}")
    else:
        print(f"  Registered {model_name} v{mv.version}")


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = MlflowClient()

    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{EXPERIMENT_NAME}' (id={experiment_id})")

    # Log one run per model variant
    print("\n--- Logging runs ---")
    run_ids = {}
    for name, info in MODELS.items():
        print(f"\n[{name}]")
        run_ids[name] = log_model_run(client, experiment_id, name, info)

    # Register models in the MLflow Model Registry
    print("\n--- Registering models ---")

    # T4 epoch-21 best -> champion
    register_checkpoint(
        client, run_ids["T4"], "varnet-t4",
        MODELS["T4"]["checkpoint_best"], alias="champion"
    )
    # T4 epoch-50 final -> needed for the K=2 checkpoint ensemble (Component 2)
    register_checkpoint(
        client, run_ids["T4"], "varnet-t4",
        MODELS["T4"]["checkpoint_final"]
    )
    # T6 and T8 best
    register_checkpoint(client, run_ids["T6"], "varnet-t6", MODELS["T6"]["checkpoint_best"])
    register_checkpoint(client, run_ids["T8"], "varnet-t8", MODELS["T8"]["checkpoint_best"])

    print(
        f"\nDone. View results:\n"
        f"  mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}"
    )


if __name__ == "__main__":
    main()
