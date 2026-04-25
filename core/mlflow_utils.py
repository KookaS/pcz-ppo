"""mlflow_utils.py: MLflow integration for PCZ-PPO training.

Provides:
    - ``_MLflowKVWriter``: SB3 KVWriter that forwards metrics to MLflow.
    - ``MLflowCallback``: SB3 callback that installs the KVWriter.
    - ``setup_mlflow`` / ``teardown_mlflow``: Run lifecycle management.

Usage::

    from core.mlflow_utils import MLflowCallback, setup_mlflow, teardown_mlflow

    mlflow_active = setup_mlflow(tracking_uri, experiment, run_name, params)
    callbacks = [MLflowCallback()] if mlflow_active else []
    model.learn(total_timesteps=N, callback=callbacks)
    teardown_mlflow(mlflow_active, run_dir, eval_metrics)
"""

import logging
import os
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

logger = logging.getLogger(__name__)


class _MLflowKVWriter(KVWriter):
    """SB3 KVWriter that forwards metrics to an active MLflow run.

    Installed by ``MLflowCallback`` at training start.  Gets called on
    every ``logger.dump()`` with all recorded key-value pairs and the
    current timestep, so we capture both train/* and rollout/* metrics
    at exactly the right moment.
    """

    def __init__(self, mlflow_module):
        self._mlflow = mlflow_module

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for key, value in key_values.items():
            # Skip non-numeric (e.g. time/total_timesteps as string)
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Skip time/* metrics — they're just wall-clock / fps
                if not key.startswith("time/"):
                    metrics[key] = value
        if metrics:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass  # best-effort, don't break training

    def close(self):
        pass


class MLflowCallback(BaseCallback):
    """Install an MLflow KVWriter into SB3's logger at training start.

    This hooks into SB3's ``logger.dump()`` to capture all train/* and
    rollout/* metrics with correct step indexing.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        try:
            import mlflow

            if mlflow.active_run() is not None:
                writer = _MLflowKVWriter(mlflow)
                self.model.logger.output_formats.append(writer)
        except ImportError:
            pass

    def _on_step(self) -> bool:
        return True


def setup_mlflow(
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str | None,
    params: dict[str, Any],
) -> bool:
    """Configure MLflow tracking. Returns True if MLflow is active."""
    if tracking_uri is None:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        print(f"  MLflow: {tracking_uri} | experiment={experiment_name}")
        return True
    except Exception as e:
        logger.warning("MLflow setup failed: %s — continuing without MLflow.", e)
        return False


def teardown_mlflow(
    mlflow_active: bool,
    run_dir: str,
    metrics: dict[str, float] | None = None,
):
    """End MLflow run, log final metrics and artifacts."""
    if not mlflow_active:
        return
    try:
        import mlflow

        if metrics:
            mlflow.log_metrics(metrics)
        best_dir = os.path.join(run_dir, "best_model")
        if os.path.isdir(best_dir):
            mlflow.log_artifacts(best_dir, artifact_path="best_model")
        mlflow.end_run()
    except Exception as e:
        logger.warning("MLflow teardown failed: %s", e)
