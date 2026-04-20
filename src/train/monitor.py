"""Training monitor -- WandB + MLFlow callback and diagnostic utilities.

Provides:
- ``TrainingMonitor``: a HuggingFace ``TrainerCallback`` that pushes custom
  metrics (GPU stats, overfitting flags) to both WandB and MLFlow during training.
- ``plot_loss_curve``: fetch a completed run's loss history from WandB and
  render a Matplotlib chart.
- ``check_overfitting``: lightweight heuristic comparing train/val loss.
- ``log_gpu_stats``: snapshot GPU memory and utilisation via ``pynvml`` /
  ``torch.cuda``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import yaml
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLFlow helpers
# ---------------------------------------------------------------------------


def _mlflow_active() -> bool:
    """Check if MLFlow is available and a run is active.

    Returns:
        ``True`` if MLFlow is installed and an active run exists.
    """
    try:
        import mlflow

        return mlflow.active_run() is not None
    except ImportError:
        return False


def _mlflow_log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLFlow if a run is active.

    Args:
        metrics: Mapping of metric names to float values.
        step: Optional global step number for the logged metrics.
    """
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def _wandb_log(metrics: dict[str, object]) -> None:
    """Log metrics to WandB if a run is active.

    Args:
        metrics: Mapping of metric names to values to log.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


def setup_mlflow(config_path: str = "config/base.yaml") -> None:
    """Initialize MLFlow tracking from config.

    Reads the ``mlflow`` section from ``config/base.yaml`` and sets up
    the tracking URI and experiment. Call this before training starts.

    Args:
        config_path: Path to the base configuration YAML file.

    Note:
        If the MLFlow server is not running, this function logs a warning
        and continues without MLFlow tracking. Training will proceed normally.
    """
    try:
        import mlflow
    except ImportError:
        logger.debug("mlflow not installed -- skipping MLFlow setup.")
        return

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f).get("mlflow", {})
    except FileNotFoundError:
        cfg = {}

    tracking_uri = cfg.get(
        "tracking_uri",
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    experiment_name = cfg.get("experiment_name", "model-tailor")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLFlow configured: uri=%s  experiment=%s", tracking_uri, experiment_name)
    except Exception as e:
        logger.warning(
            "Could not connect to MLFlow server at %s: %s. Training will continue without MLFlow tracking.",
            tracking_uri,
            e,
        )


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def log_gpu_stats() -> dict[str, float]:
    """Return a dict of GPU memory and utilisation metrics.

    Falls back gracefully when CUDA is unavailable (e.g. CI or CPU-only
    machines). When available the stats are also pushed to WandB and MLFlow.

    Returns:
        Dictionary with keys like ``gpu/memory_allocated_gb``,
        ``gpu/memory_reserved_gb``, ``gpu/utilisation_pct``.
    """
    stats: dict[str, float] = {}

    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("CUDA not available -- skipping GPU stats.")
            return stats

        device = torch.cuda.current_device()
        stats["gpu/memory_allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1024**3, 3)
        stats["gpu/memory_reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1024**3, 3)
        stats["gpu/max_memory_allocated_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 3
        )
    except Exception:
        logger.debug("Could not read torch.cuda memory stats.", exc_info=True)

    # Utilisation % requires pynvml (ships with nvidia-ml-py3)
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)
        stats["gpu/utilisation_pct"] = float(util.gpu)
        stats["gpu/memory_utilisation_pct"] = float(util.memory)
        nvmlShutdown()
    except Exception:
        logger.debug("pynvml unavailable -- GPU utilisation not logged.", exc_info=True)

    # Push to both WandB and MLFlow
    if stats:
        _wandb_log(stats)
        _mlflow_log_metrics(stats)

    return stats


# ---------------------------------------------------------------------------
# Overfitting heuristic
# ---------------------------------------------------------------------------


def check_overfitting(
    train_loss: float,
    val_loss: float,
    threshold: float = 0.1,
) -> bool:
    """Check whether the model appears to be overfitting.

    Returns ``True`` when the gap between validation and training loss
    exceeds the threshold.

    Args:
        train_loss: Current training loss.
        val_loss: Current validation / evaluation loss.
        threshold: Maximum tolerable gap. The default of 0.1 works for most
            instruction-tuning runs on 8B models with a few thousand examples.

    Returns:
        ``True`` if the model appears to be overfitting.
    """
    gap = val_loss - train_loss
    is_overfitting = gap > threshold

    if is_overfitting:
        logger.warning(
            "Possible overfitting detected: train_loss=%.4f  val_loss=%.4f  "
            "gap=%.4f (threshold=%.4f)",
            train_loss,
            val_loss,
            gap,
            threshold,
        )

    return is_overfitting


# ---------------------------------------------------------------------------
# Loss-curve plotting
# ---------------------------------------------------------------------------


def plot_loss_curve(
    run_path: str,
    output_file: Optional[str] = None,
    keys: Optional[list[str]] = None,
) -> object:
    """Fetch loss history from a finished WandB run and plot it.

    Args:
        run_path: WandB run path in ``"entity/project/run_id"`` form.
        output_file: If provided the figure is saved to this path
            (e.g. ``"loss.png"``). Otherwise ``plt.show()`` is called.
        keys: Metric keys to plot. Defaults to
            ``["train/loss", "eval/loss"]``.

    Returns:
        The rendered ``matplotlib.figure.Figure`` object.
    """
    import matplotlib.pyplot as plt
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    if keys is None:
        keys = ["train/loss", "eval/loss"]

    # history() returns a pandas-like list of dicts with one row per log step
    history = run.history(keys=keys + ["_step"], pandas=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in keys:
        if key in history.columns:
            series = history[["_step", key]].dropna()
            ax.plot(series["_step"], series[key], label=key)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve -- {run.name} ({run.id})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150)
        logger.info("Loss curve saved to %s", output_file)
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------


class TrainingMonitor(TrainerCallback):
    """HuggingFace ``TrainerCallback`` that logs to both WandB and MLFlow.

    Automatically attached by ``TrainingRunner``. At each logging step it:

    * Records GPU memory / utilisation via :func:`log_gpu_stats`.
    * Checks for overfitting after every evaluation step.
    * Logs a concise summary table at the end of training.

    All metrics are dual-logged to WandB (real-time dashboards) and MLFlow
    (model registry and experiment comparison).
    """

    def __init__(self, overfitting_threshold: float = 0.1) -> None:
        """Initialize the training monitor.

        Args:
            overfitting_threshold: Maximum tolerable gap between validation
                and training loss before flagging overfitting.
        """
        super().__init__()
        self.overfitting_threshold = overfitting_threshold
        self._train_losses: list[float] = []
        self._eval_losses: list[float] = []
        self._overfitting_flagged: bool = False
        self._global_step: int = 0

    # -- Training start -----------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Start an MLFlow run and log training parameters.

        Args:
            args: The training arguments for the current run.
            state: The current trainer state.
            control: The trainer control object.
            **kwargs: Additional keyword arguments forwarded by the Trainer.
        """
        try:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run()

            mlflow.log_params(
                {
                    "learning_rate": args.learning_rate,
                    "num_epochs": args.num_train_epochs,
                    "batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation": args.gradient_accumulation_steps,
                    "warmup_ratio": args.warmup_ratio,
                    "weight_decay": args.weight_decay,
                    "lr_scheduler": args.lr_scheduler_type.value
                    if hasattr(args.lr_scheduler_type, "value")
                    else str(args.lr_scheduler_type),
                }
            )
        except Exception:
            logger.debug("Could not log params to MLFlow.", exc_info=True)

    # -- Logging step -------------------------------------------------------

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        """Log GPU stats and training metrics to both WandB and MLFlow.

        Called every ``logging_steps`` by the Trainer.

        Args:
            args: The training arguments for the current run.
            state: The current trainer state.
            control: The trainer control object.
            logs: Dictionary of metrics produced at this logging step.
            **kwargs: Additional keyword arguments forwarded by the Trainer.
        """
        if logs is None:
            return

        self._global_step = state.global_step

        # Track train loss
        if "loss" in logs:
            self._train_losses.append(logs["loss"])
            _mlflow_log_metrics({"train/loss": logs["loss"]}, step=state.global_step)

        if "learning_rate" in logs:
            _mlflow_log_metrics({"learning_rate": logs["learning_rate"]}, step=state.global_step)

        # GPU snapshot (pushes to both WandB and MLFlow internally)
        log_gpu_stats()

    # -- Evaluation step ----------------------------------------------------

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        """Run the overfitting check after every evaluation.

        Logs results to both WandB and MLFlow.

        Args:
            args: The training arguments for the current run.
            state: The current trainer state.
            control: The trainer control object.
            metrics: Evaluation metrics dictionary from the Trainer.
            **kwargs: Additional keyword arguments forwarded by the Trainer.
        """
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            self._eval_losses.append(eval_loss)
            _mlflow_log_metrics({"eval/loss": eval_loss}, step=state.global_step)

        # Compare latest train loss with eval loss
        if self._train_losses and eval_loss is not None:
            latest_train = self._train_losses[-1]
            is_overfitting = check_overfitting(latest_train, eval_loss, self.overfitting_threshold)

            monitor_metrics = {
                "monitor/train_eval_gap": eval_loss - latest_train,
                "monitor/overfitting_flag": int(is_overfitting),
            }
            _wandb_log(monitor_metrics)
            _mlflow_log_metrics(monitor_metrics, step=state.global_step)

            if is_overfitting:
                self._overfitting_flagged = True

    # -- Training end -------------------------------------------------------

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Log a summary when training completes to both WandB and MLFlow.

        Args:
            args: The training arguments for the current run.
            state: The current trainer state.
            control: The trainer control object.
            **kwargs: Additional keyword arguments forwarded by the Trainer.
        """
        summary: dict[str, object] = {
            "monitor/total_train_loss_points": len(self._train_losses),
            "monitor/total_eval_loss_points": len(self._eval_losses),
            "monitor/overfitting_ever_flagged": self._overfitting_flagged,
        }

        if self._train_losses:
            summary["monitor/final_train_loss"] = self._train_losses[-1]
        if self._eval_losses:
            summary["monitor/final_eval_loss"] = self._eval_losses[-1]

        # WandB summary
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(summary)
                wandb.summary.update(summary)
        except Exception:
            pass

        # MLFlow summary
        try:
            import mlflow

            if mlflow.active_run() is not None:
                float_summary = {
                    k: float(v) for k, v in summary.items() if isinstance(v, (int, float, bool))
                }
                mlflow.log_metrics(float_summary)
                mlflow.end_run()
        except Exception:
            logger.debug("Could not log summary to MLFlow.", exc_info=True)

        logger.info("Training complete.  Summary: %s", summary)

    # -- Convenience --------------------------------------------------------

    @staticmethod
    def log_gpu_stats() -> dict[str, float]:
        """Proxy so callers can do ``monitor.log_gpu_stats()``.

        Returns:
            Dictionary of GPU memory and utilisation metrics.
        """
        return log_gpu_stats()
