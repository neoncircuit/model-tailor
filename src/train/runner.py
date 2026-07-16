"""Training runner -- sets up and drives SFTTrainer from HuggingFace TRL.

Handles:
- Building ``TrainingArguments`` from the project YAML config
- Constructing an ``SFTTrainer`` with the LoRA-wrapped model
- WandB integration for experiment tracking
- Checkpoint resumption
- LoRA adapter saving (without the full base model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from src.train.monitor import TrainingMonitor


@dataclass
class TrainingConfig:
    """Mirrors the ``defaults`` + ``wandb`` sections of ``config/base.yaml``."""

    # Optimiser / schedule
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Batching
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096

    # Duration
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use num_epochs

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"

    # Precision
    fp16: bool = False
    bf16: bool = True  # A100 / 4090 / H100

    # Misc
    seed: int = 42
    output_dir: str = "checkpoints"
    optim: str = "adamw_8bit"
    dataset_text_field: str = "text"
    packing: bool = False  # True can speed up training for short sequences

    # WandB
    wandb_project: str = "model-tailor"
    wandb_log_model: bool = False

    # Extra TrainingArguments kwargs that don't have explicit fields
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "config/base.yaml") -> "TrainingConfig":
        """Build a ``TrainingConfig`` by merging YAML values on top of defaults.

        Unknown keys under ``training`` are forwarded via *extra*.

        Args:
            path: Path to the project YAML configuration file.

        Returns:
            A ``TrainingConfig`` populated from the YAML file.
        """
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        defaults = raw.get("defaults", {})
        wandb_cfg = raw.get("wandb", {})
        project_cfg = raw.get("project", {})
        training_cfg = raw.get("training", {})  # optional extended section

        # Map YAML keys to dataclass fields
        mapping: dict[str, Any] = {}
        if "learning_rate" in defaults:
            mapping["learning_rate"] = float(defaults["learning_rate"])
        if "batch_size" in defaults:
            mapping["batch_size"] = int(defaults["batch_size"])
        if "num_epochs" in defaults:
            mapping["num_epochs"] = int(defaults["num_epochs"])
        if "seed" in project_cfg:
            mapping["seed"] = int(project_cfg["seed"])
        if "project" in wandb_cfg:
            mapping["wandb_project"] = wandb_cfg["project"]
        if "log_model" in wandb_cfg:
            mapping["wandb_log_model"] = wandb_cfg["log_model"]

        # Pull known fields from the optional ``training`` section
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        extra: dict[str, Any] = {}
        for key, val in training_cfg.items():
            if key in known_fields:
                mapping[key] = val
            else:
                extra[key] = val
        if extra:
            mapping["extra"] = extra

        return cls(**mapping)


class TrainingRunner:
    """Orchestrates a single LoRA fine-tuning run.

    Typical usage::

        from src.train.lora import build_lora_model, get_default_config
        from src.train.runner import TrainingRunner, TrainingConfig

        model, tokenizer = build_lora_model("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
        cfg = TrainingConfig.from_yaml()
        runner = TrainingRunner(model, tokenizer, train_ds, val_ds, cfg)
        runner.train()
        runner.save("models/my-lora-adapter")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize the training runner.

        Args:
            model: A PEFT-wrapped (LoRA) language model ready for training.
            tokenizer: The tokenizer corresponding to *model*.
            train_dataset: Training dataset with a text field.
            val_dataset: Optional validation dataset for evaluation during
                training.
            config: Training hyperparameters. Defaults to ``TrainingConfig()``
                when not provided.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self.monitor = TrainingMonitor()
        self._trainer: Optional[SFTTrainer] = None

        self._setup_wandb()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_wandb(self) -> None:
        """Configure WandB environment variables *before* the Trainer
        initialises its own WandB integration."""
        os.environ["WANDB_PROJECT"] = self.config.wandb_project
        if not self.config.wandb_log_model:
            os.environ["WANDB_LOG_MODEL"] = "false"

    def _build_training_args(self) -> SFTConfig:
        """Translate ``TrainingConfig`` into TRL ``SFTConfig``.

        Returns:
            A fully populated ``SFTConfig`` instance for SFTTrainer.
        """
        cfg = self.config
        kwargs: dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_train_epochs": cfg.num_epochs,
            "max_steps": cfg.max_steps,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "optim": cfg.optim,
            "fp16": cfg.fp16,
            "bf16": cfg.bf16,
            "logging_steps": cfg.logging_steps,
            "eval_strategy": cfg.eval_strategy,
            "eval_steps": cfg.eval_steps,
            "save_strategy": cfg.save_strategy,
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": cfg.report_to,
            "seed": cfg.seed,
            # SFT-specific parameters
            "dataset_text_field": cfg.dataset_text_field,
            "max_length": cfg.max_seq_length,
            "packing": cfg.packing,
            "eos_token": None,  # Let tokenizer use default EOS token
            **cfg.extra,
        }
        return SFTConfig(**kwargs)

    def _build_trainer(self, resume_from: Optional[str] = None) -> SFTTrainer:
        """Construct the ``SFTTrainer``.

        Args:
            resume_from: Optional path to a checkpoint directory. Currently
                reserved for future use.

        Returns:
            A configured ``SFTTrainer`` ready to call ``.train()``.
        """
        training_args = self._build_training_args()

        # Override eos_token to use tokenizer's default
        if hasattr(training_args, "eos_token"):
            training_args.eos_token = self.tokenizer.eos_token

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[self.monitor],
        )
        return trainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict[str, float]:
        """Run the full training loop.

        Args:
            resume_from_checkpoint: Path to a checkpoint directory to resume
                from. When ``True`` (the boolean, not a path), the Trainer
                will automatically find the latest checkpoint inside
                ``output_dir``.

        Returns:
            Training metrics reported by the Trainer (train loss, runtime,
            samples per second, etc.).
        """
        self._trainer = self._build_trainer()
        result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log final GPU statistics
        self.monitor.log_gpu_stats()

        metrics = result.metrics
        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)

        return metrics

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the validation set and return metrics.

        Returns:
            Evaluation metrics dictionary (e.g. ``eval_loss``).

        Raises:
            ValueError: If no validation dataset was provided at init.
        """
        if self._trainer is None:
            self._trainer = self._build_trainer()
        if self.val_dataset is None:
            raise ValueError("No validation dataset provided -- cannot evaluate.")
        metrics = self._trainer.evaluate()
        self._trainer.log_metrics("eval", metrics)
        return metrics

    def save(self, output_dir: str) -> Path:
        """Save only the LoRA adapter weights (not the full base model).

        Args:
            output_dir: Directory to write adapter weights, tokenizer files,
                and config.

        Returns:
            Resolved path to the saved adapter directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save adapter weights via PEFT
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

        return out.resolve()
