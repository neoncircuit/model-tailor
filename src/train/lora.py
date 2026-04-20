"""LoRA / QLoRA configuration and model loading via Unsloth.

Handles:
- LoRA hyperparameter management per model family
- 4-bit quantized model loading with Unsloth's FastLanguageModel
- LoRA adapter application to attention + MLP projections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Target-module presets per model family
# ---------------------------------------------------------------------------

_TARGET_MODULES: dict[str, list[str]] = {
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mistral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "phi": [
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2",
    ],
}

# Maximum sequence lengths that fit comfortably in ~24 GB VRAM at 4-bit.
_DEFAULT_MAX_SEQ_LEN: dict[str, int] = {
    "llama": 4096,
    "mistral": 4096,
    "phi": 2048,
}


@dataclass
class LoRAConfig:
    """All knobs needed to attach LoRA adapters to a base model."""

    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(_TARGET_MODULES["llama"]))
    bias: str = "none"
    use_gradient_checkpointing: bool | str = "unsloth"  # "unsloth" enables Unsloth's optimised GC
    use_rslora: bool = False  # rank-stabilised LoRA (scales alpha by sqrt(rank))
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    random_state: int = 42

    @property
    def lora_alpha_ratio(self) -> float:
        """Effective scaling factor applied to LoRA updates (alpha / rank).

        Returns:
            The ratio of alpha to rank.
        """
        return self.alpha / self.rank


def get_default_config(
    model_family: str = "llama",
    config_path: str = "config/base.yaml",
) -> LoRAConfig:
    """Return a sensible ``LoRAConfig`` for *model_family*.

    Optionally reads overrides from the project YAML.

    Args:
        model_family: One of ``"llama"``, ``"mistral"``, ``"phi"``.
        config_path: Path to the base YAML config file. Values under the
            ``defaults`` key override the hard-coded defaults.

    Returns:
        Ready-to-use ``LoRAConfig`` dataclass.

    Raises:
        ValueError: If *model_family* is not a recognised family name.
    """
    family = model_family.lower()
    if family not in _TARGET_MODULES:
        raise ValueError(
            f"Unknown model family '{model_family}'. Supported: {list(_TARGET_MODULES.keys())}"
        )

    # Start with hard-coded defaults for the family
    kwargs: dict = {
        "target_modules": list(_TARGET_MODULES[family]),
        "max_seq_length": _DEFAULT_MAX_SEQ_LEN[family],
    }

    # Layer on YAML overrides if the file is available
    try:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh).get("defaults", {})
        if "lora_rank" in cfg:
            kwargs["rank"] = int(cfg["lora_rank"])
        if "lora_alpha" in cfg:
            kwargs["alpha"] = int(cfg["lora_alpha"])
        if "lora_dropout" in cfg:
            kwargs["dropout"] = float(cfg["lora_dropout"])
    except FileNotFoundError:
        pass  # no config file -- fall through to hard-coded defaults

    return LoRAConfig(**kwargs)


def build_lora_model(
    base_model_name: str,
    lora_config: Optional[LoRAConfig] = None,
    *,
    dtype: Optional[str] = None,
) -> tuple:
    """Load a 4-bit quantised model via Unsloth and attach LoRA adapters.

    Args:
        base_model_name: HuggingFace model ID, e.g.
            ``"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"``.
        lora_config: LoRA hyperparameters. If ``None`` the Llama defaults
            are used.
        dtype: Optional torch dtype override. ``None`` lets Unsloth
            auto-detect.

    Returns:
        A ``(model, tokenizer)`` tuple. The model is PEFT-wrapped and both
        are ready for training.
    """
    from unsloth import FastLanguageModel

    if lora_config is None:
        lora_config = get_default_config()

    # ------------------------------------------------------------------
    # 1. Load the base model in 4-bit (or full precision if requested)
    # ------------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=lora_config.max_seq_length,
        dtype=dtype,
        load_in_4bit=lora_config.load_in_4bit,
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
    )

    return model, tokenizer
