"""Model export utilities -- merge LoRA adapters and save in various formats.

Supports exporting to HuggingFace safetensors (``hf``) and GGUF for llama.cpp.
Also provides a helper to push exported models to the HuggingFace Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Recognised export formats.
ExportFormat = Literal["hf", "gguf"]

# GGUF quantisation presets available through Unsloth's save helper.
_GGUF_QUANT_METHODS: dict[str, str] = {
    "f16": "f16",
    "q4_k_m": "q4_k_m",
    "q5_k_m": "q5_k_m",
    "q8_0": "q8_0",
}


def merge_and_export(
    model,
    tokenizer,
    output_dir: str | Path,
    format: ExportFormat = "hf",
    *,
    gguf_quant: str = "q4_k_m",
    push_model: bool = False,
    hub_repo: str | None = None,
    hub_token: str | None = None,
) -> Path:
    """Merge LoRA weights into the base model and export to *output_dir*.

    Args:
        model: An Unsloth / PEFT model with LoRA adapters attached.
        tokenizer: The tokenizer associated with the model.
        output_dir: Destination directory for the exported artefacts.
        format: Export format. ``"hf"`` for HuggingFace safetensors (default),
            ``"gguf"`` for GGUF file for llama.cpp / Ollama.
        gguf_quant: Quantisation method used when *format* is ``"gguf"``.
            One of ``"f16"``, ``"q4_k_m"``, ``"q5_k_m"``, ``"q8_0"``.
            Ignored for HuggingFace exports.
        push_model: If True, push the result to HuggingFace Hub after local
            export.
        hub_repo: Repository id on the Hub (e.g. ``"org/model-name"``).
            Required when *push_model* is True.
        hub_token: HuggingFace API token. Falls back to the cached token from
            ``huggingface-cli login`` when not supplied.

    Returns:
        Resolved path to the export directory.

    Raises:
        ValueError: If *format* is not ``"hf"`` or ``"gguf"``, or if
            *push_model* is True but *hub_repo* is None.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "hf":
        _export_hf(model, tokenizer, output_dir)
    elif format == "gguf":
        _export_gguf(model, tokenizer, output_dir, gguf_quant)
    else:
        raise ValueError(f"Unsupported export format: {format!r}. Choose 'hf' or 'gguf'.")

    if push_model:
        if hub_repo is None:
            raise ValueError("hub_repo is required when push_model=True")
        push_to_hub(str(output_dir), hub_repo, token=hub_token)

    return output_dir


# ---------------------------------------------------------------------------
# Internal export helpers
# ---------------------------------------------------------------------------


def _export_hf(model, tokenizer, output_dir: Path) -> None:
    """Merge LoRA adapters and save as HuggingFace safetensors.

    Args:
        model: An Unsloth / PEFT model with LoRA adapters attached.
        tokenizer: The tokenizer associated with the model.
        output_dir: Destination directory for the exported artefacts.
    """
    logger.info("Merging LoRA adapters and exporting to HuggingFace format -> %s", output_dir)

    # Unsloth exposes a dedicated method that handles 4-bit dequantisation,
    # adapter merging, and safetensors serialisation in one step.
    if _is_unsloth_model(model):
        model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        # Fallback for vanilla PEFT / transformers models.
        merged = _merge_peft_model(model)
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

    logger.info("HuggingFace export complete: %s", output_dir)


def _export_gguf(model, tokenizer, output_dir: Path, quant_method: str) -> None:
    """Merge LoRA adapters and save as a GGUF file via Unsloth.

    Args:
        model: An Unsloth model with LoRA adapters attached.
        tokenizer: The tokenizer associated with the model.
        output_dir: Destination directory for the exported GGUF file.
        quant_method: Quantisation method to apply (e.g. ``"q4_k_m"``).

    Raises:
        ValueError: If *quant_method* is not a supported GGUF quantisation
            method.
        RuntimeError: If *model* is not an Unsloth model.
    """
    if quant_method not in _GGUF_QUANT_METHODS:
        raise ValueError(
            f"Unknown GGUF quant method {quant_method!r}. Supported: {list(_GGUF_QUANT_METHODS)}"
        )

    logger.info(
        "Merging LoRA adapters and exporting to GGUF (%s) -> %s",
        quant_method,
        output_dir,
    )

    if not _is_unsloth_model(model):
        raise RuntimeError(
            "GGUF export requires an Unsloth model.  Load the model via "
            "unsloth.FastLanguageModel.from_pretrained() before exporting."
        )

    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quant_method,
    )

    logger.info("GGUF export complete: %s", output_dir)


def _is_unsloth_model(model) -> bool:
    """Return True if *model* was loaded through Unsloth.

    Args:
        model: The model instance to check.

    Returns:
        True if the model originates from Unsloth, False otherwise.
    """
    model_type = type(model).__module__ or ""
    # Unsloth patches the model class so its module lives under `unsloth.*`.
    if "unsloth" in model_type:
        return True
    # Also check for the Unsloth-specific save helpers directly.
    return hasattr(model, "save_pretrained_gguf") and hasattr(model, "save_pretrained_merged")


def _merge_peft_model(model):
    """Merge a PEFT LoRA model back into the base model.

    Args:
        model: A PEFT ``PeftModel`` or a plain transformers model.

    Returns:
        The base model with LoRA weights merged in. If *model* is not a
        ``PeftModel``, it is returned unchanged.
    """
    from peft import PeftModel

    if isinstance(model, PeftModel):
        logger.info("Merging PEFT LoRA adapters into base model")
        return model.merge_and_unload()
    # If it's already a plain transformers model, return as-is.
    return model


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------


def push_to_hub(
    output_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload model via model-tailor",
) -> str:
    """Push an exported model directory to the HuggingFace Hub.

    Args:
        output_dir: Local directory containing the model artefacts.
        repo_id: Hub repository id, e.g. ``"your-org/my-finetuned-model"``.
        token: HuggingFace API token. When None, the cached credential from
            ``huggingface-cli login`` is used.
        private: Whether to create the repo as private (default False).
        commit_message: Commit message for the upload.

    Returns:
        URL of the repository on the Hub.

    Raises:
        FileNotFoundError: If *output_dir* does not exist.
    """
    from huggingface_hub import HfApi

    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    api = HfApi(token=token)

    logger.info("Creating / updating Hub repo: %s", repo_id)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private, token=token)

    logger.info("Uploading %s -> %s", output_dir, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info("Upload complete: %s", repo_url)
    return repo_url
