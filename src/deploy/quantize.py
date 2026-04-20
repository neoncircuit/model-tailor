"""Post-export quantisation utilities for GGUF models.

Wraps ``llama.cpp``'s ``llama-quantize`` CLI to produce smaller model files and
provides helpers for estimating sizes and comparing output quality across
quantisation levels.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Supported quantisation types and their approximate bits-per-weight.
QuantType = Literal["q4_k_m", "q5_k_m", "q8_0", "f16"]

QUANT_BITS: dict[str, float] = {
    "q4_k_m": 4.83,
    "q5_k_m": 5.69,
    "q8_0": 8.50,
    "f16": 16.0,
}


@dataclass
class QuantizationConfig:
    """Settings for GGUF quantisation.

    Attributes:
        method: Quantisation method to apply (default ``"q4_k_m"``).
        bits: Override for the effective bits-per-weight. When None the value
            is looked up from ``QUANT_BITS``.
        llama_quantize_path: Path to the ``llama-quantize`` binary. When left
            empty the binary is located via ``$PATH``.
    """

    method: QuantType = "q4_k_m"
    bits: float | None = None
    llama_quantize_path: str = ""

    @property
    def effective_bits(self) -> float:
        """Return the bits-per-weight for this configuration."""
        if self.bits is not None:
            return self.bits
        return QUANT_BITS[self.method]


# ---------------------------------------------------------------------------
# Core quantisation
# ---------------------------------------------------------------------------


def _find_quantize_binary(config: QuantizationConfig | None = None) -> str:
    """Locate the ``llama-quantize`` binary.

    Checks, in order: the explicit path in *config*, then
    ``llama-quantize`` on ``$PATH``, then ``quantize`` on ``$PATH``.

    Args:
        config: Optional configuration containing an explicit binary path.

    Returns:
        Absolute path to the ``llama-quantize`` binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
    """
    if config and config.llama_quantize_path:
        path = Path(config.llama_quantize_path)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"llama-quantize not found at {path}")

    for name in ("llama-quantize", "quantize"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find the llama-quantize binary.  Either install llama.cpp "
        "and ensure the binary is on $PATH, or set "
        "QuantizationConfig.llama_quantize_path explicitly."
    )


def quantize_gguf(
    input_path: str | Path,
    output_path: str | Path,
    quant_type: QuantType = "q4_k_m",
    *,
    config: QuantizationConfig | None = None,
) -> Path:
    """Quantise a GGUF model file using ``llama-quantize``.

    Args:
        input_path: Path to the source (typically f16) GGUF file.
        output_path: Destination path for the quantised GGUF file.
        quant_type: One of ``"q4_k_m"``, ``"q5_k_m"``, ``"q8_0"``,
            ``"f16"``.
        config: Optional configuration (used to locate the
            ``llama-quantize`` binary).

    Returns:
        Resolved path to the quantised model file.

    Raises:
        FileNotFoundError: If the input GGUF or the ``llama-quantize``
            binary cannot be found.
        subprocess.CalledProcessError: If quantisation fails.
        ValueError: If *quant_type* is not a supported quantisation method.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input GGUF not found: {input_path}")

    if quant_type not in QUANT_BITS:
        raise ValueError(f"Unsupported quant_type {quant_type!r}.  Choose from: {list(QUANT_BITS)}")

    # f16 is a no-op -- just copy.
    if quant_type == "f16":
        logger.info("f16 requested -- copying %s -> %s (no quantisation)", input_path, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return output_path

    binary = _find_quantize_binary(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [binary, str(input_path), str(output_path), quant_type.upper()]
    logger.info("Running: %s", " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("llama-quantize stderr:\n%s", result.stderr)
        result.check_returncode()  # raises CalledProcessError

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Quantisation complete in %.1fs: %s (%.1f MB)",
        elapsed,
        output_path.name,
        size_mb,
    )
    return output_path


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------


def estimate_model_size(param_count: int, quant_bits: float) -> dict[str, float]:
    """Estimate model file size after quantisation.

    Args:
        param_count: Total number of model parameters (e.g. 7_000_000_000
            for a 7B model).
        quant_bits: Effective bits per weight. Use ``QUANT_BITS`` or
            ``QuantizationConfig.effective_bits``.

    Returns:
        A dict with keys ``"size_gb"``, ``"size_mb"``, and
        ``"bits_per_weight"``.

    Raises:
        ValueError: If *param_count* or *quant_bits* is not positive.
    """
    if param_count <= 0:
        raise ValueError(f"param_count must be positive, got {param_count}")
    if quant_bits <= 0:
        raise ValueError(f"quant_bits must be positive, got {quant_bits}")

    size_bytes = param_count * quant_bits / 8
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    return {
        "size_gb": round(size_gb, 2),
        "size_mb": round(size_mb, 1),
        "bits_per_weight": quant_bits,
    }


def estimate_all_quant_sizes(param_count: int) -> dict[str, dict[str, float]]:
    """Return estimated sizes for every supported quantisation type.

    Args:
        param_count: Total number of model parameters.

    Returns:
        Mapping of quant type name to its size estimate dict as returned
        by ``estimate_model_size``.
    """
    return {qt: estimate_model_size(param_count, bits) for qt, bits in QUANT_BITS.items()}


# ---------------------------------------------------------------------------
# Quality comparison
# ---------------------------------------------------------------------------


@dataclass
class _QuantResult:
    """Internal container for one quantised model's evaluation."""

    quant_type: str
    size_mb: float
    outputs: list[str] = field(default_factory=list)
    tokens_per_second: float = 0.0


def compare_quantizations(
    model_path: str | Path,
    test_prompts: list[str],
    *,
    quant_types: list[QuantType] | None = None,
    n_predict: int = 128,
) -> list[dict]:
    """Generate responses from several quantised variants and return results for comparison.

    This function loads each quantised GGUF via ``llama-cpp-python``, runs
    the provided *test_prompts*, and records the outputs along with basic
    throughput numbers so the caller can inspect quality / speed trade-offs.

    Args:
        model_path: Directory containing GGUF files or a single GGUF file
            path. When a directory is given, the function looks for files
            matching ``*<quant_type>*`` patterns.
        test_prompts: Prompts to generate completions for.
        quant_types: Quantisation levels to compare. Defaults to all
            supported types.
        n_predict: Maximum tokens to generate per prompt.

    Returns:
        A list of dicts, one per quant type, each with keys
        ``quant_type``, ``size_mb``, ``outputs``, and
        ``tokens_per_second``.

    Raises:
        FileNotFoundError: If no GGUF files are found for the requested
            quant types.
    """
    from llama_cpp import Llama

    if quant_types is None:
        quant_types = list(QUANT_BITS.keys())

    model_path = Path(model_path).resolve()
    gguf_files = _resolve_gguf_files(model_path, quant_types)

    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF files found for quant types {quant_types} in {model_path}"
        )

    results: list[dict] = []

    for qt, gguf_path in gguf_files.items():
        logger.info("Evaluating quant type %s from %s", qt, gguf_path.name)
        size_mb = gguf_path.stat().st_size / (1024 * 1024)

        llm = Llama(model_path=str(gguf_path), n_ctx=2048, verbose=False)
        outputs: list[str] = []
        total_tokens = 0
        total_time = 0.0

        for prompt in test_prompts:
            t0 = time.perf_counter()
            response = llm(prompt, max_tokens=n_predict, echo=False)
            elapsed = time.perf_counter() - t0

            text = response["choices"][0]["text"]
            n_tokens = response["usage"]["completion_tokens"]
            outputs.append(text)
            total_tokens += n_tokens
            total_time += elapsed

        tps = total_tokens / total_time if total_time > 0 else 0.0

        results.append(
            {
                "quant_type": qt,
                "size_mb": round(size_mb, 1),
                "outputs": outputs,
                "tokens_per_second": round(tps, 1),
            }
        )

        # Free the model to reclaim memory before loading the next one.
        del llm

    return results


def _resolve_gguf_files(
    model_path: Path,
    quant_types: list[str],
) -> dict[str, Path]:
    """Map each requested quant type to a GGUF file on disk.

    Args:
        model_path: Path to a single GGUF file or a directory containing
            GGUF files.
        quant_types: List of quantisation type names to search for.

    Returns:
        Mapping of quant type name to the resolved GGUF file path.
    """
    if model_path.is_file():
        # Single file -- detect its quant type from the filename.
        name_lower = model_path.name.lower()
        matched: dict[str, Path] = {}
        for qt in quant_types:
            if qt.lower().replace("_", "") in name_lower.replace("_", "").replace("-", ""):
                matched[qt] = model_path
                break
        if not matched:
            # Can't determine quant type -- just use the first requested type.
            matched[quant_types[0]] = model_path
        return matched

    # Directory -- search for GGUF files.
    found: dict[str, Path] = {}
    for qt in quant_types:
        # Common naming patterns: *q4_k_m.gguf, *Q4_K_M.gguf, etc.
        for pattern in (f"*{qt}*.gguf", f"*{qt.upper()}*.gguf"):
            matches = sorted(model_path.glob(pattern))
            if matches:
                found[qt] = matches[0]
                break
        # Also try with hyphens instead of underscores.
        if qt not in found:
            alt = qt.replace("_", "-")
            for pattern in (f"*{alt}*.gguf", f"*{alt.upper()}*.gguf"):
                matches = sorted(model_path.glob(pattern))
                if matches:
                    found[qt] = matches[0]
                    break

    return found
