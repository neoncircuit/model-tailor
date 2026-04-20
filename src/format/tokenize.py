"""Tokenization helpers for validating and adjusting dataset examples.

Provides utilities to:
- Compute token-length statistics across a dataset.
- Identify examples that exceed a maximum context length.
- Truncate or split oversized examples so they fit within the model's
  context window.

All functions accept a HuggingFace ``PreTrainedTokenizerBase`` (or any
object with an ``encode`` method returning a list of token ids).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Tokenizer protocol (so callers aren't forced to import transformers)
# ------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface a tokenizer must satisfy."""

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...
    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


# ------------------------------------------------------------------
# Token statistics
# ------------------------------------------------------------------


@dataclass
class TokenStats:
    """Analyse and store token-length distribution for a list of texts.

    Args:
        tokenizer: Any tokenizer with an ``encode`` method.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> stats = TokenStats(tok)
        >>> stats.compute(["SELECT 1", "SELECT * FROM users WHERE age > 30"])
        >>> stats.mean
        5.5
    """

    tokenizer: Tokenizer
    lengths: list[int] = field(default_factory=list, repr=False)
    total: int = 0
    min_len: int = 0
    max_len: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def compute(self, texts: list[str]) -> TokenStats:
        """Tokenize every text and compute summary statistics.

        Args:
            texts: Raw strings to tokenize.

        Returns:
            This TokenStats instance with all statistics populated.
        """
        self.lengths = [len(self.tokenizer.encode(t)) for t in texts]
        n = len(self.lengths)
        if n == 0:
            return self

        sorted_lens = sorted(self.lengths)
        self.total = n
        self.min_len = sorted_lens[0]
        self.max_len = sorted_lens[-1]
        self.mean = sum(sorted_lens) / n
        self.median = self._percentile(sorted_lens, 50)
        self.p90 = self._percentile(sorted_lens, 90)
        self.p95 = self._percentile(sorted_lens, 95)
        self.p99 = self._percentile(sorted_lens, 99)
        return self

    @staticmethod
    def _percentile(sorted_values: list[int], pct: float) -> float:
        """Compute a percentile from an already-sorted list."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        k = (pct / 100) * (n - 1)
        lo = int(math.floor(k))
        hi = min(lo + 1, n - 1)
        weight = k - lo
        return sorted_values[lo] + weight * (sorted_values[hi] - sorted_values[lo])

    def summary(self) -> dict[str, Any]:
        """Return statistics as a plain dict.

        Returns:
            A dict with keys "total", "min", "max", "mean", "median",
            "p90", "p95", and "p99", all rounded to one decimal place
            where applicable.
        """
        return {
            "total": self.total,
            "min": self.min_len,
            "max": self.max_len,
            "mean": round(self.mean, 1),
            "median": round(self.median, 1),
            "p90": round(self.p90, 1),
            "p95": round(self.p95, 1),
            "p99": round(self.p99, 1),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"TokenStats(n={s['total']}, min={s['min']}, max={s['max']}, "
            f"mean={s['mean']}, median={s['median']}, "
            f"p90={s['p90']}, p95={s['p95']}, p99={s['p99']})"
        )


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of ``validate_max_length``."""

    valid: list[dict[str, Any]]
    """Examples whose tokenized text fits within ``max_len``."""

    oversize: list[dict[str, Any]]
    """Examples that exceed ``max_len``."""

    valid_indices: list[int]
    """Original indices of valid examples."""

    oversize_indices: list[int]
    """Original indices of oversize examples."""

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_oversize(self) -> int:
        return len(self.oversize)

    @property
    def pct_oversize(self) -> float:
        total = self.n_valid + self.n_oversize
        return (self.n_oversize / total * 100) if total > 0 else 0.0


def _extract_text(example: dict[str, Any]) -> str:
    """Pull a single string from an example dict.

    Supports multiple common column layouts:
    - ``text`` column (pre-formatted string)
    - ``messages`` column (list of role/content dicts -> concatenated)
    - ``conversations`` column (ShareGPT format -> concatenated values)
    - ``prompt`` + ``completion`` columns
    - ``nl`` + ``sql`` columns (raw pair)
    """
    if "text" in example:
        return example["text"]

    if "messages" in example:
        return "\n".join(m["content"] for m in example["messages"])

    if "conversations" in example:
        return "\n".join(t["value"] for t in example["conversations"])

    if "prompt" in example and "completion" in example:
        return example["prompt"] + "\n" + example["completion"]

    if "nl" in example and "sql" in example:
        return example["nl"] + "\n" + example["sql"]

    raise ValueError(
        f"Cannot extract text from example with keys {list(example.keys())}. "
        "Expected one of: text, messages, conversations, prompt+completion, nl+sql."
    )


def validate_max_length(
    examples: list[dict[str, Any]],
    tokenizer: Tokenizer,
    max_len: int,
) -> ValidationResult:
    """Partition examples into those that fit within ``max_len`` and those that don't.

    Args:
        examples: Dataset examples in any supported format
            (see ``_extract_text``).
        tokenizer: HuggingFace tokenizer (or compatible).
        max_len: Maximum number of tokens allowed.

    Returns:
        A ValidationResult containing the valid and oversize examples
        along with their original indices.
    """
    valid: list[dict[str, Any]] = []
    oversize: list[dict[str, Any]] = []
    valid_idx: list[int] = []
    oversize_idx: list[int] = []

    for i, ex in enumerate(examples):
        text = _extract_text(ex)
        n_tokens = len(tokenizer.encode(text))
        if n_tokens <= max_len:
            valid.append(ex)
            valid_idx.append(i)
        else:
            oversize.append(ex)
            oversize_idx.append(i)

    if oversize:
        logger.info(
            "validate_max_length: %d / %d examples exceed max_len=%d (%.1f%%)",
            len(oversize),
            len(examples),
            max_len,
            len(oversize) / len(examples) * 100,
        )

    return ValidationResult(
        valid=valid,
        oversize=oversize,
        valid_indices=valid_idx,
        oversize_indices=oversize_idx,
    )


# ------------------------------------------------------------------
# Truncation / splitting
# ------------------------------------------------------------------


def truncate_or_split(
    examples: list[dict[str, Any]],
    tokenizer: Tokenizer,
    max_len: int,
    *,
    strategy: str = "truncate",
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """Handle examples that exceed ``max_len``.

    Args:
        examples: Dataset examples in any supported format.
        tokenizer: HuggingFace tokenizer (or compatible).
        max_len: Maximum number of tokens per example.
        strategy: How to handle oversize examples. "truncate" trims
            token ids and decodes back to text. "split" breaks into
            consecutive chunks of ``max_len`` tokens. "drop" discards
            oversize examples entirely.
        overlap: Number of overlapping tokens between consecutive
            chunks when ``strategy="split"``. Ignored for other
            strategies.

    Returns:
        Processed examples, each guaranteed to be at most ``max_len``
        tokens.

    Raises:
        ValueError: If strategy is not "truncate", "split", or "drop".
    """
    if strategy not in ("truncate", "split", "drop"):
        raise ValueError(f"Unknown strategy {strategy!r}. Use 'truncate', 'split', or 'drop'.")

    output: list[dict[str, Any]] = []

    for ex in examples:
        text = _extract_text(ex)
        token_ids = tokenizer.encode(text)

        if len(token_ids) <= max_len:
            output.append(ex)
            continue

        if strategy == "drop":
            continue

        if strategy == "truncate":
            truncated_ids = token_ids[:max_len]
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            output.append(_replace_text(ex, truncated_text))

        elif strategy == "split":
            step = max(1, max_len - overlap)
            for start in range(0, len(token_ids), step):
                chunk_ids = token_ids[start : start + max_len]
                if not chunk_ids:
                    break
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                output.append(_replace_text(ex, chunk_text))

    n_orig = len(examples)
    n_out = len(output)
    if n_out != n_orig:
        logger.info(
            "truncate_or_split(%s): %d examples -> %d examples (max_len=%d)",
            strategy,
            n_orig,
            n_out,
            max_len,
        )

    return output


def _replace_text(example: dict[str, Any], new_text: str) -> dict[str, Any]:
    """Create a copy of ``example`` with the text content replaced.

    Preserves any metadata keys (difficulty, id, etc.) while updating
    the primary text field.
    """
    new_ex = dict(example)

    if "text" in example:
        new_ex["text"] = new_text
    elif "messages" in example:
        # Collapse to a single-turn format since we've re-tokenized
        new_ex["messages"] = [{"role": "user", "content": new_text}]
    elif "conversations" in example:
        new_ex["conversations"] = [{"from": "human", "value": new_text}]
    elif "prompt" in example and "completion" in example:
        # Put everything into prompt, clear completion
        new_ex["prompt"] = new_text
        new_ex["completion"] = ""
    elif "nl" in example and "sql" in example:
        new_ex["text"] = new_text
        # Remove the original nl/sql keys since the text is now combined
        new_ex.pop("nl", None)
        new_ex.pop("sql", None)
    else:
        new_ex["text"] = new_text

    return new_ex
