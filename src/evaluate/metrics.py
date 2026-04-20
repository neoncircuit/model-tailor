"""Traditional evaluation metrics for SQL generation.

Provides exact-match, BLEU, ROUGE, and execution-accuracy metrics
plus a batch evaluator that aggregates results over a dataset.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Callable

import nltk
import sqlparse
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Ensure the NLTK punkt tokeniser data is available (downloaded lazily).
# The download check is slow, so we skip it if already downloaded.
# Call nltk.download('punkt_tab') before importing this module to avoid delay.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    pass  # Will fail on first use, which is acceptable


def _normalize_sql(text: str) -> str:
    """Normalise a SQL string for comparison.

    Lowercases, strips whitespace, collapses internal whitespace, removes
    trailing semicolons, and formats with sqlparse for consistent keyword
    casing / indentation.

    Args:
        text: Raw SQL string to normalise.

    Returns:
        Normalised SQL string suitable for exact comparison.
    """
    text = text.strip().lower()
    text = text.rstrip(";").strip()
    text = sqlparse.format(text, strip_comments=True, reindent=False, keyword_case="lower")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_text(text: str) -> str:
    """Lightweight text normalisation (lowercase + strip).

    Args:
        text: Raw text string to normalise.

    Returns:
        Lowercased string with collapsed whitespace.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def exact_match(pred: str, target: str) -> float:
    """Return 1.0 if pred and target are identical after normalisation, else 0.0.

    Both values are normalised as SQL: lowercased, whitespace-collapsed,
    trailing semicolons removed, and formatted via sqlparse.

    Args:
        pred: The predicted SQL string.
        target: The gold/reference SQL string.

    Returns:
        1.0 if the normalised strings match, 0.0 otherwise.
    """
    return 1.0 if _normalize_sql(pred) == _normalize_sql(target) else 0.0


def bleu_score(pred: str, target: str) -> float:
    """Compute sentence-level BLEU between pred and target.

    Tokenisation uses nltk.word_tokenize on the normalised text.
    Smoothing method 1 (add-epsilon) avoids zero scores for short
    sequences.

    Args:
        pred: The predicted text.
        target: The gold/reference text.

    Returns:
        BLEU score as a float in [0, 1].
    """
    pred_tokens = nltk.word_tokenize(_normalize_text(pred))
    target_tokens = nltk.word_tokenize(_normalize_text(target))

    if not target_tokens:
        return 1.0 if not pred_tokens else 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        [target_tokens],
        pred_tokens,
        smoothing_function=smoothing,
    )


def rouge_scores(pred: str, target: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L F-measure scores.

    Args:
        pred: The predicted text.
        target: The gold/reference text.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL``, each mapped
        to the corresponding F-measure (float in [0, 1]).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(_normalize_text(target), _normalize_text(pred))
    return {key: scores[key].fmeasure for key in ("rouge1", "rouge2", "rougeL")}


def sql_execution_accuracy(
    pred_sql: str,
    target_sql: str,
    db_path: str,
    timeout: float = 5.0,
) -> float:
    """Execute both queries against a SQLite database and compare result sets.

    Returns 1.0 if the (unordered) result sets match exactly, 0.0 otherwise.
    If either query fails to execute, returns 0.0.

    Args:
        pred_sql: The predicted SQL query.
        target_sql: The gold/reference SQL query.
        db_path: Path to the SQLite database file.
        timeout: Per-query execution time limit in seconds (default 5).

    Returns:
        1.0 if the unordered result sets match, 0.0 otherwise.
    """

    def _execute(sql: str, conn: sqlite3.Connection) -> set[tuple] | None:
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            return set(rows)
        except Exception as exc:
            logger.debug("SQL execution error: %s — query: %s", exc, sql[:200])
            return None

    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        # Limit how long a single statement can run.
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    except sqlite3.Error as exc:
        logger.warning("Could not connect to database %s: %s", db_path, exc)
        return 0.0

    try:
        pred_result = _execute(pred_sql, conn)
        target_result = _execute(target_sql, conn)
    finally:
        conn.close()

    if pred_result is None or target_result is None:
        return 0.0

    return 1.0 if pred_result == target_result else 0.0


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

# Registry of metric names → callables understood by evaluate_batch.
METRIC_REGISTRY: dict[str, Callable] = {
    "exact_match": exact_match,
    "bleu": bleu_score,
    "rouge": rouge_scores,
    "exec_accuracy": sql_execution_accuracy,
}


def evaluate_batch(
    predictions: list[str],
    targets: list[str],
    db_path: str | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a batch of predictions against targets.

    Args:
        predictions: Model-generated SQL strings.
        targets: Gold reference SQL strings.
        db_path: Path to a SQLite database. Required when ``"exec_accuracy"``
            is included in *metrics*.
        metrics: Which metrics to compute. Defaults to all non-execution
            metrics: ``["exact_match", "bleu", "rouge"]``. Pass
            ``"exec_accuracy"`` to include execution accuracy (requires
            *db_path*).

    Returns:
        Mapping of metric name to its mean score across the batch. ROUGE
        sub-scores are flattened: ``rouge1``, ``rouge2``, ``rougeL``.

    Raises:
        ValueError: If predictions and targets have different lengths, if
            ``"exec_accuracy"`` is requested without *db_path*, or if an
            unknown metric name is provided.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions ({len(predictions)}) and targets ({len(targets)}) "
            "must have the same length"
        )

    if metrics is None:
        metrics = ["exact_match", "bleu", "rouge"]

    if "exec_accuracy" in metrics and db_path is None:
        raise ValueError("db_path is required for exec_accuracy metric")

    n = len(predictions)
    if n == 0:
        return {}

    # Accumulators -- will be divided by n at the end.
    accum: dict[str, float] = {}

    for pred, target in zip(predictions, targets):
        for metric_name in metrics:
            if metric_name == "rouge":
                scores = rouge_scores(pred, target)
                for sub_key, value in scores.items():
                    accum[sub_key] = accum.get(sub_key, 0.0) + value
            elif metric_name == "exec_accuracy":
                score = sql_execution_accuracy(pred, target, db_path)  # type: ignore[arg-type]
                accum["exec_accuracy"] = accum.get("exec_accuracy", 0.0) + score
            else:
                fn = METRIC_REGISTRY.get(metric_name)
                if fn is None:
                    raise ValueError(f"Unknown metric: {metric_name!r}")
                score = fn(pred, target)
                accum[metric_name] = accum.get(metric_name, 0.0) + score

    return {key: value / n for key, value in accum.items()}
