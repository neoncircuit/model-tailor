"""Evaluation module for fine-tuned text-to-SQL models.

Provides traditional metrics (exact match, BLEU, ROUGE, execution accuracy),
LLM-as-judge evaluation, an automated benchmark runner, and an execution
 correctness gate.

Note:
    ``BenchmarkRunner`` and ``LLMJudge`` are intentionally *not* imported at
    package level because they depend on heavy third-party libraries
    (``transformers``, ``torch``, teacher API clients). Import them directly
    from their submodules when needed::

        from src.evaluate.benchmark import BenchmarkRunner
        from src.evaluate.judge import LLMJudge
"""

from src.evaluate.gate import ExecutionGate, ExecutionResult, GateMetrics
from src.evaluate.metrics import (
    bleu_score,
    evaluate_batch,
    exact_match,
    rouge_scores,
    sql_execution_accuracy,
)

__all__ = [
    # gate
    "ExecutionGate",
    "ExecutionResult",
    "GateMetrics",
    # metrics
    "exact_match",
    "bleu_score",
    "rouge_scores",
    "sql_execution_accuracy",
    "evaluate_batch",
]
