"""Evaluation module for fine-tuned text-to-SQL models.

Provides traditional metrics (exact match, BLEU, ROUGE, execution accuracy),
LLM-as-judge evaluation, and an automated benchmark runner.
"""

from src.evaluate.benchmark import BenchmarkResults, BenchmarkRunner
from src.evaluate.judge import JudgeExample, JudgeResult, LLMJudge
from src.evaluate.metrics import (
    bleu_score,
    evaluate_batch,
    exact_match,
    rouge_scores,
    sql_execution_accuracy,
)

__all__ = [
    # metrics
    "exact_match",
    "bleu_score",
    "rouge_scores",
    "sql_execution_accuracy",
    "evaluate_batch",
    # judge
    "LLMJudge",
    "JudgeResult",
    "JudgeExample",
    # benchmark
    "BenchmarkRunner",
    "BenchmarkResults",
]
