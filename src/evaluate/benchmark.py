"""Automated benchmark runner for fine-tuned text-to-SQL models.

Runs a model through a test set, collects predictions, computes
traditional metrics, and optionally invokes the LLM judge.  Provides
comparison and reporting utilities.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.evaluate.judge import JudgeExample, JudgeResult, LLMJudge
from src.evaluate.metrics import evaluate_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Each test record follows the project convention:
# {"nl": "...", "sql": "...", "difficulty": "easy", "category": "select"}
TestRecord = dict[str, Any]


@dataclass
class BenchmarkResults:
    """Container for a complete benchmark run."""

    model_name: str
    metric_scores: dict[str, float] = field(default_factory=dict)
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    difficulty_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    judge_scores: dict[str, float] = field(default_factory=dict)
    predictions: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    num_examples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark results to a plain dictionary.

        Returns:
            Dictionary containing model name, example count, timing,
            metric scores, category/difficulty breakdowns, and judge scores.
        """
        return {
            "model_name": self.model_name,
            "num_examples": self.num_examples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "metrics": self.metric_scores,
            "by_category": self.category_scores,
            "by_difficulty": self.difficulty_scores,
            "judge": self.judge_scores,
        }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _build_prompt(nl: str) -> str:
    """Build a simple instruction prompt for text-to-SQL generation.

    Args:
        nl: Natural language question to embed in the prompt.

    Returns:
        Formatted instruction prompt string ending with ``SQL:``.
    """
    return (
        "Convert the following natural language question into a SQL query.\n"
        "Output ONLY the SQL query, with no explanation or markdown formatting.\n\n"
        f"Question: {nl}\n\n"
        "SQL:"
    )


def _generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Run batched inference and return decoded predictions.

    Args:
        model: The fine-tuned model.
        tokenizer: Matching tokenizer (must have a pad token set).
        prompts: Formatted input prompts.
        batch_size: Number of prompts per forward pass.
        max_new_tokens: Maximum tokens to generate for each prediction.

    Returns:
        List of decoded model outputs with the prompt portion stripped.
    """
    # Make sure we have a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    predictions: list[str] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Strip the prompt tokens from each generated sequence.
        prompt_lengths = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_lengths:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean up: take only the first line / first SQL statement.
        for text in decoded:
            sql = text.strip().split("\n")[0].strip()
            predictions.append(sql)

    return predictions


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Run a fine-tuned model through a test set and compute metrics.

    Args:
        batch_size: Inference batch size (default 8).
        max_new_tokens: Maximum tokens to generate per example (default 256).
        metrics: Which traditional metrics to compute. Defaults to
            ``["exact_match", "bleu", "rouge"]``.
        db_path: Path to SQLite database for execution accuracy.
        use_judge: Whether to also run LLM-as-judge evaluation
            (default False).
        judge_concurrency: Parallelism for LLM judge calls (default 4).
    """

    def __init__(
        self,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        metrics: list[str] | None = None,
        db_path: str | None = None,
        use_judge: bool = False,
        judge_concurrency: int = 4,
    ) -> None:
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.metrics = metrics or ["exact_match", "bleu", "rouge"]
        self.db_path = db_path
        self.use_judge = use_judge
        self.judge_concurrency = judge_concurrency

        if "exec_accuracy" in self.metrics and self.db_path is None:
            raise ValueError("db_path is required when exec_accuracy is in metrics")

    # -- main entry point ---------------------------------------------------

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        test_data: list[TestRecord],
        metrics: list[str] | None = None,
    ) -> BenchmarkResults:
        """Generate predictions for *test_data* and compute all metrics.

        Args:
            model: The fine-tuned model to evaluate.
            tokenizer: Matching tokenizer.
            test_data: Each record must have at least ``"nl"`` and ``"sql"``
                keys. Optional ``"difficulty"`` and ``"category"`` enable
                breakdowns.
            metrics: Override instance-level metrics for this run.

        Returns:
            Full BenchmarkResults including per-metric, per-category, and
            per-difficulty breakdowns.
        """
        active_metrics = metrics or self.metrics
        model_name = getattr(model, "name_or_path", None) or model.__class__.__name__

        logger.info(
            "Starting benchmark: %d examples, batch_size=%d, metrics=%s",
            len(test_data),
            self.batch_size,
            active_metrics,
        )

        # --- Generate predictions ---
        prompts = [_build_prompt(rec["nl"]) for rec in test_data]
        targets = [rec["sql"] for rec in test_data]

        t0 = time.perf_counter()
        predictions = _generate_predictions(
            model, tokenizer, prompts, self.batch_size, self.max_new_tokens
        )
        elapsed = time.perf_counter() - t0

        # --- Aggregate metrics ---
        overall_scores = evaluate_batch(
            predictions, targets, db_path=self.db_path, metrics=active_metrics
        )

        # --- Per-category breakdown ---
        category_scores = self._compute_group_scores(
            test_data, predictions, targets, group_key="category", metrics=active_metrics
        )

        # --- Per-difficulty breakdown ---
        difficulty_scores = self._compute_group_scores(
            test_data, predictions, targets, group_key="difficulty", metrics=active_metrics
        )

        # --- LLM judge (optional) ---
        judge_summary: dict[str, float] = {}
        if self.use_judge:
            judge_summary = self._run_judge(test_data, predictions, targets)

        results = BenchmarkResults(
            model_name=model_name,
            metric_scores=overall_scores,
            category_scores=category_scores,
            difficulty_scores=difficulty_scores,
            judge_scores=judge_summary,
            predictions=predictions,
            targets=targets,
            elapsed_seconds=elapsed,
            num_examples=len(test_data),
        )

        logger.info("Benchmark complete in %.1fs: %s", elapsed, overall_scores)
        return results

    # -- grouped scoring ----------------------------------------------------

    def _compute_group_scores(
        self,
        test_data: list[TestRecord],
        predictions: list[str],
        targets: list[str],
        group_key: str,
        metrics: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute metrics broken down by a grouping field (e.g. category).

        Args:
            test_data: Test records containing the grouping field.
            predictions: Model-generated SQL strings.
            targets: Gold/reference SQL strings.
            group_key: Record field to group by (e.g. ``"category"`` or
                ``"difficulty"``).
            metrics: Which metrics to compute for each group.

        Returns:
            Mapping of group name to a dict of metric name to mean score.
        """
        groups: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))

        for rec, pred, tgt in zip(test_data, predictions, targets):
            group = rec.get(group_key, "unknown")
            groups[group][0].append(pred)
            groups[group][1].append(tgt)

        result: dict[str, dict[str, float]] = {}
        for group_name, (preds, tgts) in sorted(groups.items()):
            result[group_name] = evaluate_batch(preds, tgts, db_path=self.db_path, metrics=metrics)

        return result

    # -- LLM judge integration ---------------------------------------------

    def _run_judge(
        self,
        test_data: list[TestRecord],
        predictions: list[str],
        targets: list[str],
    ) -> dict[str, Any]:
        """Run the LLM judge and return summary statistics.

        Args:
            test_data: Test records with ``"nl"`` keys.
            predictions: Model-generated SQL strings.
            targets: Gold/reference SQL strings.

        Returns:
            Summary dict with ``mean_score``, ``score_distribution``,
            ``num_judged``, and ``num_errors``.
        """
        judge = LLMJudge()
        examples: list[JudgeExample] = [
            JudgeExample(nl=rec["nl"], pred_sql=pred, target_sql=tgt)
            for rec, pred, tgt in zip(test_data, predictions, targets)
        ]

        logger.info(
            "Running LLM judge on %d examples (concurrency=%d)",
            len(examples),
            self.judge_concurrency,
        )
        results: list[JudgeResult] = judge.judge_batch(examples, concurrency=self.judge_concurrency)

        mean = judge.mean_score(results)
        # Distribution of scores.
        dist: dict[int, int] = defaultdict(int)
        for r in results:
            if r.score > 0:
                dist[r.score] += 1
        valid_count = sum(dist.values())

        return {
            "mean_score": round(mean, 3),
            "score_distribution": {str(k): dist.get(k, 0) for k in range(1, 6)},
            "num_judged": valid_count,
            "num_errors": len(results) - valid_count,
        }

    # -- comparison ---------------------------------------------------------

    @staticmethod
    def compare(
        results_a: BenchmarkResults,
        results_b: BenchmarkResults,
    ) -> dict[str, Any]:
        """Compare two benchmark runs side by side.

        Args:
            results_a: First completed benchmark run.
            results_b: Second completed benchmark run.

        Returns:
            Comparison dict with per-metric deltas, optional judge
            comparison, and a summary indicating which model performed
            better overall.
        """
        all_metrics = set(results_a.metric_scores) | set(results_b.metric_scores)

        comparison: dict[str, Any] = {
            "model_a": results_a.model_name,
            "model_b": results_b.model_name,
            "num_examples_a": results_a.num_examples,
            "num_examples_b": results_b.num_examples,
            "metrics": {},
        }

        for metric in sorted(all_metrics):
            score_a = results_a.metric_scores.get(metric, 0.0)
            score_b = results_b.metric_scores.get(metric, 0.0)
            delta = score_b - score_a
            comparison["metrics"][metric] = {
                "model_a": round(score_a, 4),
                "model_b": round(score_b, 4),
                "delta": round(delta, 4),
                "improved": delta > 0,
            }

        # Judge comparison (if available).
        if results_a.judge_scores and results_b.judge_scores:
            judge_a = results_a.judge_scores.get("mean_score", 0.0)
            judge_b = results_b.judge_scores.get("mean_score", 0.0)
            comparison["judge"] = {
                "model_a": judge_a,
                "model_b": judge_b,
                "delta": round(judge_b - judge_a, 3),
                "improved": judge_b > judge_a,
            }

        # Overall winner heuristic: count how many metrics improved.
        wins_b = sum(1 for m in comparison["metrics"].values() if m["improved"])
        wins_a = sum(
            1 for m in comparison["metrics"].values() if not m["improved"] and m["delta"] < 0
        )
        if wins_b > wins_a:
            comparison["summary"] = (
                f"{results_b.model_name} is better on {wins_b}/{len(all_metrics)} metrics"
            )
        elif wins_a > wins_b:
            comparison["summary"] = (
                f"{results_a.model_name} is better on {wins_a}/{len(all_metrics)} metrics"
            )
        else:
            comparison["summary"] = "Models are tied across metrics"

        return comparison

    # -- reporting ----------------------------------------------------------

    @staticmethod
    def report(results: BenchmarkResults) -> dict[str, Any]:
        """Generate a summary report from benchmark results.

        Args:
            results: A completed benchmark run.

        Returns:
            Structured report dict with overall scores, per-category and
            per-difficulty breakdowns, and timing information.
        """
        report_dict: dict[str, Any] = {
            "model": results.model_name,
            "num_examples": results.num_examples,
            "elapsed_seconds": round(results.elapsed_seconds, 2),
            "throughput_examples_per_sec": (
                round(results.num_examples / results.elapsed_seconds, 2)
                if results.elapsed_seconds > 0
                else 0.0
            ),
            "overall_metrics": {k: round(v, 4) for k, v in results.metric_scores.items()},
        }

        if results.category_scores:
            report_dict["by_category"] = {
                cat: {k: round(v, 4) for k, v in scores.items()}
                for cat, scores in results.category_scores.items()
            }

        if results.difficulty_scores:
            report_dict["by_difficulty"] = {
                diff: {k: round(v, 4) for k, v in scores.items()}
                for diff, scores in results.difficulty_scores.items()
            }

        if results.judge_scores:
            report_dict["llm_judge"] = results.judge_scores

        return report_dict
