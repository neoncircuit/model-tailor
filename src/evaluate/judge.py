"""LLM-as-judge evaluation for SQL generation.

Uses a teacher model (GPT-4, Claude, etc.) to evaluate predicted SQL
against a reference, producing a structured score (1-5) with reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypedDict

from src.llm.client import Message, TeacherClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class JudgeExample(TypedDict):
    """A single example to be judged."""

    nl: str
    pred_sql: str
    target_sql: str


@dataclass
class JudgeResult:
    """Structured output from the LLM judge."""

    score: int  # 1-5 scale
    correctness: str
    efficiency: str
    readability: str
    edge_cases: str
    overall: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "score": self.score,
            "correctness": self.correctness,
            "efficiency": self.efficiency,
            "readability": self.readability,
            "edge_cases": self.edge_cases,
            "overall": self.overall,
        }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert SQL evaluator. You will be given a natural language question, \
a reference (gold) SQL query, and a predicted SQL query. Your job is to evaluate \
the predicted query on the following criteria:

1. **Correctness** - Does the predicted SQL produce the same results as the reference \
for all possible database states? Consider edge cases like NULLs, empty tables, and \
boundary values.
2. **Efficiency** - Is the predicted SQL reasonably efficient? Does it avoid \
unnecessary subqueries, redundant joins, or full table scans where an index could help?
3. **Readability** - Is the predicted SQL well-structured and easy to understand? \
Proper use of aliases, formatting, and standard SQL conventions.
4. **Edge Case Handling** - Does the predicted SQL handle edge cases correctly? \
Consider NULL handling, empty results, data type mismatches, and boundary conditions.

Respond with ONLY a JSON object in this exact format (no markdown fences):
{
  "score": <integer 1-5>,
  "correctness": "<brief assessment>",
  "efficiency": "<brief assessment>",
  "readability": "<brief assessment>",
  "edge_cases": "<brief assessment>",
  "overall": "<1-2 sentence summary>"
}

Scoring guide:
- 5: Perfect or near-perfect match; any differences are cosmetic
- 4: Correct results in almost all cases; minor style issues
- 3: Mostly correct but may fail on some edge cases or has notable style issues
- 2: Partially correct; significant logical errors or missing clauses
- 1: Incorrect; produces wrong results or fails to execute\
"""

_USER_TEMPLATE = """\
**Natural language question:**
{nl_query}

**Reference SQL:**
{target_sql}

**Predicted SQL:**
{pred_sql}\
"""


# ---------------------------------------------------------------------------
# LLMJudge class
# ---------------------------------------------------------------------------


class LLMJudge:
    """Use a teacher model to evaluate generated SQL queries.

    Args:
        client: Pre-configured teacher client. If None, a new client is
            created from ``config/base.yaml``.
        temperature: Sampling temperature for the judge (low for consistency).
        max_tokens: Maximum tokens in the judge response.
        max_retries: Number of times to retry if output parsing fails.
    """

    def __init__(
        self,
        client: TeacherClient | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> None:
        self.client = client or TeacherClient()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # -- output parsing -----------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> JudgeResult:
        """Parse the LLM response into a JudgeResult.

        Handles minor formatting issues like markdown code fences or
        trailing commas that models sometimes produce.

        Args:
            text: Raw text response from the LLM judge.

        Returns:
            Parsed JudgeResult with score and per-criterion reasoning.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON or
                if the score is outside the 1-5 range.
        """
        # Strip optional markdown fences.
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Remove trailing commas before closing braces (invalid JSON but common).
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse judge response as JSON: {exc}\n{text}") from exc

        score = int(data.get("score", 0))
        if not 1 <= score <= 5:
            raise ValueError(f"Score must be 1-5, got {score}")

        return JudgeResult(
            score=score,
            correctness=str(data.get("correctness", "")),
            efficiency=str(data.get("efficiency", "")),
            readability=str(data.get("readability", "")),
            edge_cases=str(data.get("edge_cases", "")),
            overall=str(data.get("overall", "")),
        )

    # -- single evaluation --------------------------------------------------

    def judge_single(
        self,
        nl_query: str,
        pred_sql: str,
        target_sql: str,
    ) -> JudgeResult:
        """Score a single predicted SQL query against a reference.

        Args:
            nl_query: The natural language question that prompted the SQL.
            pred_sql: The model-generated SQL query.
            target_sql: The gold/reference SQL query.

        Returns:
            Structured JudgeResult with score (1-5) and per-criterion
            reasoning.

        Raises:
            ValueError: If a valid judge response cannot be obtained after
                all retry attempts.
        """
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(
                    nl_query=nl_query,
                    target_sql=target_sql,
                    pred_sql=pred_sql,
                ),
            ),
        ]

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.complete(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return self._parse_response(response)
            except (ValueError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "Judge parse attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        raise ValueError(
            f"Failed to get valid judge response after {self.max_retries} attempts: {last_error}"
        )

    # -- batch evaluation ---------------------------------------------------

    def judge_batch(
        self,
        examples: list[JudgeExample],
        concurrency: int = 4,
    ) -> list[JudgeResult]:
        """Judge multiple examples, optionally in parallel.

        Args:
            examples: Each dict must contain ``nl``, ``pred_sql``, and
                ``target_sql``.
            concurrency: Number of parallel API calls. Set to 1 for
                sequential execution (useful for rate-limited APIs).

        Returns:
            List of JudgeResult objects in the same order as *examples*.
            Failed evaluations are represented by a sentinel result with
            score 0.
        """
        if concurrency <= 1:
            results: list[JudgeResult] = []
            for i, ex in enumerate(examples):
                logger.info("Judging example %d/%d", i + 1, len(examples))
                result = self.judge_single(
                    nl_query=ex["nl"],
                    pred_sql=ex["pred_sql"],
                    target_sql=ex["target_sql"],
                )
                results.append(result)
            return results

        # Parallel execution with a thread pool.
        results_map: dict[int, JudgeResult] = {}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    self.judge_single,
                    nl_query=ex["nl"],
                    pred_sql=ex["pred_sql"],
                    target_sql=ex["target_sql"],
                ): idx
                for idx, ex in enumerate(examples)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_map[idx] = future.result()
                except Exception as exc:
                    logger.error("Judge failed for example %d: %s", idx, exc)
                    # Return a zero-score sentinel so the batch always
                    # returns the same number of results as inputs.
                    results_map[idx] = JudgeResult(
                        score=0,
                        correctness="Error during evaluation",
                        efficiency="",
                        readability="",
                        edge_cases="",
                        overall=f"Evaluation failed: {exc}",
                    )

        return [results_map[i] for i in range(len(examples))]

    # -- convenience --------------------------------------------------------

    def mean_score(self, results: list[JudgeResult]) -> float:
        """Return the average score across a list of judge results.

        Results with score=0 (error sentinels) are excluded from the
        calculation.

        Args:
            results: List of JudgeResult objects to average.

        Returns:
            Mean score as a float, or 0.0 if no valid results exist.
        """
        valid = [r.score for r in results if r.score > 0]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)
