"""Repair / fixer stage for failed SQL generation.

Takes a generated SQL query that failed the execution gate, prompts the teacher
model with the failure feedback, and re-runs the gate on the corrected query.
This is the "fixer" in the writer -> gate -> fixer -> gate loop inspired by the
recent LinkedIn case study on deterministic correctness gates.

Usage:
    repairer = SQLRepairer(client=client, gate=gate, max_attempts=2)
    result = repairer.repair(record, schema=schema)
    if result.final_result.passed:
        print(f"Fixed after {result.attempts} attempts")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.evaluate.gate import ExecutionGate, ExecutionResult
from src.llm.client import Message, TeacherClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_REPAIR_TEMPERATURE = 0.3


@dataclass
class RepairResult:
    """Outcome of attempting to repair a failing SQL query.

    Attributes:
        original_sql: The SQL query that originally failed the gate.
        repaired_sql: The final SQL query after repair attempts. When the
            original query already passes, this is identical to
            ``original_sql``.
        attempts: Number of repair attempts made. Zero when the original SQL
            passed the gate on the first check.
        final_result: The ``ExecutionResult`` from the final gate check.
        history: A chronological list of repair attempt details. Each entry is
            a dict with keys such as ``attempt``, ``action``, ``sql``,
            ``status`` and ``error_message``.
    """

    original_sql: str
    repaired_sql: str
    attempts: int
    final_result: ExecutionResult
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a compact human-readable summary of the repair outcome."""
        if self.final_result.passed:
            if self.attempts == 0:
                return "[PASS: no repair needed]"
            return f"[REPAIRED after {self.attempts} attempt(s)]"
        return f"[FAIL after {self.attempts} attempt(s): {self.final_result.status}]"


class SQLRepairer:
    """Fix failing SQL queries using teacher-model feedback and the gate.

    The repairer checks the SQL with the execution gate. If it fails, it builds
    a prompt containing the natural language question, the failing SQL, the
    gate status and error message, and the database schema, and asks the
    teacher model for a corrected query. The corrected query is checked again,
    and the loop continues until it passes or ``max_attempts`` is reached.

    Args:
        client: TeacherClient used to ask the model for corrected SQL.
        gate: ExecutionGate that validates SQL execution correctness.
        max_attempts: Maximum number of repair attempts per failing query.
        temperature: Sampling temperature for repair prompts. Lower values
            discourage unnecessary creativity when fixing syntax/semantics.
    """

    def __init__(
        self,
        client: TeacherClient,
        gate: ExecutionGate,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float = DEFAULT_REPAIR_TEMPERATURE,
    ) -> None:
        """Initialize the repairer.

        Args:
            client: TeacherClient instance for LLM calls.
            gate: ExecutionGate instance for validating repaired SQL.
            max_attempts: Maximum repair attempts for a single failing query.
            temperature: Sampling temperature for repair generation.
        """
        self.client = client
        self.gate = gate
        self.max_attempts = max(max_attempts, 0)
        self.temperature = temperature

    def repair(self, record: dict, schema: str | None = None) -> RepairResult:
        """Attempt to repair a single generated SQL record.

        Args:
            record: Dict with at least ``natural_language`` and ``sql`` keys.
                An optional ``target_sql`` key is used by the gate for
                result-set comparison but is not exposed to the repair prompt.
            schema: Optional SQL schema for context in the repair prompt and
                gate execution.

        Returns:
            RepairResult describing the original query, any repaired query,
            the number of attempts, and the final gate result.
        """
        original_sql = record.get("sql", "").strip()
        natural_language = record.get("natural_language", "").strip()
        target_sql = record.get("target_sql")

        history: list[dict] = []

        initial_result = self.gate.check(original_sql, target_sql)
        history.append(
            {
                "attempt": 0,
                "action": "initial_check",
                "sql": original_sql,
                "status": initial_result.status,
                "passed": initial_result.passed,
                "error_message": initial_result.error_message,
            }
        )

        if initial_result.passed:
            return RepairResult(
                original_sql=original_sql,
                repaired_sql=original_sql,
                attempts=0,
                final_result=initial_result,
                history=history,
            )

        current_sql = original_sql
        final_result = initial_result

        for attempt in range(1, self.max_attempts + 1):
            try:
                corrected_sql = self._ask_teacher(
                    natural_language=natural_language,
                    failing_sql=current_sql,
                    gate_result=final_result,
                    schema=schema,
                )
            except Exception:
                logger.exception(
                    "Teacher repair call failed for record: %s",
                    natural_language[:80],
                )
                history.append(
                    {
                        "attempt": attempt,
                        "action": "teacher_call_failed",
                        "sql": None,
                        "status": "teacher_error",
                        "passed": False,
                        "error_message": "Teacher model call raised an exception",
                    }
                )
                break

            history.append(
                {
                    "attempt": attempt,
                    "action": "repair_attempt",
                    "sql": corrected_sql,
                    "status": "pending",
                    "passed": False,
                    "error_message": None,
                }
            )

            final_result = self.gate.check(corrected_sql, target_sql)
            history[-1]["status"] = final_result.status
            history[-1]["passed"] = final_result.passed
            history[-1]["error_message"] = final_result.error_message

            if final_result.passed:
                return RepairResult(
                    original_sql=original_sql,
                    repaired_sql=corrected_sql,
                    attempts=attempt,
                    final_result=final_result,
                    history=history,
                )

            current_sql = corrected_sql

        return RepairResult(
            original_sql=original_sql,
            repaired_sql=current_sql,
            attempts=self.max_attempts,
            final_result=final_result,
            history=history,
        )

    def _ask_teacher(
        self,
        natural_language: str,
        failing_sql: str,
        gate_result: ExecutionResult,
        schema: str | None,
    ) -> str:
        """Prompt the teacher model for a corrected SQL query.

        Args:
            natural_language: The natural language question the SQL should
                answer.
            failing_sql: The SQL query that failed the gate.
            gate_result: The ``ExecutionResult`` containing the failure status
                and message.
            schema: Optional SQL schema to ground the corrected query.

        Returns:
            The corrected SQL string extracted from the model response.
        """
        schema_block = ""
        if schema:
            schema_block = f"\n\nDatabase schema:\n```sql\n{schema}\n```\n"

        messages = [
            Message(
                role="system",
                content=(
                    "You are an expert SQL assistant. A generated SQL query "
                    "failed validation against a real database. Correct the "
                    "query so it accurately answers the user's question. "
                    "Return ONLY the corrected SQL query, with no explanation, "
                    "no markdown commentary, and no code fences."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Natural language question:{schema_block}\n\n"
                    f"{natural_language}\n\n"
                    f"Failing SQL query:\n{failing_sql}\n\n"
                    f"Validation failure: {gate_result.status}\n"
                    f"Error details: {gate_result.error_message or 'No details'}\n\n"
                    "Provide the corrected SQL query only:"
                ),
            ),
        ]

        response = self.client.complete(messages, temperature=self.temperature)
        return self._extract_sql(response)

    @staticmethod
    def _extract_sql(response: str) -> str:
        """Extract a SQL query from a model response.

        Strips markdown code fences and surrounding prose. If no fenced block
        is found, returns the trimmed response.

        Args:
            response: Raw text response from the teacher model.

        Returns:
            The extracted SQL string, stripped of fences and whitespace.
        """
        text = response.strip()

        # Match a ```sql ... ``` or ``` ... ``` block, capturing the content.
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Some models prefix the SQL with a label like "SQL:"; remove it.
        cleaned = re.sub(r"^(?:sql|query)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        return cleaned.strip()
