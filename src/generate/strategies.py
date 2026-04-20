"""Prompting strategies for generating synthetic NL->SQL training pairs.

Each strategy builds specialized prompts and calls the teacher model
to produce (natural_language, sql) pairs at varying difficulty levels.

Strategies:
- SeedExpansion: Varies existing seed examples to create new pairs.
- SelfInstruct: Generates entirely new examples given only a DB schema.
- EvolInstruct: Takes simple examples and increases their complexity.
"""

from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.llm.client import Message, TeacherClient

logger = logging.getLogger(__name__)

DIFFICULTY_DESCRIPTIONS: dict[str, str] = {
    "easy": "single table, basic SELECT/WHERE, simple conditions",
    "medium": "JOINs across 2-3 tables, GROUP BY, HAVING, ORDER BY with LIMIT",
    "hard": "subqueries, window functions (ROW_NUMBER, RANK), CTEs, CASE expressions",
    "expert": "complex multi-step reasoning, correlated subqueries, multiple CTEs, "
    "nested window functions, UNION/INTERSECT",
}


@dataclass
class GeneratedExample:
    """A single generated NL->SQL pair with metadata."""

    natural_language: str
    sql: str
    difficulty: str = "medium"
    category: str = ""
    strategy: str = ""
    metadata: dict = field(default_factory=dict)


def _parse_examples_from_response(response: str, strategy_name: str) -> list[dict]:
    """Parse JSON examples from a teacher model response.

    The teacher is prompted to return a JSON array of objects with
    'natural_language' and 'sql' keys. This parser handles common
    formatting issues like markdown code fences and trailing commas.

    Args:
        response: Raw text response from the teacher model.
        strategy_name: Name of the calling strategy, used in log messages.

    Returns:
        A list of dicts, each containing at least 'natural_language' and
        'sql' keys. May be empty if parsing fails entirely.
    """
    # Strip markdown code fences if present
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try to find a JSON array in the response
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        text = text[bracket_start : bracket_end + 1]

    # Remove trailing commas before closing brackets (common LLM mistake)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        examples = json.loads(text)
        if not isinstance(examples, list):
            examples = [examples]
        return examples
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse JSON from %s response, attempting line-by-line extraction",
            strategy_name,
        )

    # Fallback: try to extract individual JSON objects
    results = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            if "natural_language" in obj and "sql" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if not results:
        logger.error("Could not extract any examples from %s response", strategy_name)
    return results


def _raw_to_generated(
    raw_examples: list[dict],
    difficulty: str,
    strategy_name: str,
) -> list[GeneratedExample]:
    """Convert raw parsed dicts into GeneratedExample objects, filtering bad entries.

    Args:
        raw_examples: Dicts with 'natural_language' and 'sql' keys, as
            returned by ``_parse_examples_from_response``.
        difficulty: Default difficulty label assigned to examples that lack one.
        strategy_name: Strategy name stored on each GeneratedExample.

    Returns:
        A list of GeneratedExample objects. Entries with empty
        'natural_language' or 'sql' values are silently dropped.
    """
    results = []
    for raw in raw_examples:
        nl = raw.get("natural_language", "").strip()
        sql = raw.get("sql", "").strip()
        if not nl or not sql:
            continue
        results.append(
            GeneratedExample(
                natural_language=nl,
                sql=sql,
                difficulty=raw.get("difficulty", difficulty),
                category=raw.get("category", ""),
                strategy=strategy_name,
            )
        )
    return results


class GenerationStrategy(ABC):
    """Base class for all generation strategies."""

    name: str = "base"

    def __init__(self, client: TeacherClient, temperature: float = 0.7) -> None:
        """Initialize the generation strategy.

        Args:
            client: TeacherClient instance used to call the teacher model.
            temperature: Sampling temperature for generation calls.
        """
        self.client = client
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        num_examples: int,
        schema: str,
        seeds: list[dict] | None = None,
        difficulty: str = "medium",
    ) -> list[GeneratedExample]:
        """Generate NL->SQL training pairs.

        Args:
            num_examples: Number of examples to generate.
            schema: SQL schema (CREATE TABLE statements).
            seeds: Optional seed examples, each with 'natural_language' and 'sql' keys.
            difficulty: Target difficulty level (easy/medium/hard/expert).

        Returns:
            List of GeneratedExample objects.
        """
        ...

    def _call_teacher(self, messages: list[Message], temperature: float | None = None) -> str:
        """Call the teacher model with error handling.

        Args:
            messages: Conversation messages to send to the teacher model.
            temperature: Sampling temperature override. Falls back to the
                instance default when ``None``.

        Returns:
            The text content of the teacher model's response.

        Raises:
            Exception: Re-raises any exception from the underlying client
                after logging.
        """
        temp = temperature if temperature is not None else self.temperature
        try:
            return self.client.complete(messages, temperature=temp)
        except Exception:
            logger.exception("Teacher model call failed in %s", self.name)
            raise


class SeedExpansion(GenerationStrategy):
    """Generates variations of existing seed examples.

    Takes seed NL->SQL pairs and asks the teacher to create new examples
    that follow similar patterns but with different tables, filters,
    aggregations, etc. Best for bootstrapping from a small set of
    high-quality hand-written examples.
    """

    name = "seed_expansion"

    def generate(
        self,
        num_examples: int,
        schema: str,
        seeds: list[dict] | None = None,
        difficulty: str = "medium",
    ) -> list[GeneratedExample]:
        """Generate NL->SQL pairs by expanding on existing seed examples.

        Randomly samples subsets of seeds and asks the teacher model to
        produce new, diverse variations in batches of up to 10.

        Args:
            num_examples: Number of examples to generate.
            schema: SQL schema (CREATE TABLE statements).
            seeds: Seed examples, each with 'natural_language' and 'sql' keys.
                Required for this strategy.
            difficulty: Target difficulty level (easy/medium/hard/expert).

        Returns:
            A list of GeneratedExample objects (at most ``num_examples``).

        Raises:
            ValueError: When ``seeds`` is empty or ``None``.
        """
        if not seeds:
            raise ValueError("SeedExpansion requires at least one seed example.")

        all_examples: list[GeneratedExample] = []
        # Work in batches — ask the teacher for up to 10 at a time
        batch_size = min(num_examples, 10)
        remaining = num_examples

        while remaining > 0:
            n = min(remaining, batch_size)
            seed_subset = random.sample(seeds, min(len(seeds), 5))
            seed_text = "\n".join(
                f"  NL: {s['natural_language']}\n  SQL: {s['sql']}" for s in seed_subset
            )

            difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, difficulty)

            messages = [
                Message(
                    role="system",
                    content=(
                        "You are an expert SQL instructor generating training data. "
                        "You produce diverse, realistic NL->SQL pairs for fine-tuning. "
                        "Always respond with a JSON array only — no extra commentary."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Here is a database schema:\n\n```sql\n{schema}\n```\n\n"
                        f"Here are some seed examples of NL->SQL pairs:\n\n{seed_text}\n\n"
                        f"Generate {n} NEW and DIVERSE NL->SQL pairs that follow similar "
                        f"patterns but are meaningfully different from the seeds.\n"
                        f"Target difficulty: {difficulty} ({difficulty_desc}).\n\n"
                        f"Vary the tables, columns, filter conditions, aggregations, and "
                        f"phrasing. Make the natural language sound like real user questions — "
                        f"varied wording, sometimes casual, sometimes formal.\n\n"
                        f"Return a JSON array of objects with keys: "
                        f'"natural_language", "sql", "category".\n'
                        f"Example format:\n"
                        f'[{{"natural_language": "...", "sql": "...", "category": "select"}}]'
                    ),
                ),
            ]

            response = self._call_teacher(messages)
            raw = _parse_examples_from_response(response, self.name)
            batch = _raw_to_generated(raw, difficulty, self.name)
            all_examples.extend(batch)
            remaining -= len(batch)

            if not batch:
                logger.warning("SeedExpansion produced 0 examples in batch, stopping early")
                break

        return all_examples[:num_examples]


class SelfInstruct(GenerationStrategy):
    """Generates new NL->SQL examples from scratch given only a schema.

    Inspired by the Self-Instruct paper. The teacher model invents both
    the natural-language question and the SQL query, guided only by the
    schema and difficulty constraints. Good for broad coverage when you
    have few or no seeds.
    """

    name = "self_instruct"

    def generate(
        self,
        num_examples: int,
        schema: str,
        seeds: list[dict] | None = None,
        difficulty: str = "medium",
    ) -> list[GeneratedExample]:
        """Generate NL->SQL pairs from scratch using only the schema.

        The teacher model invents both the natural-language question and the
        corresponding SQL query. A small number of seeds may optionally be
        included for format illustration, but they are not required.

        Args:
            num_examples: Number of examples to generate.
            schema: SQL schema (CREATE TABLE statements).
            seeds: Optional seed examples used as format references only.
            difficulty: Target difficulty level (easy/medium/hard/expert).

        Returns:
            A list of GeneratedExample objects (at most ``num_examples``).
        """
        all_examples: list[GeneratedExample] = []
        batch_size = min(num_examples, 10)
        remaining = num_examples

        difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, difficulty)

        # Optionally include a couple of seeds as illustration (not required)
        seed_block = ""
        if seeds:
            sample = random.sample(seeds, min(len(seeds), 2))
            formatted = "\n".join(
                f'  {{"natural_language": "{s["natural_language"]}", "sql": "{s["sql"]}"}}'
                for s in sample
            )
            seed_block = (
                f"\nHere are a couple of example pairs for reference format "
                f"(generate DIFFERENT ones):\n{formatted}\n"
            )

        while remaining > 0:
            n = min(remaining, batch_size)

            messages = [
                Message(
                    role="system",
                    content=(
                        "You are an expert SQL instructor creating a training dataset. "
                        "Given a database schema, invent realistic questions a user might "
                        "ask about the data, along with the correct SQL query. "
                        "Always respond with a JSON array only — no extra commentary."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Database schema:\n\n```sql\n{schema}\n```\n"
                        f"{seed_block}\n"
                        f"Generate {n} diverse, realistic NL->SQL pairs.\n"
                        f"Target difficulty: {difficulty} ({difficulty_desc}).\n\n"
                        f"Requirements:\n"
                        f"- Each natural language question should sound like a real user.\n"
                        f"- SQL must be valid for the given schema.\n"
                        f"- Cover different query types: SELECT, aggregations, JOINs, "
                        f"filtering, sorting as appropriate for the difficulty.\n"
                        f"- Vary phrasing and vocabulary in the natural language.\n\n"
                        f"Return a JSON array of objects with keys: "
                        f'"natural_language", "sql", "category".\n'
                    ),
                ),
            ]

            response = self._call_teacher(messages)
            raw = _parse_examples_from_response(response, self.name)
            batch = _raw_to_generated(raw, difficulty, self.name)
            all_examples.extend(batch)
            remaining -= len(batch)

            if not batch:
                logger.warning("SelfInstruct produced 0 examples in batch, stopping early")
                break

        return all_examples[:num_examples]


class EvolInstruct(GenerationStrategy):
    """Evolves existing examples into harder versions.

    Inspired by WizardLM's Evol-Instruct. Takes a simple NL->SQL pair
    and asks the teacher to create a more complex version — adding JOINs,
    subqueries, window functions, etc. Useful for generating hard/expert
    examples from an existing pool of easy/medium ones.
    """

    name = "evol_instruct"

    EVOLUTION_TYPES: list[str] = [
        "add a JOIN with another table",
        "add a subquery or CTE",
        "add a window function (ROW_NUMBER, RANK, LAG, etc.)",
        "add GROUP BY with HAVING",
        "combine multiple conditions with AND/OR/NOT",
        "add ORDER BY with LIMIT/OFFSET",
        "make the question require a CASE expression",
        "add a UNION or INTERSECT with another query",
    ]

    def generate(
        self,
        num_examples: int,
        schema: str,
        seeds: list[dict] | None = None,
        difficulty: str = "hard",
    ) -> list[GeneratedExample]:
        """Evolve simple seed examples into harder NL->SQL pairs.

        Picks seeds one at a time, selects random evolution transformations
        (e.g. adding JOINs, subqueries, window functions), and asks the
        teacher to produce more complex versions.

        Args:
            num_examples: Number of evolved examples to generate.
            schema: SQL schema (CREATE TABLE statements).
            seeds: Seed examples to evolve, each with 'natural_language' and
                'sql' keys. Required for this strategy.
            difficulty: Target difficulty level (easy/medium/hard/expert).

        Returns:
            A list of GeneratedExample objects (at most ``num_examples``).

        Raises:
            ValueError: When ``seeds`` is empty or ``None``.
        """
        if not seeds:
            raise ValueError("EvolInstruct requires seed examples to evolve.")

        all_examples: list[GeneratedExample] = []
        # Evolve one seed at a time for quality — each prompt focuses on one pair
        source_pool = seeds.copy()
        remaining = num_examples

        while remaining > 0 and source_pool:
            # Pick a seed to evolve
            seed = random.choice(source_pool)
            # Pick 1-2 evolution types
            evolutions = random.sample(self.EVOLUTION_TYPES, min(2, len(self.EVOLUTION_TYPES)))
            evolution_instructions = " and ".join(evolutions)

            difficulty_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, difficulty)

            messages = [
                Message(
                    role="system",
                    content=(
                        "You are an expert SQL instructor. Your task is to take a simple "
                        "NL->SQL pair and evolve it into a more complex version. "
                        "The evolved question should require a harder SQL query while "
                        "remaining natural and realistic. "
                        "Always respond with a JSON array only — no extra commentary."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Database schema:\n\n```sql\n{schema}\n```\n\n"
                        f"Original example:\n"
                        f"  NL: {seed['natural_language']}\n"
                        f"  SQL: {seed['sql']}\n\n"
                        f"Evolve this into 3 harder examples by applying these "
                        f"transformations: {evolution_instructions}.\n"
                        f"Target difficulty: {difficulty} ({difficulty_desc}).\n\n"
                        f"The evolved natural language should still sound like a "
                        f"real user question, not overly technical.\n"
                        f"The SQL must be valid for the given schema.\n\n"
                        f"Return a JSON array of objects with keys: "
                        f'"natural_language", "sql", "category".\n'
                    ),
                ),
            ]

            response = self._call_teacher(messages)
            raw = _parse_examples_from_response(response, self.name)
            batch = _raw_to_generated(raw, difficulty, self.name)
            all_examples.extend(batch)
            remaining -= len(batch)

            if not batch:
                # Remove this seed from pool so we don't retry it forever
                source_pool.remove(seed)
                logger.warning(
                    "EvolInstruct failed to evolve a seed, %d seeds left", len(source_pool)
                )

        return all_examples[:num_examples]


# Registry for looking up strategies by name
STRATEGY_REGISTRY: dict[str, type[GenerationStrategy]] = {
    "seed_expansion": SeedExpansion,
    "self_instruct": SelfInstruct,
    "evol_instruct": EvolInstruct,
}


def get_strategy(name: str, client: TeacherClient, **kwargs) -> GenerationStrategy:
    """Instantiate a generation strategy by name.

    Args:
        name: One of 'seed_expansion', 'self_instruct', 'evol_instruct'.
        client: TeacherClient instance.
        **kwargs: Extra keyword arguments passed to the strategy constructor.

    Returns:
        An initialized GenerationStrategy.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_REGISTRY)}")
    return cls(client=client, **kwargs)
