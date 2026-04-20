"""Few-shot example management for NL->SQL generation prompts.

Handles loading seed examples from JSONL files and formatting them
into prompt blocks for the teacher model. Supports random sampling
and difficulty-based selection to control the style of generated pairs.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SeedExample:
    """A single seed NL->SQL pair loaded from disk."""

    natural_language: str
    sql: str
    difficulty: str = "medium"
    category: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the seed example to a plain dictionary.

        Returns:
            A dict with 'natural_language', 'sql', 'difficulty', and
            'category' keys.
        """
        return {
            "natural_language": self.natural_language,
            "sql": self.sql,
            "difficulty": self.difficulty,
            "category": self.category,
        }


class SeedLoader:
    """Loads seed examples from JSONL files.

    Each line in the JSONL file should be a JSON object with at least
    'natural_language' and 'sql' keys. Optional keys: 'difficulty',
    'category', and any extra metadata.

    Usage:
        loader = SeedLoader("data/seeds/sql_examples.jsonl")
        seeds = loader.load()
        easy_seeds = loader.filter_by_difficulty("easy")
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the seed loader.

        Args:
            path: Path to a JSONL file containing seed examples.
        """
        self.path = Path(path)
        self._examples: list[SeedExample] = []
        self._loaded = False

    def load(self) -> list[SeedExample]:
        """Load and parse all seed examples from the JSONL file.

        Returns:
            A list of SeedExample objects parsed from the file. Subsequent
            calls return the cached result without re-reading.

        Raises:
            FileNotFoundError: When the seed file does not exist.
        """
        if self._loaded:
            return self._examples

        if not self.path.exists():
            raise FileNotFoundError(f"Seed file not found: {self.path}")

        examples = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON on line %d of %s", line_num, self.path)
                    continue

                nl = data.get("natural_language", "").strip()
                sql = data.get("sql", "").strip()
                if not nl or not sql:
                    logger.warning("Skipping line %d: missing natural_language or sql", line_num)
                    continue

                # Pull known fields out, put the rest in metadata
                known_keys = {"natural_language", "sql", "difficulty", "category"}
                extra = {k: v for k, v in data.items() if k not in known_keys}

                examples.append(
                    SeedExample(
                        natural_language=nl,
                        sql=sql,
                        difficulty=data.get("difficulty", "medium"),
                        category=data.get("category", ""),
                        metadata=extra,
                    )
                )

        self._examples = examples
        self._loaded = True
        logger.info("Loaded %d seed examples from %s", len(examples), self.path)
        return self._examples

    def filter_by_difficulty(self, difficulty: str) -> list[SeedExample]:
        """Return only seeds matching the given difficulty.

        Args:
            difficulty: Difficulty level to filter on (e.g. 'easy', 'medium').

        Returns:
            A list of SeedExample objects whose difficulty matches.
        """
        if not self._loaded:
            self.load()
        return [ex for ex in self._examples if ex.difficulty == difficulty]

    def filter_by_category(self, category: str) -> list[SeedExample]:
        """Return only seeds matching the given category.

        Args:
            category: Category string to filter on.

        Returns:
            A list of SeedExample objects whose category matches.
        """
        if not self._loaded:
            self.load()
        return [ex for ex in self._examples if ex.category == category]

    @property
    def examples(self) -> list[SeedExample]:
        """All loaded seed examples.

        Returns:
            The full list of SeedExample objects, loading from disk on first
            access if needed.
        """
        if not self._loaded:
            self.load()
        return self._examples

    def to_dicts(self) -> list[dict]:
        """Return all examples as plain dicts (for passing to strategies).

        Returns:
            A list of dicts, each with 'natural_language', 'sql',
            'difficulty', and 'category' keys.
        """
        return [ex.to_dict() for ex in self.examples]


class FewShotFormatter:
    """Selects and formats few-shot examples for inclusion in prompts.

    Supports multiple selection modes:
    - random: uniformly random sample
    - difficulty: weighted sampling that favors examples at the target difficulty
    - category: filter by SQL category before sampling

    Usage:
        formatter = FewShotFormatter(seeds)
        prompt_block = formatter.format(n=3, difficulty="hard")
    """

    def __init__(self, examples: list[SeedExample]) -> None:
        """Initialize the formatter and build internal lookup indexes.

        Args:
            examples: Seed examples available for few-shot selection.
        """
        self.examples = list(examples)
        if not self.examples:
            logger.warning("FewShotFormatter initialized with no examples")

        # Build indexes for fast lookup
        self._by_difficulty: dict[str, list[SeedExample]] = {}
        self._by_category: dict[str, list[SeedExample]] = {}
        for ex in self.examples:
            self._by_difficulty.setdefault(ex.difficulty, []).append(ex)
            if ex.category:
                self._by_category.setdefault(ex.category, []).append(ex)

    def select(
        self,
        n: int,
        mode: str = "random",
        difficulty: str | None = None,
        category: str | None = None,
    ) -> list[SeedExample]:
        """Select few-shot examples according to the given mode.

        Args:
            n: Number of examples to select.
            mode: Selection mode — 'random', 'difficulty', or 'category'.
            difficulty: Target difficulty for 'difficulty' mode. Also used as a
                preference hint in 'random' mode when provided.
            category: Target category for 'category' mode.

        Returns:
            A list of selected SeedExample objects (at most n).
        """
        pool = self.examples

        if mode == "category" and category:
            pool = self._by_category.get(category, self.examples)

        if mode == "difficulty" and difficulty:
            # Weighted selection: 70% from target difficulty, 30% from others
            target_pool = self._by_difficulty.get(difficulty, [])
            other_pool = [ex for ex in self.examples if ex.difficulty != difficulty]

            n_target = min(max(1, int(n * 0.7)), len(target_pool))
            n_other = min(n - n_target, len(other_pool))

            selected = random.sample(target_pool, n_target) if target_pool else []
            if other_pool and n_other > 0:
                selected += random.sample(other_pool, n_other)
            return selected[:n]

        # Default: random sampling
        return random.sample(pool, min(n, len(pool)))

    def format(
        self,
        n: int = 3,
        mode: str = "random",
        difficulty: str | None = None,
        category: str | None = None,
        style: str = "numbered",
    ) -> str:
        """Select examples and format them into a prompt-ready string.

        Args:
            n: Number of few-shot examples.
            mode: Selection mode (random/difficulty/category).
            difficulty: Target difficulty.
            category: Target category.
            style: Formatting style — 'numbered', 'labeled', or 'json'.

        Returns:
            A formatted string block ready to paste into a prompt.
        """
        selected = self.select(n, mode=mode, difficulty=difficulty, category=category)

        if not selected:
            return "(No few-shot examples available.)"

        if style == "json":
            return _format_json(selected)
        elif style == "labeled":
            return _format_labeled(selected)
        else:
            return _format_numbered(selected)


def _format_numbered(examples: list[SeedExample]) -> str:
    """Format examples as a numbered list.

    Args:
        examples: Seed examples to format.

    Returns:
        A multi-line string with numbered question/SQL/difficulty entries.
    """
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Question: {ex.natural_language}")
        lines.append(f"  SQL: {ex.sql}")
        if ex.difficulty:
            lines.append(f"  Difficulty: {ex.difficulty}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_labeled(examples: list[SeedExample]) -> str:
    """Format examples with labeled fields, no numbering.

    Args:
        examples: Seed examples to format.

    Returns:
        A multi-line string with NL/SQL labels separated by '---' dividers.
    """
    lines = []
    for ex in examples:
        lines.append(f"NL: {ex.natural_language}")
        lines.append(f"SQL: {ex.sql}")
        lines.append("---")
    return "\n".join(lines).rstrip("- \n")


def _format_json(examples: list[SeedExample]) -> str:
    """Format examples as a JSON array (useful when asking LLM to return JSON).

    Args:
        examples: Seed examples to format.

    Returns:
        A pretty-printed JSON string containing an array of example dicts.
    """
    return json.dumps([ex.to_dict() for ex in examples], indent=2)
