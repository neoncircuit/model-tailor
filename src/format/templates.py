"""Chat/conversation templates for instruction-tuning data.

Converts curated NL->SQL pairs into structured conversation format
suitable for fine-tuning with SFTTrainer, TRL, or similar frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_SQL_SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. Given a natural language question and "
    "an optional database schema, generate a correct SQL query that answers "
    "the question. Output only the SQL query, without explanation."
)

DETAILED_SQL_SYSTEM_PROMPT = (
    "You are an expert SQL assistant specializing in translating natural "
    "language questions into precise SQL queries. Follow these rules:\n"
    "1. Use standard SQL syntax compatible with PostgreSQL.\n"
    "2. Use meaningful table aliases when joining multiple tables.\n"
    "3. Prefer explicit JOIN syntax over implicit joins.\n"
    "4. Include appropriate WHERE clauses to filter results.\n"
    "5. Output only the SQL query, without any explanation or markdown."
)


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Conversation:
    """A multi-turn conversation for instruction tuning."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> Conversation:
        """Append a turn and return self for chaining.

        Args:
            role: The role for this turn ("system", "user", or "assistant").
            content: The text content of the turn.

        Returns:
            This conversation instance, allowing method chaining.
        """
        self.turns.append(Turn(role=role, content=content))
        return self

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to HuggingFace messages format.

        Returns:
            A list of dicts, each containing "role" and "content" keys.
        """
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def to_sharegpt_turns(self) -> list[dict[str, str]]:
        """Convert to ShareGPT format with "from"/"value" dicts.

        Roles are mapped as follows: "user" -> "human",
        "assistant" -> "gpt", "system" -> "system".

        Returns:
            A list of dicts, each containing "from" and "value" keys.
        """
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}
        return [{"from": role_map.get(t.role, t.role), "value": t.content} for t in self.turns]

    def __len__(self) -> int:
        """Return the number of turns in this conversation.

        Returns:
            The number of turns.
        """
        return len(self.turns)


class ChatTemplate:
    """Builds instruction-tuning conversations from NL->SQL example pairs.

    Args:
        system_prompt: System prompt to prepend to every conversation. If
            None, uses the default SQL generation system prompt.
        include_schema: If True and the example contains a "schema" key,
            the schema is included in the user message.
        include_difficulty: If True and the example contains a "difficulty"
            key, a difficulty hint is appended to the user message.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        include_schema: bool = True,
        include_difficulty: bool = False,
    ) -> None:
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_SQL_SYSTEM_PROMPT
        )
        self.include_schema = include_schema
        self.include_difficulty = include_difficulty

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def _build_user_content(self, example: dict[str, Any]) -> str:
        """Compose the user turn content from an NL->SQL example.

        Args:
            example: A dict containing at least an "nl" key. May also
                contain "schema" and "difficulty" keys.

        Returns:
            The assembled user message string.
        """
        parts: list[str] = []

        if self.include_schema and "schema" in example and example["schema"]:
            parts.append(f"Schema:\n{example['schema']}\n")

        parts.append(example["nl"])

        if self.include_difficulty and "difficulty" in example:
            parts.append(f"\n[Difficulty: {example['difficulty']}]")

        return "\n".join(parts) if len(parts) > 1 else parts[0]

    def build_conversation(
        self,
        example: dict[str, Any],
        system_prompt: str | None = None,
    ) -> Conversation:
        """Build a single Conversation from an NL->SQL example.

        Args:
            example: Must contain "nl" (natural language) and "sql" keys.
                May optionally contain "schema" and "difficulty".
            system_prompt: Override the instance-level system prompt for this
                conversation. If None, uses ``self.system_prompt``.

        Returns:
            A Conversation with system, user, and assistant turns.
        """
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        user_content = self._build_user_content(example)

        conv = Conversation()
        conv.add("system", prompt)
        conv.add("user", user_content)
        conv.add("assistant", example["sql"])
        return conv

    # ------------------------------------------------------------------
    # Batch conversions
    # ------------------------------------------------------------------

    def to_conversations(
        self,
        examples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[Conversation]:
        """Convert a batch of examples to Conversation objects.

        Args:
            examples: List of NL->SQL example dicts, each containing at
                least "nl" and "sql" keys.
            system_prompt: Override the instance-level system prompt for
                all conversations. If None, uses ``self.system_prompt``.

        Returns:
            A list of Conversation objects, one per example.
        """
        return [self.build_conversation(ex, system_prompt=system_prompt) for ex in examples]

    def to_messages(
        self,
        examples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert examples to HuggingFace ``messages`` column format.

        Each returned dict has a "messages" key holding a list of
        role/content dicts. This is directly compatible with
        ``datasets.Dataset.from_list`` and SFTTrainer.

        Args:
            examples: List of NL->SQL example dicts.
            system_prompt: Override the instance-level system prompt. If
                None, uses ``self.system_prompt``.

        Returns:
            A list of dicts, each with a "messages" key.
        """
        return [
            {"messages": self.build_conversation(ex, system_prompt=system_prompt).to_messages()}
            for ex in examples
        ]

    def to_sharegpt(
        self,
        examples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert examples to ShareGPT format.

        Each returned dict has a "conversations" key containing turns
        as ``{"from": "human"/"gpt"/"system", "value": "..."}``.

        Args:
            examples: List of NL->SQL example dicts.
            system_prompt: Override the instance-level system prompt. If
                None, uses ``self.system_prompt``.

        Returns:
            A list of dicts, each with a "conversations" key.
        """
        results: list[dict[str, Any]] = []
        for ex in examples:
            conv = self.build_conversation(ex, system_prompt=system_prompt)
            results.append({"conversations": conv.to_sharegpt_turns()})
        return results

    def to_prompt_completion(
        self,
        examples: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Convert examples to simple prompt/completion pairs.

        Useful for legacy fine-tuning APIs that expect a flat
        "prompt" + "completion" format.

        Args:
            examples: List of NL->SQL example dicts.
            system_prompt: Override the instance-level system prompt. If
                None, uses ``self.system_prompt``.

        Returns:
            A list of dicts, each with "prompt" and "completion" keys.
        """
        prompt_prefix = system_prompt if system_prompt is not None else self.system_prompt
        results: list[dict[str, str]] = []
        for ex in examples:
            user_content = self._build_user_content(ex)
            prompt = f"{prompt_prefix}\n\n{user_content}" if prompt_prefix else user_content
            results.append({"prompt": prompt, "completion": ex["sql"]})
        return results
