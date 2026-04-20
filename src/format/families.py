"""Model family adapters for applying chat templates.

Each model family has its own special-token conventions for
instruction-tuning.  The adapters here render a ``Conversation``
(from ``templates.py``) into the raw string that the model's
tokenizer expects, including all special tokens.

Supported families
------------------
- **Llama 3 / 3.1** -- Meta's latest chat format.
- **Mistral Instruct** -- Mistral AI instruct format.
- **Phi-3** -- Microsoft Phi-3 chat format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.format.templates import Conversation


class ModelFamily(ABC):
    """Base class for model-family chat-template adapters."""

    name: str = "base"

    @abstractmethod
    def apply_template(self, conversation: Conversation) -> str:
        """Render a Conversation into the model's expected string format.

        Args:
            conversation: A structured multi-turn conversation
                (system + user + assistant).

        Returns:
            The fully formatted string with all special tokens, ready to
            be tokenized by the model's tokenizer.
        """

    def apply_batch(self, conversations: list[Conversation]) -> list[str]:
        """Apply the template to a batch of conversations.

        Args:
            conversations: List of Conversation objects to format.

        Returns:
            A list of fully formatted strings, one per conversation.
        """
        return [self.apply_template(c) for c in conversations]

    def format_for_dataset(
        self,
        conversations: list[Conversation],
    ) -> list[dict[str, str]]:
        """Return a list of dicts with a "text" key for HuggingFace datasets.

        This is the format expected by ``SFTTrainer`` when using
        ``dataset_text_field="text"``.

        Args:
            conversations: List of Conversation objects to format.

        Returns:
            A list of dicts, each containing a "text" key with the
            fully formatted string.
        """
        return [{"text": self.apply_template(c)} for c in conversations]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ======================================================================
# Llama 3 / 3.1
# ======================================================================


class LlamaFamily(ModelFamily):
    """Meta Llama 3 / 3.1 instruct chat template.

    Format::

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {assistant_message}<|eot_id|>
    """

    name: str = "llama"

    def apply_template(self, conversation: Conversation) -> str:
        """Render a Conversation into Llama 3 instruct format.

        Args:
            conversation: A structured multi-turn conversation.

        Returns:
            The formatted string with Llama special tokens.
        """
        parts: list[str] = ["<|begin_of_text|>"]

        for turn in conversation.turns:
            parts.append(
                f"<|start_header_id|>{turn.role}<|end_header_id|>\n\n{turn.content}<|eot_id|>"
            )

        return "".join(parts)


# ======================================================================
# Mistral Instruct
# ======================================================================


class MistralFamily(ModelFamily):
    """Mistral instruct chat template.

    Format::

        <s>[INST] {system_prompt}

        {user_message} [/INST]{assistant_message}</s>

    For multi-turn, subsequent user turns are wrapped in ``[INST]...[/INST]``
    without the system prompt.
    """

    name: str = "mistral"

    def apply_template(self, conversation: Conversation) -> str:
        """Render a Conversation into Mistral instruct format.

        Args:
            conversation: A structured multi-turn conversation.

        Returns:
            The formatted string with Mistral special tokens.
        """
        parts: list[str] = ["<s>"]

        system_content: str | None = None
        pending_user: str | None = None

        for turn in conversation.turns:
            if turn.role == "system":
                system_content = turn.content

            elif turn.role == "user":
                # Build the [INST] block; include system prompt only for
                # the first user turn.
                inst_parts: list[str] = []
                if system_content is not None:
                    inst_parts.append(system_content)
                    inst_parts.append("")  # blank line separator
                    system_content = None  # only include once
                inst_parts.append(turn.content)
                pending_user = "\n".join(inst_parts)

            elif turn.role == "assistant":
                if pending_user is not None:
                    parts.append(f"[INST] {pending_user} [/INST]")
                    pending_user = None
                parts.append(f"{turn.content}</s>")

        # Handle trailing user turn with no assistant response (inference)
        if pending_user is not None:
            parts.append(f"[INST] {pending_user} [/INST]")

        return "".join(parts)


# ======================================================================
# Phi-3
# ======================================================================


class PhiFamily(ModelFamily):
    """Microsoft Phi-3 instruct chat template.

    Format::

        <|system|>
        {system_message}<|end|>
        <|user|>
        {user_message}<|end|>
        <|assistant|>
        {assistant_message}<|end|>
    """

    name: str = "phi"

    def apply_template(self, conversation: Conversation) -> str:
        """Render a Conversation into Phi-3 instruct format.

        Args:
            conversation: A structured multi-turn conversation.

        Returns:
            The formatted string with Phi-3 special tokens.
        """
        parts: list[str] = []

        for turn in conversation.turns:
            tag = turn.role  # system | user | assistant
            parts.append(f"<|{tag}|>\n{turn.content}<|end|>")

        return "\n".join(parts)


# ======================================================================
# Factory
# ======================================================================

_FAMILY_REGISTRY: dict[str, type[ModelFamily]] = {
    "llama": LlamaFamily,
    "llama3": LlamaFamily,
    "llama3.1": LlamaFamily,
    "mistral": MistralFamily,
    "mixtral": MistralFamily,
    "phi": PhiFamily,
    "phi3": PhiFamily,
    "phi-3": PhiFamily,
}


def get_family(name: str) -> ModelFamily:
    """Look up and instantiate a model family adapter by name.

    Args:
        name: One of "llama", "mistral", "phi" (and common aliases
            such as "llama3", "mixtral", "phi-3").

    Returns:
        An instantiated ModelFamily adapter for the given name.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = name.lower().strip()
    if key not in _FAMILY_REGISTRY:
        supported = sorted(set(cls.name for cls in _FAMILY_REGISTRY.values()))
        raise ValueError(f"Unknown model family {name!r}. Supported: {supported}")
    return _FAMILY_REGISTRY[key]()


def list_families() -> list[str]:
    """Return the canonical names of all supported model families."""
    return sorted(set(cls.name for cls in _FAMILY_REGISTRY.values()))
