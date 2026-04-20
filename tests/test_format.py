"""Tests for src/format/templates.py and src/format/families.py.

All tests run without API keys, GPU, or external services.
"""

from __future__ import annotations

import pytest

from src.format.families import (
    LlamaFamily,
    MistralFamily,
    PhiFamily,
    get_family,
    list_families,
)
from src.format.templates import ChatTemplate, Conversation, Turn

# ======================================================================
# Turn
# ======================================================================


class TestTurn:
    """Tests for the Turn dataclass."""

    def test_instantiation(self) -> None:
        """Turn stores role and content correctly."""
        turn = Turn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_instantiation_system(self) -> None:
        """Turn accepts the system role."""
        turn = Turn(role="system", content="You are helpful.")
        assert turn.role == "system"
        assert turn.content == "You are helpful."


# ======================================================================
# Conversation
# ======================================================================


class TestConversation:
    """Tests for the Conversation class."""

    def test_add_chaining(self) -> None:
        """add() returns the same Conversation instance for chaining."""
        conv = Conversation()
        result = conv.add("system", "sys").add("user", "hi")
        assert result is conv
        assert len(conv.turns) == 2

    def test_to_messages(self) -> None:
        """to_messages() returns HuggingFace-compatible role/content dicts."""
        conv = Conversation()
        conv.add("system", "Be helpful.")
        conv.add("user", "What is 1+1?")
        conv.add("assistant", "2")

        msgs = conv.to_messages()
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1] == {"role": "user", "content": "What is 1+1?"}
        assert msgs[2] == {"role": "assistant", "content": "2"}

    def test_to_sharegpt_turns_role_mapping(self) -> None:
        """to_sharegpt_turns() maps user->human, assistant->gpt, system->system."""
        conv = Conversation()
        conv.add("system", "sys")
        conv.add("user", "question")
        conv.add("assistant", "answer")

        turns = conv.to_sharegpt_turns()
        assert turns[0]["from"] == "system"
        assert turns[1]["from"] == "human"
        assert turns[2]["from"] == "gpt"

        # Values are preserved.
        assert turns[0]["value"] == "sys"
        assert turns[1]["value"] == "question"
        assert turns[2]["value"] == "answer"

    def test_len(self) -> None:
        """__len__() returns the number of turns."""
        conv = Conversation()
        assert len(conv) == 0
        conv.add("user", "hi")
        assert len(conv) == 1
        conv.add("assistant", "hello")
        assert len(conv) == 2


# ======================================================================
# ChatTemplate
# ======================================================================


EXAMPLE = {"nl": "Show all users", "sql": "SELECT * FROM users"}
EXAMPLE_WITH_SCHEMA = {
    "nl": "Show all users",
    "sql": "SELECT * FROM users",
    "schema": "CREATE TABLE users (id INT, name TEXT)",
}
EXAMPLE_WITH_DIFFICULTY = {
    "nl": "Show all users",
    "sql": "SELECT * FROM users",
    "difficulty": "easy",
}


class TestChatTemplate:
    """Tests for the ChatTemplate class."""

    def test_build_conversation_three_turns(self) -> None:
        """build_conversation() creates a 3-turn conversation (system, user, assistant)."""
        tpl = ChatTemplate()
        conv = tpl.build_conversation(EXAMPLE)
        assert len(conv) == 3
        assert conv.turns[0].role == "system"
        assert conv.turns[1].role == "user"
        assert conv.turns[2].role == "assistant"
        assert conv.turns[2].content == EXAMPLE["sql"]

    def test_build_user_content_includes_schema(self) -> None:
        """_build_user_content() includes the schema when include_schema=True."""
        tpl = ChatTemplate(include_schema=True)
        content = tpl._build_user_content(EXAMPLE_WITH_SCHEMA)
        assert "CREATE TABLE users" in content
        assert "Show all users" in content

    def test_build_user_content_excludes_schema(self) -> None:
        """_build_user_content() omits schema when include_schema=False."""
        tpl = ChatTemplate(include_schema=False)
        content = tpl._build_user_content(EXAMPLE_WITH_SCHEMA)
        assert "CREATE TABLE" not in content

    def test_build_user_content_includes_difficulty(self) -> None:
        """_build_user_content() includes difficulty when include_difficulty=True."""
        tpl = ChatTemplate(include_difficulty=True)
        content = tpl._build_user_content(EXAMPLE_WITH_DIFFICULTY)
        assert "[Difficulty: easy]" in content

    def test_build_user_content_excludes_difficulty(self) -> None:
        """_build_user_content() omits difficulty when include_difficulty=False."""
        tpl = ChatTemplate(include_difficulty=False)
        content = tpl._build_user_content(EXAMPLE_WITH_DIFFICULTY)
        assert "Difficulty" not in content

    def test_to_messages(self) -> None:
        """to_messages() returns list of dicts each with a 'messages' key."""
        tpl = ChatTemplate()
        result = tpl.to_messages([EXAMPLE])
        assert len(result) == 1
        assert "messages" in result[0]
        msgs = result[0]["messages"]
        assert isinstance(msgs, list)
        assert all("role" in m and "content" in m for m in msgs)

    def test_to_sharegpt(self) -> None:
        """to_sharegpt() returns list of dicts each with a 'conversations' key."""
        tpl = ChatTemplate()
        result = tpl.to_sharegpt([EXAMPLE])
        assert len(result) == 1
        assert "conversations" in result[0]
        turns = result[0]["conversations"]
        roles = [t["from"] for t in turns]
        assert "human" in roles
        assert "gpt" in roles

    def test_to_prompt_completion(self) -> None:
        """to_prompt_completion() returns dicts with 'prompt' and 'completion' keys."""
        tpl = ChatTemplate()
        result = tpl.to_prompt_completion([EXAMPLE])
        assert len(result) == 1
        assert "prompt" in result[0]
        assert "completion" in result[0]
        assert result[0]["completion"] == EXAMPLE["sql"]


# ======================================================================
# Model Families
# ======================================================================


def _make_conversation() -> Conversation:
    """Helper: build a simple 3-turn conversation."""
    conv = Conversation()
    conv.add("system", "You are helpful.")
    conv.add("user", "What is SQL?")
    conv.add("assistant", "Structured Query Language.")
    return conv


class TestLlamaFamily:
    """Tests for LlamaFamily.apply_template()."""

    def test_special_tokens(self) -> None:
        """Output contains Llama 3 special tokens."""
        family = LlamaFamily()
        text = family.apply_template(_make_conversation())
        assert "<|begin_of_text|>" in text
        assert "<|start_header_id|>" in text
        assert "<|end_header_id|>" in text
        assert "<|eot_id|>" in text

    def test_content_present(self) -> None:
        """Output contains all turn contents."""
        family = LlamaFamily()
        text = family.apply_template(_make_conversation())
        assert "You are helpful." in text
        assert "What is SQL?" in text
        assert "Structured Query Language." in text


class TestMistralFamily:
    """Tests for MistralFamily.apply_template()."""

    def test_special_tokens(self) -> None:
        """Output contains Mistral special tokens."""
        family = MistralFamily()
        text = family.apply_template(_make_conversation())
        assert "[INST]" in text
        assert "[/INST]" in text
        assert text.startswith("<s>")
        assert "</s>" in text

    def test_content_present(self) -> None:
        """Output contains all turn contents."""
        family = MistralFamily()
        text = family.apply_template(_make_conversation())
        assert "You are helpful." in text
        assert "What is SQL?" in text
        assert "Structured Query Language." in text


class TestPhiFamily:
    """Tests for PhiFamily.apply_template()."""

    def test_special_tokens(self) -> None:
        """Output contains Phi-3 special tokens."""
        family = PhiFamily()
        text = family.apply_template(_make_conversation())
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert "<|system|>" in text
        assert "<|end|>" in text

    def test_content_present(self) -> None:
        """Output contains all turn contents."""
        family = PhiFamily()
        text = family.apply_template(_make_conversation())
        assert "You are helpful." in text
        assert "What is SQL?" in text
        assert "Structured Query Language." in text


class TestFamilyFactory:
    """Tests for get_family() and list_families()."""

    def test_get_family_llama(self) -> None:
        """get_family('llama') returns a LlamaFamily instance."""
        family = get_family("llama")
        assert isinstance(family, LlamaFamily)

    def test_get_family_mistral(self) -> None:
        """get_family('mistral') returns a MistralFamily instance."""
        family = get_family("mistral")
        assert isinstance(family, MistralFamily)

    def test_get_family_phi(self) -> None:
        """get_family('phi') returns a PhiFamily instance."""
        family = get_family("phi")
        assert isinstance(family, PhiFamily)

    def test_get_family_unknown_raises(self) -> None:
        """get_family('unknown') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model family"):
            get_family("unknown")

    def test_list_families_returns_all(self) -> None:
        """list_families() returns all canonical family names."""
        families = list_families()
        assert "llama" in families
        assert "mistral" in families
        assert "phi" in families
        # Should be a list of strings.
        assert all(isinstance(f, str) for f in families)
