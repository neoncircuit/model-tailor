"""Format module -- converts curated data into instruction-tuning format.

Submodules
----------
templates
    Chat/conversation template builder (``ChatTemplate``).
families
    Model-family adapters for Llama, Mistral, Phi (``get_family``).
tokenize
    Token-length validation and truncation helpers (``TokenStats``).
"""

from src.format.families import (
    LlamaFamily,
    MistralFamily,
    ModelFamily,
    PhiFamily,
    get_family,
    list_families,
)
from src.format.templates import (
    DEFAULT_SQL_SYSTEM_PROMPT,
    DETAILED_SQL_SYSTEM_PROMPT,
    ChatTemplate,
    Conversation,
    Turn,
)
from src.format.tokenize import (
    TokenStats,
    ValidationResult,
    truncate_or_split,
    validate_max_length,
)

__all__ = [
    # templates
    "ChatTemplate",
    "Conversation",
    "Turn",
    "DEFAULT_SQL_SYSTEM_PROMPT",
    "DETAILED_SQL_SYSTEM_PROMPT",
    # families
    "ModelFamily",
    "LlamaFamily",
    "MistralFamily",
    "PhiFamily",
    "get_family",
    "list_families",
    # tokenize
    "TokenStats",
    "ValidationResult",
    "validate_max_length",
    "truncate_or_split",
]
