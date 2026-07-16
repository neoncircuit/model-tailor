"""Unified client for teacher model APIs (OpenAI, Anthropic, Ollama, Gemini, GLM).

Used across the pipeline for:
- Synthetic data generation (generate/)
- Quality scoring (curate/)
- LLM-as-judge evaluation (evaluate/)
"""

import os
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class TeacherClient:
    """Calls a teacher LLM (GPT-4, Claude, Gemini, GLM, Ollama) to generate
    training data or evaluate outputs."""

    def __init__(
        self, provider: str = None, model: str = None, config_path: str = "config/base.yaml"
    ) -> None:
        """Initialize the teacher client from a YAML config file.

        Args:
            provider: LLM provider name ('openai', 'anthropic', 'gemini', 'glm',
                or 'ollama'). Overrides the value in the config file when
                provided.
            model: Model identifier (e.g. 'gpt-4'). Overrides the config value
                when provided.
            config_path: Path to the YAML configuration file containing teacher
                model settings.
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["teacher"]

        self.provider = provider or cfg["provider"]
        self.model = model or cfg["model"]
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 2048)
        self._ollama_base_url = cfg.get(
            "ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self._client = self._init_client()

    def _init_client(self) -> object:
        """Create and return the underlying API client for the configured provider.

        Returns:
            An initialized API client (OpenAI, Anthropic, Gemini, or GLM instance).

        Raises:
            ValueError: When the provider is not one of 'openai', 'anthropic',
                'gemini', 'glm', or 'ollama'.
        """
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            # Explicitly use real Anthropic API, ignore ANTHROPIC_BASE_URL env var
            return Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com"
            )
        elif self.provider == "gemini":
            from google import genai

            return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self.provider == "glm":
            from openai import OpenAI

            # GLM (Z.ai) uses OpenAI-compatible API
            return OpenAI(
                api_key=os.getenv("GLM_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
        elif self.provider == "ollama":
            from openai import OpenAI

            return OpenAI(base_url=f"{self._ollama_base_url}/v1", api_key="ollama")
        else:
            supported = "openai, anthropic, gemini, glm, ollama"
            raise ValueError(f"Unknown provider: {self.provider}. Supported: {supported}")

    def complete(
        self, messages: list[Message], temperature: float = None, max_tokens: int = None
    ) -> str:
        """Send messages to the teacher model and return the response text.

        Args:
            messages: Conversation messages to send to the model.
            temperature: Sampling temperature override. Uses the instance
                default when not provided.
            max_tokens: Maximum tokens override. Uses the instance default
                when not provided.

        Returns:
            The text content of the model's response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        if self.provider in ("openai", "ollama", "glm"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            from google.genai import types

            contents = []
            system_instruction = None
            for m in messages:
                if m.role == "system":
                    system_instruction = m.content
                else:
                    role = "model" if m.role == "assistant" else "user"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=m.content)],
                        )
                    )

            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=tokens,
                system_instruction=system_instruction,
            )

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text

        elif self.provider == "anthropic":
            system = None
            chat_messages = []
            for m in messages:
                if m.role == "system":
                    system = m.content
                else:
                    chat_messages.append({"role": m.role, "content": m.content})

            kwargs = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            if system:
                kwargs["system"] = system

            response = self._client.messages.create(**kwargs)
            return response.content[0].text

    def complete_batch(self, prompts: list[list[Message]], temperature: float = None) -> list[str]:
        """Process multiple prompts sequentially.

        For parallel execution, see generate/batch.py.

        Args:
            prompts: A list of message lists, one per prompt.
            temperature: Sampling temperature override applied to every call.

        Returns:
            A list of response strings, one per prompt, in the same order.
        """
        return [self.complete(msgs, temperature=temperature) for msgs in prompts]
