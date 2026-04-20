"""FastAPI inference server for fine-tuned models.

Supports two model backends:
* **GGUF** models loaded via ``llama-cpp-python`` (lightweight, CPU-friendly).
* **HuggingFace** models loaded via ``transformers`` (GPU, full-precision or
  quantised via bitsandbytes).

Usage::

    # Start with a GGUF model
    uvicorn src.deploy.serve:app --host 0.0.0.0 --port 8000

    # Or programmatically:
    from src.deploy.serve import create_app
    app = create_app("models/my-model.gguf", model_type="gguf")
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body for the ``/generate`` endpoint."""

    prompt: str = Field(..., min_length=1, description="The input prompt for generation.")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    stop: list[str] | None = Field(None, description="Optional stop sequences.")


class GenerateResponse(BaseModel):
    """Response from the ``/generate`` endpoint."""

    response: str
    tokens_generated: int
    elapsed_seconds: float


class GenerateSQLRequest(BaseModel):
    """Body for the ``/generate_sql`` endpoint."""

    question: str = Field(..., min_length=1, description="Natural language question.")
    schema: str = Field(..., min_length=1, description="Database schema (DDL or description).")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate.")


class GenerateSQLResponse(BaseModel):
    """Response from the ``/generate_sql`` endpoint."""

    sql: str
    tokens_generated: int
    elapsed_seconds: float


class HealthResponse(BaseModel):
    """Response from the ``/health`` endpoint."""

    status: str
    model_path: str
    model_type: str


# ---------------------------------------------------------------------------
# Model backend abstraction
# ---------------------------------------------------------------------------

ModelType = Literal["gguf", "hf"]

# Module-level model holder so the lifespan hook and route handlers can share
# state without globals scattered around.


class _ModelState:
    """Mutable container for the loaded model and its metadata."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_path: str = ""
        self.model_type: ModelType = "gguf"

    @property
    def loaded(self) -> bool:
        return self.model is not None


_state = _ModelState()


def _load_gguf(model_path: str) -> None:
    """Load a GGUF model via llama-cpp-python.

    Args:
        model_path: Path to the GGUF model file.
    """
    from llama_cpp import Llama

    logger.info("Loading GGUF model from %s", model_path)
    _state.model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,  # offload all layers to GPU when available
        verbose=False,
    )
    _state.model_type = "gguf"
    _state.model_path = model_path
    logger.info("GGUF model loaded successfully")


def _load_hf(model_path: str) -> None:
    """Load a HuggingFace model via transformers.

    Args:
        model_path: Path to the HuggingFace model directory containing
            safetensors and tokenizer files.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HuggingFace model from %s", model_path)

    _state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _state.tokenizer.pad_token is None:
        _state.tokenizer.pad_token = _state.tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _state.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        _state.model = _state.model.to(device)

    _state.model.eval()
    _state.model_type = "hf"
    _state.model_path = model_path
    logger.info("HuggingFace model loaded on %s", device)


_LOADERS: dict[ModelType, Callable[[str], None]] = {
    "gguf": _load_gguf,
    "hf": _load_hf,
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _generate_gguf(
    prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
) -> tuple[str, int]:
    """Generate text with the GGUF backend.

    Args:
        prompt: The input prompt for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        stop: Optional list of stop sequences.

    Returns:
        A tuple of the generated text and the number of tokens produced.
    """
    result = _state.model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [],
        echo=False,
    )
    text = result["choices"][0]["text"]
    n_tokens = result["usage"]["completion_tokens"]
    return text.strip(), n_tokens


def _generate_hf(
    prompt: str, max_tokens: int, temperature: float, stop: list[str] | None
) -> tuple[str, int]:
    """Generate text with the HuggingFace backend.

    Args:
        prompt: The input prompt for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        stop: Optional list of stop sequences.

    Returns:
        A tuple of the generated text and the number of tokens produced.
    """
    import torch

    tokenizer = _state.tokenizer
    model = _state.model

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # Collect stop token ids when stop sequences are provided.
    if stop:
        stop_ids = []
        for seq in stop:
            encoded = tokenizer.encode(seq, add_special_tokens=False)
            if encoded:
                stop_ids.append(encoded[0])
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip(), len(new_ids)


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(
    model_path: str | Path | None = None,
    model_type: ModelType | None = None,
) -> FastAPI:
    """Create and return a configured FastAPI application.

    Args:
        model_path: Path to the model file (GGUF) or directory (HF
            safetensors). Falls back to the ``MODEL_PATH`` environment
            variable.
        model_type: Backend to use. Auto-detected from the file extension
            when None: files ending in ``.gguf`` use the GGUF backend,
            everything else uses HF. Also falls back to the ``MODEL_TYPE``
            environment variable.

    Returns:
        A configured FastAPI application with model inference endpoints.
    """

    resolved_path = str(model_path) if model_path else os.getenv("MODEL_PATH", "")
    resolved_type = model_type or os.getenv("MODEL_TYPE", "")

    if not resolved_type:
        resolved_type = "gguf" if resolved_path.endswith(".gguf") else "hf"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load model.
        if resolved_path:
            loader = _LOADERS.get(resolved_type)
            if loader is None:
                raise ValueError(f"Unknown model_type: {resolved_type!r}")
            loader(resolved_path)
        else:
            logger.warning(
                "No MODEL_PATH provided -- server will start but /generate "
                "and /generate_sql will return 503 until a model is loaded."
            )
        yield
        # Shutdown: release model.
        _state.model = None
        _state.tokenizer = None

    application = FastAPI(
        title="model-tailor inference server",
        version="0.1.0",
        description="Serve fine-tuned models for inference.",
        lifespan=lifespan,
    )

    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:
    """Attach endpoint handlers to *application*.

    Args:
        application: The FastAPI application to register routes on.
    """

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Return the server and model status.

        Returns:
            Current health status including model path and type.
        """
        return HealthResponse(
            status="ok" if _state.loaded else "no_model",
            model_path=_state.model_path,
            model_type=_state.model_type,
        )

    @application.post("/generate", response_model=GenerateResponse, tags=["inference"])
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate text from a free-form prompt.

        Args:
            request: The generation request containing prompt and parameters.

        Returns:
            Generated text with token count and elapsed time.

        Raises:
            HTTPException: If no model is loaded (503).
        """
        if not _state.loaded:
            raise HTTPException(status_code=503, detail="No model loaded.")

        t0 = time.perf_counter()

        if _state.model_type == "gguf":
            text, n_tokens = _generate_gguf(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.stop,
            )
        else:
            text, n_tokens = _generate_hf(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.stop,
            )

        elapsed = time.perf_counter() - t0

        return GenerateResponse(
            response=text,
            tokens_generated=n_tokens,
            elapsed_seconds=round(elapsed, 3),
        )

    @application.post("/generate_sql", response_model=GenerateSQLResponse, tags=["inference"])
    async def generate_sql(request: GenerateSQLRequest) -> GenerateSQLResponse:
        """Generate a SQL query from a natural-language question and a database schema.

        Args:
            request: The SQL generation request containing a question and
                database schema.

        Returns:
            Generated SQL query with token count and elapsed time.

        Raises:
            HTTPException: If no model is loaded (503).
        """
        if not _state.loaded:
            raise HTTPException(status_code=503, detail="No model loaded.")

        # Build a structured prompt that mirrors the fine-tuning format.
        prompt = _build_sql_prompt(request.question, request.schema)

        t0 = time.perf_counter()

        if _state.model_type == "gguf":
            text, n_tokens = _generate_gguf(prompt, request.max_tokens, 0.1, [";", "\n\n"])
        else:
            text, n_tokens = _generate_hf(prompt, request.max_tokens, 0.1, [";", "\n\n"])

        elapsed = time.perf_counter() - t0

        sql = _clean_sql_output(text)

        return GenerateSQLResponse(
            sql=sql,
            tokens_generated=n_tokens,
            elapsed_seconds=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# SQL prompt helpers
# ---------------------------------------------------------------------------

_SQL_SYSTEM = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate the correct SQL query. Output ONLY the SQL query, no explanation."
)

_SQL_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "### Schema:\n{schema}\n\n### Question:\n{question}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _build_sql_prompt(question: str, schema: str) -> str:
    """Build a Llama-3 chat-formatted prompt for SQL generation.

    Args:
        question: The natural-language question to answer with SQL.
        schema: The database schema (DDL or description).

    Returns:
        A fully formatted prompt string ready for model inference.
    """
    return _SQL_TEMPLATE.format(
        system=_SQL_SYSTEM, schema=schema.strip(), question=question.strip()
    )


def _clean_sql_output(raw: str) -> str:
    """Strip markdown fences and trailing whitespace from generated SQL.

    Args:
        raw: The raw generated text that may contain markdown fences.

    Returns:
        Cleaned SQL string, guaranteed to end with a semicolon.
    """
    text = raw.strip()
    # Remove ```sql ... ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines.
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Ensure the query ends with a semicolon.
    if text and not text.endswith(";"):
        text += ";"
    return text


# ---------------------------------------------------------------------------
# Default app instance (for ``uvicorn src.deploy.serve:app``)
# ---------------------------------------------------------------------------

app = create_app()
