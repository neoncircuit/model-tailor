"""Standalone gate + repair benchmark for text-to-SQL models.

Loads a test JSONL, generates SQL predictions with a teacher model (or a
local model), runs the deterministic execution gate, optionally repairs
failures, and logs the resulting metrics to MLFlow.

Usage:
    python scripts/run_gate_benchmark.py \
        --test-file data/curated/test.jsonl \
        --schema-file tasks/sql_generation/schemas.sql \
        --use-repair \
        --mlflow

This script is intentionally lightweight: it does not require a local GPU
because it defaults to a teacher model for predictions.  Pass ``--model-path``
to evaluate a local HuggingFace model instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.evaluate.benchmark import BenchmarkRunner, _build_prompt
from src.evaluate.gate import ExecutionGate, GateMetrics
from src.generate.repair import SQLRepairer
from src.llm.client import Message, TeacherClient
from src.train.monitor import log_gate_metrics, setup_mlflow

logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of records, one per line.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _records_with_nl(records: list[dict]) -> list[dict]:
    """Ensure every record has a ``natural_language`` key.

    Accepts either ``"nl"`` or ``"natural_language"`` as the question field.

    Args:
        records: Raw test records.

    Returns:
        Records with a guaranteed ``natural_language`` key.
    """
    normalized: list[dict] = []
    for rec in records:
        rec = dict(rec)
        if "natural_language" not in rec:
            rec["natural_language"] = rec.get("nl", "")
        normalized.append(rec)
    return normalized


def _slugify(value: str) -> str:
    """Convert a model name into a filesystem-safe slug.

    Args:
        value: Raw model name or path.

    Returns:
        Lowercase slug with non-alphanumeric runs replaced by dashes.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


def _teacher_model_name(config_path: str) -> str:
    """Read the teacher model name from a base config file.

    Args:
        config_path: Path to ``config/base.yaml``.

    Returns:
        Teacher model identifier, or ``"teacher"`` when unavailable.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("teacher", {}).get("model", "teacher"))
    except Exception:  # pragma: no cover - config unreadable
        return "teacher"


def _default_output_path(model_name: str) -> Path:
    """Build the default JSON report path for a benchmark run.

    The report is written to ``tasks/sql_generation/results/`` under the
    project root (the parent of this script's directory).

    Args:
        model_name: Model identifier used in the filename.

    Returns:
        Path of the form
        ``tasks/sql_generation/results/<model_slug>_YYYYMMDD_HHMMSS.json``.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "tasks" / "sql_generation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{_slugify(model_name)}_{timestamp}.json"


def _predict_with_teacher(
    client: TeacherClient,
    records: list[dict],
    temperature: float = 0.0,
) -> list[str]:
    """Generate SQL predictions using a teacher model.

    Args:
        client: Initialized teacher client.
        records: Test records with ``natural_language`` or ``nl`` keys.
        temperature: Sampling temperature.  0.0 is best for deterministic
            evaluation.

    Returns:
        List of predicted SQL strings in the same order as *records*.
    """
    prompts: list[list[Message]] = []
    for rec in records:
        nl = rec.get("natural_language") or rec.get("nl", "")
        prompts.append([Message(role="user", content=_build_prompt(nl))])

    return client.complete_batch(prompts, temperature=temperature)


def _predict_with_model(model_path: str, records: list[dict]) -> list[str]:
    """Generate SQL predictions using a local HuggingFace model.

    Args:
        model_path: Path or HuggingFace ID of the model.
        records: Test records.

    Returns:
        List of predicted SQL strings.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    runner = BenchmarkRunner(batch_size=8, max_new_tokens=256)
    # BenchmarkRunner.run expects a PreTrainedModel and tokenizer; the
    # predictions are then available on the returned object.
    results = runner.run(model, tokenizer, records)
    return results.predictions


def _run_gate_and_repair(
    gate: ExecutionGate,
    records: list[dict],
    predictions: list[str],
    use_repair: bool,
    repairer: SQLRepairer | None,
) -> tuple[GateMetrics, dict[str, float]]:
    """Execute the gate and optional repair loop.

    Args:
        gate: Initialized execution gate.
        records: Test records with ``natural_language`` and ``sql`` keys.
        predictions: Model-generated SQL strings.
        use_repair: Whether to attempt repairs on failing predictions.
        repairer: Optional repairer.  Required when ``use_repair`` is True.

    Returns:
        Tuple of ``(gate_metrics, repair_summary)``.  The repair summary is
        empty when ``use_repair`` is False.
    """
    gate_records = [
        {
            "natural_language": rec.get("natural_language", ""),
            "sql": pred,
            "target_sql": rec.get("sql", ""),
        }
        for rec, pred in zip(records, predictions)
    ]

    initial_results = gate.check_batch(gate_records)
    gate_metrics = gate.compute_metrics(initial_results)

    repair_summary: dict[str, float] = {}
    if use_repair:
        if repairer is None:
            raise ValueError("use_repair requires a SQLRepairer")

        repaired_passed = 0
        repair_attempts = 0
        repair_successes = 0

        for record, initial_result in zip(gate_records, initial_results):
            if initial_result.passed:
                continue
            repair_attempts += 1
            result = repairer.repair(record)
            if result.final_result.passed:
                repaired_passed += 1
                repair_successes += 1

        total = len(gate_records)
        final_passed = gate_metrics.passed_first_try + repaired_passed
        gate_metrics.final_passed = final_passed
        gate_metrics.final_failed = total - final_passed
        gate_metrics.repair_attempts = repair_attempts
        gate_metrics.repair_success = repair_successes
        gate_metrics.repair_failed = repair_attempts - repair_successes

        repair_summary = {
            "repair/attempts": float(repair_attempts),
            "repair/success": float(repair_successes),
            "repair/success_rate": (
                repair_successes / repair_attempts if repair_attempts > 0 else 0.0
            ),
            "repair/writer_plus_fixer_pass_rate": (
                final_passed / total if total > 0 else 0.0
            ),
        }

    return gate_metrics, repair_summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the gate benchmark.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Run an execution-gate benchmark on a text-to-SQL test set."
    )
    parser.add_argument("--test-file", required=True, help="Path to test JSONL file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schema-file", help="Path to DDL schema SQL file.")
    group.add_argument("--db-path", help="Path to an existing SQLite database.")
    parser.add_argument(
        "--model-path",
        help="Path or HuggingFace ID of a local model.  If omitted, the teacher "
        "model from ``--teacher-config`` is used.",
    )
    parser.add_argument(
        "--teacher-config",
        default="config/base.yaml",
        help="Base config with teacher model settings (default: config/base.yaml).",
    )
    parser.add_argument(
        "--use-repair",
        action="store_true",
        help="Repair failing predictions using the teacher model.",
    )
    parser.add_argument(
        "--repair-max-attempts",
        type=int,
        default=2,
        help="Maximum repair attempts per failing query (default: 2).",
    )
    parser.add_argument(
        "--repair-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for repair prompts (default: 0.3).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit the benchmark to the first N examples.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log metrics to MLFlow.",
    )
    parser.add_argument(
        "--mlflow-config",
        default="config/base.yaml",
        help="Config file for MLFlow tracking (default: config/base.yaml).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write a JSON report. Defaults to "
        "tasks/sql_generation/results/<model>_<timestamp>.json.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    records = _load_jsonl(args.test_file)
    records = _records_with_nl(records)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    logger.info("Loaded %d test records", len(records))

    start_time = time.perf_counter()

    if args.model_path:
        model_name = args.model_path
        predictions = _predict_with_model(args.model_path, records)
    else:
        model_name = _teacher_model_name(args.teacher_config)
        client = TeacherClient(config_path=args.teacher_config)
        predictions = _predict_with_teacher(client, records)

    gate = ExecutionGate(
        schema_sql_path=args.schema_file,
        db_path=args.db_path,
    )

    repairer: SQLRepairer | None = None
    if args.use_repair:
        client = TeacherClient(config_path=args.teacher_config)
        repairer = SQLRepairer(
            client=client,
            gate=gate,
            max_attempts=args.repair_max_attempts,
            temperature=args.repair_temperature,
        )

    gate_metrics, repair_summary = _run_gate_and_repair(
        gate, records, predictions, args.use_repair, repairer
    )

    elapsed_seconds = time.perf_counter() - start_time

    metrics = gate_metrics.to_flat_dict()
    metrics.update(repair_summary)

    logger.info("Benchmark metrics: %s", metrics)

    if args.mlflow:
        setup_mlflow(config_path=args.mlflow_config)
        log_gate_metrics(metrics)

    num_examples = len(records)
    report = {
        "model": model_name,
        "num_examples": num_examples,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_examples_per_sec": (
            round(num_examples / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0
        ),
        "overall_metrics": metrics,
        "gate_metrics": gate_metrics.to_flat_dict(),
        "repair_metrics": repair_summary,
    }
    output_path = Path(args.output) if args.output else _default_output_path(model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", output_path)

    gate.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
