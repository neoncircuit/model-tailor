#!/usr/bin/env python
"""Train SQL generation model with LoRA fine-tuning.

This script loads formatted training data, builds a LoRA-wrapped Llama 3.1 8B model,
and runs fine-tuning with QLoRA (4-bit quantization) on the SQL generation task.
"""

import argparse
import logging
import sys
from pathlib import Path

import jsonlines
from datasets import Dataset
from tqdm import tqdm

from src.format.families import ModelFamily
from src.format.templates import ChatTemplate
from src.train.lora import build_lora_model, get_default_config
from src.train.monitor import setup_mlflow, TrainingMonitor
from src.train.runner import TrainingRunner, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_formatted_data(path: str) -> Dataset:
    """Load formatted JSONL data into a HuggingFace Dataset.

    Args:
        path: Path to the formatted JSONL file.

    Returns:
        A HuggingFace Dataset with a 'text' column.
    """
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            # Handle both dict format and string format
            if isinstance(obj, dict):
                if "text" in obj:
                    data.append({"text": obj["text"]})
                elif "messages" in obj:
                    # Convert messages back to text format
                    template = ChatTemplate()
                    text = template.apply_template(
                        family=ModelFamily.llama,
                        messages=obj["messages"],
                    )
                    data.append({"text": text})
                else:
                    logger.warning(f"Skipping malformed entry: {obj.keys()}")
            elif isinstance(obj, str):
                data.append({"text": obj})
            else:
                logger.warning(f"Skipping unknown entry type: {type(obj)}")

    logger.info(f"Loaded {len(data)} examples from {path}")
    return Dataset.from_list(data)


def main() -> int:
    """Run training.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Train SQL generation model")
    parser.add_argument(
        "--train-data",
        default="data/formatted/train.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--val-data",
        default="data/formatted/val.jsonl",
        help="Path to validation data JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="models/sql-llama-8b-lora",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2 for RTX 3070 Ti)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Path to base config YAML",
    )

    args = parser.parse_args()

    # Verify data files exist
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        return 1

    if not Path(args.val_data).exists():
        logger.error(f"Validation data not found: {args.val_data}")
        return 1

    # Setup MLFlow (may fail gracefully if server not running)
    setup_mlflow(args.config)

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = load_formatted_data(args.train_data)
    val_ds = load_formatted_data(args.val_data)

    logger.info(f"Train examples: {len(train_ds)}")
    logger.info(f"Val examples: {len(val_ds)}")

    # Build LoRA model
    logger.info("Building LoRA model...")
    lora_cfg = get_default_config("llama", args.config)
    model, tokenizer = build_lora_model(
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        lora_cfg,
    )
    logger.info(f"Model loaded: {type(model).__name__}")

    # Create training config
    train_cfg = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
    )

    logger.info(f"Training config: epochs={args.num_epochs}, batch_size={args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create runner and train
    runner = TrainingRunner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=train_cfg,
    )

    logger.info("Starting training...")
    try:
        metrics = runner.train()
        logger.info("Training complete!")
        logger.info(f"Final metrics: {metrics}")

        # Save LoRA adapter
        output_path = runner.save(args.output_dir)
        logger.info(f"Model saved to: {output_path}")

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
