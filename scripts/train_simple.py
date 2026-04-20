#!/usr/bin/env python
"""Simple training script using standard transformers/PEFT (avoids TRL compatibility issues)."""

import jsonlines
import os
import logging
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path: str) -> Dataset:
    """Load formatted JSONL data into a Dataset."""
    data = []
    with jsonlines.open(path) as reader:
        for line in reader:
            # Handle both dict format and raw text format
            if isinstance(line, dict) and "text" in line:
                data.append({"text": line["text"]})
            elif isinstance(line, str):
                data.append({"text": line})
            elif isinstance(line, dict) and "messages" in line:
                # Skip messages format for now - use text format
                continue
    logger.info(f"Loaded {len(data)} examples from {path}")
    if len(data) == 0:
        logger.error(f"No examples loaded from {path}!")
        raise ValueError(f"Failed to load data from {path}")
    return Dataset.from_list(data)

def main():
    # Paths
    train_path = "data/formatted/train.jsonl"
    val_path = "data/formatted/val.jsonl"
    output_dir = "models/sql-llama-8b-lora"

    # Load data
    train_ds = load_data(train_path)
    val_ds = load_data(val_path)

    # Load model and tokenizer
    logger.info("Loading model...")
    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Preprocess: tokenize the text data
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids (shifted internally by model)
        result["labels"] = result["input_ids"].clone()
        # Remove the text field to avoid dimension issues
        result.pop("text", None)
        return result

    logger.info("Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    val_tokenized = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments (conservative settings to prevent OOM/BSoD)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # 2 → 1 for VRAM headroom
        per_device_eval_batch_size=1,   # 2 → 1 for VRAM headroom
        gradient_accumulation_steps=8,  # 4 → 8 to maintain effective batch size
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        bf16=True,
        gradient_checkpointing=True,  # Saves VRAM
        report_to="none",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Reduce CPU overhead
        max_grad_norm=1.0,  # Gradient clipping for stability
    )

    # Data collator - for causal LM with tokenized data
    data_collator = None  # Use default collator for tokenized data

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
