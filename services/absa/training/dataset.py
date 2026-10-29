from __future__ import annotations

from datasets import load_dataset
from transformers import AutoTokenizer


def load_sft_dataset(train_file, validation_file):
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
    )

    return dataset


def create_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def format_chat_example(example, tokenizer):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if not isinstance(text, str):
        raise TypeError(
            f"Expected chat template output to be str, "
            f"got {type(text)}"
        )

    return {
        "text": text,
        "review_id": example.get("review_id"),
    }


def tokenize_example(example, tokenizer, max_length=1024):
    text = example["text"]

    if not isinstance(text, str):
        raise TypeError(
            f"Expected text to be str, got {type(text)}"
        )
    
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # Start by ignoring every token
    encoded["labels"] = [-100] * len(encoded["input_ids"])

    # Find the assistant response
    assistant_marker = "<|im_start|>assistant\n"

    assistant_start = text.find(assistant_marker)

    if assistant_start == -1:
        raise ValueError("Assistant message not found")

    # Tokenize everything before assistant response
    prefix = text[:assistant_start]

    prefix_tokens = tokenizer(
        prefix,
        add_special_tokens=False,
    )["input_ids"]

    prefix_length = len(prefix_tokens)

    # Only assistant tokens contribute to loss
    encoded["labels"][prefix_length:] = encoded["input_ids"][prefix_length:]

    return encoded


def prepare_dataset(
    dataset,
    tokenizer,
    max_length=2048,
):
    # --------------------------------------------------
    # 1. Convert messages -> text
    # --------------------------------------------------

    dataset = dataset.map(
        lambda example: format_chat_example(
            example,
            tokenizer,
        ),
        desc="Formatting chat examples",
    )

    # --------------------------------------------------
    # 2. Convert text -> tokens
    # --------------------------------------------------

    dataset = dataset.map(
        lambda example: tokenize_example(
            example,
            tokenizer,
            max_length=max_length,
        ),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    return dataset