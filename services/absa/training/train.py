from __future__ import annotations

import json
import sys

import torch

from peft import get_peft_model, prepare_model_for_kbit_training
from matplotlib import pyplot as plt

# ============================================================
# PROJECT ROOT
# ============================================================

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ============================================================
# IMPORTS
# ============================================================

from services.absa.training.config import (
    TrainingConfig
)

from services.absa.training.dataset import (
    load_sft_dataset,
    create_tokenizer,
    prepare_dataset
)

from services.absa.training.model import (
    load_model,
    create_lora_config
)

from services.absa.training.trainer import (
    create_trainer
)

from services.absa.training.callbacks import (
    TrainingProgessCallback
)

# ============================================================
# ENVIRONMENT
# ============================================================

def check_environment():
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    print(f"Pytorch: {torch.__version__}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for this "
            "training configuration."
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    if torch.cuda.is_bf16_supported():
        print("BF16 supported: True")
    else:
        print("BF16 supported: False")

def save_training_history_and_plot(trainer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_loss_history.json"

    existing_train_losses = []
    exitsing_val_losses = []

    if json_path.exists():
        try: 
            with json_path.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_train_losses = old_data.get("train_losses", [])
                exitsing_val_losses = old_data.get("validation_losses", [])

        except Exception as e:
            print("Checked old file fail: {e}")
    
    history = trainer.state.log_history
    new_train_losses = []
    new_val_losses = []

    for entry in history:
        if "loss" in entry and "step" in entry:
            new_train_losses.append({
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
                "train_loss": entry.get("loss")
            })

        if "eval_loss" in entry and "step" in entry:
            new_val_losses.append({
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
                "eval_loss": entry.get("eval_loss")
            })

    def merge_losses(old_list, new_list):
        seen_steps = {item["step"] for item in old_list}
        merged = list(old_list)
        for item in new_list:
            if item["step"] not in seen_steps:
                merged.append(item)
                seen_steps.add(item["step"])
        
        return sorted(merged, key=lambda x: x['step'])
    
    
    train_losses = merge_losses(existing_train_losses, new_train_losses)
    val_losses = merge_losses(exitsing_val_losses, new_val_losses)

    report = {
        "train_losses": train_losses,
        "validation": val_losses
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("Updated train/eval loss")

    try:
        train_steps = [x["step"] for x in train_losses if "step" in x and "train_loss" in x]
        t_loss = [x["train_loss"] for x in train_losses if "step" in x and "train_loss" in x]

        val_steps = [x["step"] for x in val_losses if "step" in x and "eval_loss" in x]
        v_loss = [x["eval_loss"] for x in val_losses if "step" in x and "eval_loss" in x]

        if train_steps or val_steps:
            plt.figure(figsize=(10, 6))
            
            if train_steps:
                plt.plot(train_steps, t_loss, label="Train Loss", color="royalblue", marker="o", linestyle="-", linewidth=2)
            
            if val_steps:
                plt.plot(val_steps, v_loss, label="Validation Loss", color="darkorange", marker="s", linestyle="--", linewidth=2)

            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Loss Value", fontsize=12)
            plt.title("Biểu đồ Train Loss và Validation Loss (Tiếp tục cập nhật)", fontsize=14, fontweight="bold")
            plt.legend(fontsize=11)
            plt.grid(True, linestyle=":", alpha=0.7)

            output_img = output_dir / "loss_curve.png"
            plt.savefig(output_img, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Updated graph at: {output_img}")
    except ImportError:
        print(f"Error when create graph: {e}")

# ============================================================
# MAIN
# ============================================================

def main():
    config = TrainingConfig()
    check_environment()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------
    
    config.output_dir.mkdir(parents=True, exist_ok=True)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.final_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(f"Model: {config.model_name}")

    print(f"Train: {config.train_file}")

    print(f"Validation: {config.validation_file}")

    print(
        f"Epochs: "
        f"{config.num_train_epochs}"
    )

    print(
        f"Batch size: "
        f"{config.per_device_train_batch_size}"
    )

    print(
        f"Gradient accumulation steps: "
        f"{config.gradient_accumulation_steps}"
    )

    print(
        f"LoRA rank: "
        f"{config.lora_r}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("=" * 70)
    print("Loading tokenizer...")
    print("=" * 70)

    tokenizer = create_tokenizer(
        config.model_name
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)

    dataset = load_sft_dataset(
        config.train_file,
        config.validation_file
    )

    print(dataset)

    dataset = prepare_dataset(
        dataset,
        tokenizer
    )

    for split in ["train", "validation"]:
        print(f"\n===== {split.upper()} =====")

        zero_label_examples = 0
        min_valid_labels = float("inf")
        max_valid_labels = 0

        for i, example in enumerate(dataset[split]):
            valid_labels = sum(
                token != -100
                for token in example["labels"]
            )

            min_valid_labels = min(
                min_valid_labels,
                valid_labels,
            )
            max_valid_labels = max(
                max_valid_labels,
                valid_labels,
            )

            if valid_labels == 0:
                zero_label_examples += 1
                print(
                    f"BAD example {i}: "
                    f"sequence_length={len(example['input_ids'])}"
                )

        print(f"zero-label examples: {zero_label_examples}")
        print(f"min valid labels:    {min_valid_labels}")
        print(f"max valid labels:    {max_valid_labels}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading model...")
    print("=" * 70)

    model = load_model(
        model_name=config.model_name,
        use_4bit=config.use_4bit,
    )

    # Required for stable QLoRA training with gradient checkpointing
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_config = create_lora_config()

    model = get_peft_model(
        model,
        lora_config,
    )
    
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=config.checkpoint_dir,
        num_train_epochs=(
            config.num_train_epochs
        ),
        learning_rate=(
            config.learning_rate
        ),
        train_batch_size=(
            config.per_device_train_batch_size
        ),
        eval_batch_size=(
            config.per_device_eval_batch_size
        ),
        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),
        warmup_steps=(
            config.warmup_steps
        ),
        weight_decay=(
            config.weight_decay
        ),
        logging_steps=(
            config.logging_steps
        ),
        eval_steps=(
            config.eval_steps
        ),
        save_steps=(
            config.save_steps
        ),
        save_total_limit=(
            config.save_total_limit
        ),
        seed=config.seed,
    )

    trainer.add_callback(
        TrainingProgessCallback()
    )

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    resume_checkpoint = None

    checkpoints = sorted(
        config.checkpoint_dir.glob("checkpoint-*")
    )

    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])

        print()
        print(
            f"Resuming from checkpoint: "
            f"\n{resume_checkpoint}"
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    # --------------------------------------------------------
    # Save Loss History and Plot Graph
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("Saving loss history and generating loss curve...")
    print("=" * 70)
    
    save_training_history_and_plot(trainer, config.output_dir)

    # --------------------------------------------------------
    # Save final adapter
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Saving final LoRA adapter...")
    print("=" * 70)

    trainer.save_model(
        str(config.final_dir)
    )

    tokenizer.save_pretrained(str(config.final_dir))

    # --------------------------------------------------------
    # Final evaluation on validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluating final model on evaluation set...")
    print("=" * 70)

    metrics = trainer.evaluate()

    print()
    print("Validation metrics")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print("=" * 70)
    print("Training complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()

