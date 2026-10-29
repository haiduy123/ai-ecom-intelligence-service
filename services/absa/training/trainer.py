from __future__ import annotations

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

def create_trainer(
    model,
    tokenizer,
    dataset,
    output_dir,
    num_train_epochs,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    warmup_steps,
    weight_decay,
    logging_steps,
    eval_steps,
    save_steps,
    save_total_limit,
    seed,
):
    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,

        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,

        gradient_accumulation_steps=gradient_accumulation_steps,

        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        logging_steps=logging_steps,

        eval_strategy="steps",
        eval_steps=eval_steps,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=(
            True if "cuda" in model.device.type else False
        ),
        fp16=(
            True 
            if (
                "cuda" in model.device.type 
                and not __import__("torch").cuda.is_bf16_supported()
            )
            else False
        ),

        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        report_to="none",

        seed=seed,
        
        remove_unused_columns=False
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    return trainer