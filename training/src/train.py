from functools import partial

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    PreTrainedModel
)
from peft import LoraConfig, get_peft_model


def get_datasets() -> DatasetDict:
    data_file = "../data/wikipedia.jsonl"

    dataset = datasets.load_dataset("json", data_files=data_file)
    assert isinstance(dataset, DatasetDict)

    full_dataset = dataset["train"]
    
    human_examples = Dataset.from_dict({
        "text": full_dataset["human_text"],
        "label": [0] * len(full_dataset),
    })

    ai_examples = Dataset.from_dict({
        "text": full_dataset["ai_text"],
        "label": [1] * len(full_dataset),
    })

    combined_dataset = concatenate_datasets([human_examples, ai_examples])
    combined_dataset = combined_dataset.shuffle(seed=42)

    dataset = combined_dataset.train_test_split(test_size=0.05, seed=42)

    return dataset


def _preprocess_function(
    dataset: Dataset | dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> BatchEncoding:
    texts = dataset["text"]
    model_inputs = tokenizer(texts, max_length=max_length, truncation=True)

    model_inputs["label"] = dataset["label"]

    return model_inputs


def _compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
    metric_accuracy: evaluate.EvaluationModule,
    metric_f1: evaluate.EvaluationModule,
) -> dict[str, float]:
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)

    assert accuracy is not None and f1 is not None

    result = {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

    return result


if __name__ == "__main__":
    raw_datasets = get_datasets()

    checkpoint = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    ).to(device='cuda')

    peft_config = LoraConfig(
        r=16,
        target_modules="all-linear",
        lora_alpha=16,
        bias="none",
        lora_dropout=0.05,
        use_rslora=True,
        modules_to_save=["classifier"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    preprocess_function = partial(_preprocess_function, tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    train_batch_size = 4
    gradient_accumulation_steps = 8
    eval_batch_size = 4

    training_args = TrainingArguments(
        "../models/bert-base-classifier-peft",
        
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.1,
        
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        
        save_strategy="steps",
        save_total_limit=2,
        save_steps=64,
        metric_for_best_model="eval_accuracy",
        load_best_model_at_end=True,
        
        eval_strategy="steps",
        eval_steps=64,
        
        logging_strategy="steps",
        logging_steps=16,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    compute_metrics = partial(_compute_metrics, metric_accuracy=metric_accuracy, metric_f1=metric_f1)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
    )

    trainer.train()
    trainer.evaluate()
