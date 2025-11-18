from functools import partial

import datasets
import evaluate
import nltk
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import remove_emojis


def get_datasets() -> DatasetDict:
    data_files = {"train": "../data/train/*.jsonl", "test": "../data/test/*.jsonl"}

    dataset = datasets.load_dataset("json", data_files=data_files)
    assert isinstance(dataset, DatasetDict)

    # Augmentations: Clean should map to clean
    train_dataset = dataset["train"]
    # test_dataset = dataset["test"]
    
    augmented_train = Dataset.from_dict({
        "slop": train_dataset["clean"],
        "clean": train_dataset["clean"]
    })    
    
    # augmented_test = Dataset.from_dict({
        # "slop": test_dataset["clean"],
        # "clean": test_dataset["clean"]
    # })

    dataset["train"] = concatenate_datasets([train_dataset, augmented_train])
    # dataset["test"] = concatenate_datasets([test_dataset, augmented_test])

    return dataset


def _preprocess_function(
    dataset: Dataset | dict,
    tokenizer: BartTokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 1024,
) -> BatchEncoding:
    """Preprocess the dataset for BART conditional generation."""
    inputs = dataset["slop"]
    targets = dataset["clean"]

    # Remove emojis
    inputs = [remove_emojis(input) for input in inputs]
    targets = [remove_emojis(target) for target in targets]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
    tokenizer: BartTokenizer,
    metric: evaluate.EvaluationModule,
) -> dict[str, float]:
    predictions, labels = eval_pred

    assert isinstance(tokenizer.pad_token_id, int)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute generation lengths
    gen_lengths = [
        len(tokenizer.encode(pred, add_special_tokens=False)) for pred in decoded_preds
    ]
    length_min = np.min(gen_lengths) if gen_lengths else 0
    length_median = np.median(gen_lengths) if gen_lengths else 0
    length_max = np.max(gen_lengths) if gen_lengths else 0

    # Format for ROUGE
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    assert result is not None

    result.update(
        {
            "gen_length_min": length_min,
            "gen_length_median": length_median,
            "gen_length_max": length_max,
            "sample_pred": decoded_preds[0] if decoded_preds else None,
        }
    )

    return result


if __name__ == "__main__":
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except:
        nltk.download("punkt_tab")

    raw_datasets = get_datasets()

    checkpoint = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    model.generation_config.early_stopping = True
    model.generation_config.num_beams = 2
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.forced_bos_token_id = 0

    metric = evaluate.load("rouge")

    preprocess_function = partial(_preprocess_function, tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    train_batch_size = 2
    gradient_accumulation_steps = 16
    eval_batch_size = 4

    training_args = Seq2SeqTrainingArguments(
        "../models/bart-base-finetuned",
        
        num_train_epochs=30,
        learning_rate=5e-5,
        weight_decay=0.01,
        
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        
        save_strategy="steps",
        save_total_limit=5,
        save_steps=128,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        
        eval_strategy="steps",
        eval_steps=128,
        predict_with_generate=True,
        generation_max_length=512,
        
        logging_strategy="steps",
        logging_steps=64,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    compute_metrics = partial(_compute_metrics, tokenizer=tokenizer, metric=metric)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore
    )

    trainer.train()
