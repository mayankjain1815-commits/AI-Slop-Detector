from functools import partial

import datasets
from datasets import DatasetDict, Dataset
from transformers import (
    T5Tokenizer, BatchEncoding,
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import evaluate
import nltk
# nltk.download('punkt_tab')
import numpy as np


def get_datasets() -> DatasetDict:
    data_files = {"train": "../data/train/*.jsonl", "test": "../data/test/*.jsonl"}

    dataset = datasets.load_dataset("json", data_files=data_files)
    assert isinstance(dataset, DatasetDict)

    return dataset


def _preprocess_function(
    dataset: Dataset | dict,
    tokenizer: T5Tokenizer,
    prefix: str = "summarize: ",
    max_slop_length: int = 1024,
    max_clean_length: int = 1024,
) -> BatchEncoding:
    inputs = [prefix + slop for slop in dataset["slop"]]
    result = tokenizer(inputs, max_length=max_slop_length, truncation=True)

    cleans = tokenizer(dataset["clean"], max_length=max_clean_length, truncation=True)

    result["labels"] = cleans["input_ids"]
    return result


def _compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
    tokenizer: T5Tokenizer,
    metric: evaluate.EvaluationModule,
) -> dict[str, float]:
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute generation lengths
    gen_lengths = [len(tokenizer.encode(pred, add_special_tokens=False)) for pred in decoded_preds]
    length_min = np.min(gen_lengths) if gen_lengths else 0
    length_median = np.median(gen_lengths) if gen_lengths else 0
    length_max = np.max(gen_lengths) if gen_lengths else 0

    # Format for ROUGE
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels,
        use_stemmer=True, use_aggregator=True
    )
    assert result is not None

    result.update({
        "gen_length_min": length_min,
        "gen_length_median": length_median,
        "gen_length_max": length_max,
        "sample_pred": decoded_preds[0] if decoded_preds else None,
    })

    return result


if __name__ == "__main__":
    raw_datasets = get_datasets()

    checkpoint = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    metric = evaluate.load('rouge')

    preprocess_function = partial(_preprocess_function, tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    batch_size = 16
    training_args = Seq2SeqTrainingArguments(
        f"checkpoints/{checkpoint.split('/')[-1]}-finetuned",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    compute_metrics = partial(_compute_metrics, tokenizer=tokenizer, metric=metric)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], # type: ignore
        data_collator=data_collator,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics # type: ignore
    )

    trainer.train()
