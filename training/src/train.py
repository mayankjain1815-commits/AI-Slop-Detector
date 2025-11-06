from functools import partial

import datasets
from datasets import DatasetDict, Dataset
from transformers import (
    T5Tokenizer, BatchEncoding,
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import evaluate
import torch
import nltk


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
    eval_pred: tuple[torch.Tensor, torch.Tensor],
    tokenizer: T5Tokenizer,
    metric: evaluate.EvaluationModule,
) -> dict[str, float]:
    predictions, labels = eval_pred
    decoded_preds: list[str] = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # -100 token is unknown
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_type_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels,
        use_stemmer=True, use_aggregator=True
    )
    assert result != None

    result = {key: 100 * value for key, value in result.items()}

    # pred_lengths = [torch.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    
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
