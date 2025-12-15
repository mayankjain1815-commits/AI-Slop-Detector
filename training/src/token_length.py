from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset, DatasetDict
import numpy as np

from train import get_datasets


def analyze_token_lengths(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer
):
    def tokenize_and_get_length(examples: Dataset) -> dict[str, list[int]]:
        """Tokenize texts and return their lengths."""
        tokenized = tokenizer(examples["text"], truncation=False, padding=False)
        return {"token_length": [len(ids) for ids in tokenized["input_ids"]]}  # type: ignore
    
    # Add token lengths to the dataset
    dataset_with_lengths = dataset.map(
        tokenize_and_get_length,
        batched=True,
        desc="Tokenizing and computing lengths"
    )
    
    # Compute statistics for each split
    stats = {}
    for split_name in dataset_with_lengths.keys():
        lengths = dataset_with_lengths[split_name]["token_length"]
        lengths = np.array(lengths)
        
        stats[split_name] = {
            "count": len(lengths),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "q95": float(np.percentile(lengths, 95)),
            "q99": float(np.percentile(lengths, 99)),
            "num_above_512": np.sum(lengths > 512),
            "num_above_1024": np.sum(lengths > 1024),
        }
    
    return stats, dataset_with_lengths


def main():
    raw_datasets = get_datasets()
    checkpoint = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    stats, _ = analyze_token_lengths(raw_datasets, tokenizer)
    for split_name, split_stats in stats.items():
        print(split_name)
        print("="*len(split_name))
        for stat, value in split_stats.items():
            print(f"{stat}: {value}")
        print("")


if __name__ == '__main__':
    main()