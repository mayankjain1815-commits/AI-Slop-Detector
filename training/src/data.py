import json
import os
from transformers import T5Tokenizer


def get_filepaths() -> list[str]:
    return [
        f"../data/{filename}"
        for filename in os.listdir("../data")
        if filename.endswith(".jsonl")
    ]


def add_file_data(
    filepath: str,
    slops: list[str],
    cleans: list[str],
) -> None:
    with open(filepath) as file:
        for line in file:
            datum: dict[str, str] = json.loads(line)  # pyright:ignore[reportAny]

            slops.append(datum["slop"])
            cleans.append(datum["clean"])


def load_data() -> tuple[list[str], list[str]]:
    slops: list[str] = []
    cleans: list[str] = []

    for filepath in get_filepaths():
        add_file_data(filepath, slops, cleans)

    return slops, cleans


def count_tokens_per_sequence(
    tokenizer: T5Tokenizer, inputs: list[str], outputs: list[str]
) -> tuple[list[int], list[int]]:
    input_token_counts: list[int] = []
    output_token_counts: list[int] = []

    for input_text in inputs:
        tokens = tokenizer.encode(input_text, add_special_tokens=True)
        input_token_counts.append(len(tokens))

    for output_text in outputs:
        tokens = tokenizer.encode(output_text, add_special_tokens=True)
        output_token_counts.append(len(tokens))

    return input_token_counts, output_token_counts


def print_token_statistics(token_counts: list[int], label: str = "Sequence"):
    import numpy as np

    counts_array = np.array(token_counts)

    stats = {
        "min": counts_array.min(),
        "max": counts_array.max(),
        "mean": counts_array.mean(),
        "median": np.median(counts_array),
        "std": counts_array.std(),
        "percentile_95": np.percentile(counts_array, 95),
        "percentile_99": np.percentile(counts_array, 99),
    }

    print(f"\n{label} Token Statistics:")
    print(f"  Min tokens: {stats['min']}")
    print(f"  Max tokens: {stats['max']}")
    print(f"  Mean tokens: {stats['mean']:.2f}")
    print(f"  Median tokens: {stats['median']:.2f}")
    print(f"  Std deviation: {stats['std']:.2f}")
    print(f"  95th percentile: {stats['percentile_95']:.2f}")
    print(f"  99th percentile: {stats['percentile_99']:.2f}")

    return stats


if __name__ == "__main__":
    slops, cleans = load_data()

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    in_counts, out_counts = count_tokens_per_sequence(tokenizer, slops, cleans)
    _ = print_token_statistics(in_counts, "Slops")
    _ = print_token_statistics(out_counts, "Cleans")
