from typing import Callable

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from peft import PeftModel
from tqdm import tqdm

from inference import load_model, run_inference
from data import sample_data
from aggregators import all_or_nothing, take_max, take_mean, take_min

N_TEST = 10_000


def add_predictions(
    df: pd.DataFrame,
    aggregator_list: list[Callable[[torch.Tensor], float]],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer
) -> None:
    predictions = {aggregator.__name__: [] for aggregator in aggregator_list}

    for generation in tqdm(df["generation"]):
        probs = run_inference(generation, model, tokenizer)

        for aggregator in aggregator_list:
            predictions[aggregator.__name__].append(aggregator(probs))
    
    for aggregator in aggregator_list:
        df[f"predictions_{aggregator.__name__}"] = predictions[aggregator.__name__]


def save_predictions(
    df: pd.DataFrame,
    path: str = f"./analysis/predictions_{N_TEST}.csv",
) -> None:
    df = df.drop("generation", axis=1)
    df.to_csv(path, index=False)


def main():
    df = sample_data(N_TEST)
    model, tokenizer = load_model()
    aggregator_list = [
        all_or_nothing, take_max, take_mean, take_min
    ]

    add_predictions(df, aggregator_list, model, tokenizer)
    save_predictions(df)


if __name__ == '__main__':
    main()