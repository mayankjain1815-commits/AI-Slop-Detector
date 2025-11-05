from datasets import load_dataset


if __name__ == "__main__":
    data_files = {"train": "../data/train/*.jsonl", "test": "../data/test/*.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    print(len(dataset["test"]))
