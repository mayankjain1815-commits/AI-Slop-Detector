import glob
import json
from collections.abc import Generator


def get_data(file_path: str) -> Generator[dict[str, str], None, None]:
    with open(file_path) as file:
        for line in file:
            datum = json.loads(line)  # pyright:ignore[reportAny]
            yield datum


def main():
    file_paths = glob.glob("../data/train/*.jsonl")

    slop_lengths = []
    clean_lengths = []

    # Collect data
    for file_path in file_paths:
        data = get_data(file_path)

        for idx, datum in enumerate(data):
            slop_lengths.append(len(datum["slop"]))
            clean_lengths.append(len(datum["clean"]))

    sorted_slop_lengths = sorted(slop_lengths)
    sorted_clean_lengths = sorted(clean_lengths)

    for file_path in file_paths:
        data = get_data(file_path)

        for idx, datum in enumerate(data):
            slop_length = len(datum["slop"])
            if (
                # slop_length >= sorted_slop_lengths[-3]
                # or slop_length <= sorted_slop_lengths[3]
                "[" in datum["slop"]  # check for [placeholders]
            ):
                title = f"ABNORMAL SLOP: {file_path}, {idx + 1}"
                print(title)
                print("=" * len(title))
                print(datum["slop"])
                _ = input("")

            clean_length = len(datum["clean"])
            if (
                # clean_length >= sorted_clean_lengths[-3]
                # or clean_length <= sorted_clean_lengths[3]
                "[" in datum["clean"]  # check for [placeholders]
            ):
                title = f"ABNORMAL CLEAN: {file_path}, {idx + 1}"
                print(title)
                print("=" * len(title))
                print(datum["clean"])
                _ = input("")


if __name__ == "__main__":
    main()
