import argparse
import json
from collections.abc import Generator


def get_data(id: str) -> Generator[dict[str, str], None, None]:
    with open(f"../data/{id}.jsonl") as file:
        for line in file:
            datum = json.loads(line)  # pyright:ignore[reportAny]
            yield datum


def main():
    argparser = argparse.ArgumentParser()
    _ = argparser.add_argument("id", type=str)
    args = argparser.parse_args()
    id = str(args.id)  # pyright:ignore[reportAny]

    for idx, datum in enumerate(get_data(id)):
        print(f"SLOP #{idx + 1}")
        print("=" * 12)
        print(datum["slop"])
        print("")

        print(f"CLEAN #{idx + 1}")
        print("=" * 12)
        print(datum["clean"])
        print("")

        _ = input("")


if __name__ == "__main__":
    main()
