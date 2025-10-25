import argparse
import json
from typing import Generator


def get_data(id: str) -> Generator[dict[str, str], None, None]:
    with open(f"../data/{id}.jsonl") as file:
        for line in file:
            datum = json.loads(line)
            yield datum


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("id", type=str)
    args = argparser.parse_args()
    id = str(args.id)

    for idx, datum in enumerate(get_data(id)):
        print(f"SLOP #{idx + 1}")
        print("=" * 12)
        print(datum["slop"])
        print("")

        print(f"CLEAN #{idx + 1}")
        print("=" * 12)
        print(datum["clean"])
        input("")


if __name__ == "__main__":
    main()
