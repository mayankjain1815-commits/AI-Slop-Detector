import argparse

from utils import (
    await_batch,
    get_batch,
    get_client,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI batch for summarization."
    )

    parser.add_argument(
        "batch_id",
        type=str,
        help="ID of batch to check.",
    )

    parser.add_argument(
        "--check-interval",
        type=float,
        default=10.0,
        help="How often to check for updates.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    batch_id = args.batch_id
    check_interval = args.check_interval

    batch_data = get_batch(batch_id, client)
    await_batch(batch_data, check_interval, client)


if __name__ == "__main__":
    main()
