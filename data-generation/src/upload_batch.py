import argparse

from utils import (
    create_batch,
    get_client,
    upload_batch_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI batch for rewriting a paragraph from its summary."
    )

    parser.add_argument(
        "batch_file_path",
        type=str,
        help="Path of batch input file.",
    )

    parser.add_argument(
        "task",
        type=str,
        help="Task.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    batch_file_path: str = args.batch_file_path
    task: str = args.task

    batch_file = upload_batch_file(batch_file_path, client)
    _ = create_batch(batch_file, task, client)


if __name__ == "__main__":
    main()
