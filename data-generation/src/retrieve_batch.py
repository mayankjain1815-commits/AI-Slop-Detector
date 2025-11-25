import argparse

from utils import get_batch, get_client, retrieve_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI batch for summarization."
    )

    parser.add_argument(
        "batch_id",
        type=str,
        help="ID of batch to retrieve.",
    )

    parser.add_argument("write_path", type=str, help="Path to write result to.")

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    batch_id = args.batch_id
    write_path = args.write_path

    batch_data = get_batch(batch_id, client)

    results = retrieve_results(batch_data, client)

    with open(write_path, "w") as f:
        f.write(results)
