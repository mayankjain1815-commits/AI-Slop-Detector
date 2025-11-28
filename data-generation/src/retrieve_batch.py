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

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    batch_id = args.batch_id

    batch_data = get_batch(batch_id, client)

    # errors
    error_file_id = batch_data.error_file_id
    if error_file_id is not None:
        file_response = client.files.content(error_file_id)

        write_path = f"./tmp/errors_{batch_id}.jsonl"
        with open(write_path, "w") as f:
            f.write(file_response.text)

    # Results
    results = retrieve_results(batch_data, client)

    write_path = f"./tmp/retrievals_{batch_id}.jsonl"
    with open(write_path, "w") as f:
        f.write(results)


if __name__ == "__main__":
    main()
