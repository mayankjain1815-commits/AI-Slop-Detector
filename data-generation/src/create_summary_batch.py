import argparse
from functools import cache
from typing import Any

from utils import (
    DEFAULT_MODEL,
    create_batch,
    get_client,
    prepare_batch_file,
    upload_batch_file,
)


@cache
def get_summary_system_prompt() -> str:
    prompt = "Extract the key points from the paragraph given by the user."
    prompt += " Focus on the main facts, concepts, and relationships."
    prompt += " Present your summary as a concise list of key points."
    prompt += "\n\nGive just a bullet list of the key points — do NOT include other text in your response."

    return prompt


def get_summary_user_prompt(datum: dict[str, Any]) -> str:
    prompt = f'Article title: "{datum["page_title"]}".\n\n'
    prompt += f"Paragraph:\n{datum['text']}"

    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI batch for summarization."
    )

    parser.add_argument(
        "scrape_file_path",
        type=str,
        help="Path to input jsonl file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines to process from input file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    scrape_file_path: str = args.scrape_file_path
    model: str = args.model
    max_lines: int | None = args.max_lines
    task: str = "summarize"

    batch_file_path = f"./tmp/{task}_{max_lines}_{model}.jsonl"

    prepare_batch_file(
        scrape_file_path,
        batch_file_path,
        get_summary_system_prompt,
        get_summary_user_prompt,
        model,
        max_lines=max_lines,
    )

    batch_file = upload_batch_file(batch_file_path, client)
    _ = create_batch(batch_file, task, client)


if __name__ == "__main__":
    main()
