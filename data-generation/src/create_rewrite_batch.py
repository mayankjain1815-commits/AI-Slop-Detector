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
def get_rewrite_system_prompt() -> str:
    prompt = "You are writing a single paragraph for a Wikipedia article."
    prompt += " Using the key points provided by the user, write one cohesive paragraph in Wikipedia's encyclopedic style."

    prompt += "\n\nRequirements:"
    prompt += "\n- Use formal, neutral, encyclopedic tone"
    prompt += "\n- Write in third person"
    prompt += "\n- Present information objectively"
    prompt += "\n- Create smooth transitions between ideas"
    prompt += "\n- Do NOT copy phrases verbatim from the key points — rephrase naturally, using synonyms where appropriate"
    prompt += "\n- Do NOT include any markdown formatting (e.g., do NOT use *italics* and do NOT use **bold**)"

    prompt += "\n\nGive just your written paragraph — do NOT include other text in your response."

    return prompt


def get_article_titles(scrape_file_path: str) -> list[str]:
    raise NotImplementedError


def get_rewrite_user_prompt(datum: dict[str, Any], article_titles: list[str]) -> str:
    raise NotImplementedError
    # prompt = f'Article title: "{datum["page_title"]}".\n\n'
    # prompt += f"Key points:\n{datum['text']}"

    # return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenAI batch for rewriting a paragraph from its summary."
    )

    parser.add_argument(
        "scrape_file_path",
        type=str,
        help="Path of scraped data, used for article titles.",
    )

    parser.add_argument(
        "summaries_file_path",
        type=str,
        help="Path of summaries file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    client = get_client()

    scrape_file_path: str = args.scrape_file_path
    summaries_file_path: str = args.summaries_file_path
    model: str = args.model
    task = "rewrite"

    batch_file_path = f"./tmp/{task}_{model}.jsonl"

    article_titles = get_article_titles(scrape_file_path)

    prepare_batch_file(
        summaries_file_path,
        batch_file_path,
        get_rewrite_system_prompt,
        get_rewrite_user_prompt,
        model,
        None,
        article_titles,
    )

    batch_file = upload_batch_file(batch_file_path, client)
    _ = create_batch(batch_file, task, client)


if __name__ == "__main__":
    main()
