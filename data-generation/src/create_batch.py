import argparse
import json
from functools import cache
from typing import Any, Callable

import openai

import utils

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"


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
    prompt += "\n- Do NOT include any markdown formatting (e.g., do NOT use *italics* and do NOT use **bold**."

    return prompt


def get_rewrite_user_prompt(datum: dict[str, Any]) -> str:
    raise NotImplementedError
    # prompt = f'Article title: "{datum["page_title"]}".\n\n'
    # prompt += f"Key points:\n{datum['text']}"

    # return prompt


def prepare_batch_file(
    input_path: str,
    output_path: str,
    get_system_prompt: Callable[[], str],
    get_user_prompt: Callable[[dict[str, Any]], str],
    model: str,
    max_lines: int | None = None,
) -> None:
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for idx, line in enumerate(fin):
            if max_lines and idx + 1 > max_lines:
                break

            datum: dict[str, Any] = json.loads(line)

            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": get_system_prompt()},
                        {"role": "user", "content": get_user_prompt(datum)},
                    ],
                    "max_tokens": 2_048,
                },
            }

            fout.write(json.dumps(request, ensure_ascii=False))
            fout.write("\n")


def upload_batch_file(
    batch_file_path: str,
    client: openai.OpenAI,
) -> openai.types.FileObject:
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch",
        )

    return batch_input_file


def create_batch(
    batch_input_file: openai.types.FileObject,
    client: openai.OpenAI,
    task: str,
) -> openai.types.Batch:
    batch_data = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{task.capitalize()} Wikipedia",
        },
    )

    print(f"[INFO] Created batch with id: {batch_data.id}")
    return batch_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OpenAI batch")

    parser.add_argument(
        "input_file_path",
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

    parser.add_argument(
        "--task",
        type=str,
        choices=["summarize", "rewrite"],
        help="Task to perform: 'summarize' or 'rewrite'.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_file_path: str = args.input_file_path
    model: str = args.model
    max_lines: int = args.max_lines
    task: str = args.task

    batch_file_path = f"./tmp/{task}_{max_lines}_{model}.jsonl"

    if task == "summarize":
        prepare_batch_file(
            input_file_path,
            batch_file_path,
            get_summary_system_prompt,
            get_summary_user_prompt,
            model,
            max_lines=max_lines,
        )
    elif task == "rewrite":
        prepare_batch_file(
            input_file_path,
            batch_file_path,
            get_rewrite_system_prompt,
            get_rewrite_user_prompt,
            model,
            max_lines=max_lines,
        )
    else:
        raise ValueError("Unknown task!")

    client = utils.get_client()

    batch_file = upload_batch_file(batch_file_path, client)
    _ = create_batch(batch_file, client, task)


if __name__ == "__main__":
    main()
