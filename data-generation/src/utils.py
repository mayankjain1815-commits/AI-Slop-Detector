import json
import time
from typing import Any, Callable, Concatenate, ParamSpec

import openai

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

NEGATIVE_STATUSES = {"failed", "expired", "cancelling", "cancelled"}
NEUTRAL_STATUSES = {"validating", "in_progress", "finalizing"}
POSITIVE_STATUS = "completed"

P = ParamSpec("P")


def get_env() -> dict[str, str]:
    env: dict[str, str] = {}

    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()

    return env


def get_client() -> openai.OpenAI:
    env = get_env()
    api_key = env["OPEN_AI_KEY"]
    return openai.OpenAI(api_key=api_key)


def prepare_batch_file(
    input_path: str,
    output_path_base: str,
    get_system_prompt: Callable[[], str],
    get_user_prompt: Callable[Concatenate[dict[str, Any], P], str],
    model: str,
    max_lines: int | None = None,
    split_file_count: int = 1,
    *user_prompt_args: P.args,
    **user_prompt_kwargs: P.kwargs,
) -> set[str]:
    # Get number of lines
    n_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1

    lines_per_split = n_lines // split_file_count

    # Write to output files
    used_outputs = set()
    with open(input_path, "r", encoding="utf-8") as fin:
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
                        {
                            "role": "user",
                            "content": get_user_prompt(
                                datum, *user_prompt_args, **user_prompt_kwargs
                            ),
                        },
                    ],
                },
            }

            if split_file_count == 1:
                output_path = output_path_base
            else:
                path_split = output_path_base.split(".")
                output_path = (
                    ".".join(path_split[:-1])
                    + f"_{idx // lines_per_split}."
                    + path_split[-1]
                )
            used_outputs.add(output_path)
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(request, ensure_ascii=False))
                fout.write("\n")

    return used_outputs


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
    task: str,
    client: openai.OpenAI,
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


def get_batch(batch_id: str, client: openai.OpenAI) -> openai.types.Batch:
    return client.batches.retrieve(batch_id)


def await_batch(
    batch_data: openai.types.Batch,
    check_interval: float,
    client: openai.OpenAI,
) -> openai.types.Batch | None:
    try:
        while True:
            batch_data = get_batch(batch_data.id, client)

            if batch_data.usage:
                input_tokens = batch_data.usage.input_tokens
                input_cached = batch_data.usage.input_tokens_details.cached_tokens
                output_tokens = batch_data.usage.output_tokens
                print(
                    f"[INFO] Usage: {input_tokens} input tokens ({input_cached} cached), {output_tokens} output tokens"
                )

            if batch_data.request_counts:
                completed = batch_data.request_counts.completed
                failed = batch_data.request_counts.failed
                total = batch_data.request_counts.total
                print(
                    f"[INFO] Request counts: {completed} of {total} completed, ({failed} failed)"
                )

            if batch_data.status in NEGATIVE_STATUSES:
                print(f"[ERROR] Batch status is {batch_data.status}")
                return None
            elif batch_data.status in NEUTRAL_STATUSES:
                print(f"[INFO] Status: {batch_data.status}")
            elif batch_data.status == POSITIVE_STATUS:
                print("[SUCCESS] Batch completed!")
                return batch_data

            time.sleep(check_interval)
    except KeyboardInterrupt:
        return


def retrieve_results(
    batch_data: openai.types.Batch,
    client: openai.OpenAI,
) -> str:
    output_file_id = batch_data.output_file_id
    assert isinstance(output_file_id, str)

    file_response = client.files.content(output_file_id)
    return file_response.text
