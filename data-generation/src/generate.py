import asyncio

import utils
from file_writer import FileWriter
from prompt_generator import PromptGenerator
from rate_limiter import RateLimiter
from request_handler import RequestHandler

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SECONDS_BETWEEN_REQUESTS = 4.01  # OpenRouter rate limit is 20 requests / minute
N_DATA = 100  # OpenRouter limits to 1000 requests per day => 500 data points per run

ENV = utils.get_env()


async def run_task(
    prompt_generator: PromptGenerator,
    request_handler: RequestHandler,
    file_writer: FileWriter,
    task_id: int,
) -> None:
    try:
        slop_prompt = prompt_generator.generate_slop_prompt()

        print(f"[INFO] Task {task_id} making slop request")
        slop_post = await request_handler.get_slop_post(slop_prompt)
        if isinstance(slop_post, utils.ErrorResponse):
            print(slop_post)
            return

        clean_up_prompt = prompt_generator.generate_clean_up_prompt(slop_post)

        print(f"[INFO] Task {task_id} making clean-up request")
        clean_post = await request_handler.get_clean_post(clean_up_prompt)
        if isinstance(clean_post, utils.ErrorResponse):
            print(clean_post)
            return

        print(f"[INFO] Task {task_id} writing to file")
        await file_writer.write(slop_post, clean_post)
    except Exception as e:
        print(f"[ERROR] Task {task_id} had major failure")
        print(e)
        pass


async def main():
    prompt_generator = PromptGenerator()
    request_handler = RequestHandler(
        OPENROUTER_URL,
        ENV["OPENROUTER_KEY"],
        RateLimiter(SECONDS_BETWEEN_REQUESTS),
    )
    file_writer = FileWriter()
    print(f"[INFO] Created FileWriter with ID {file_writer.id}")

    tasks = [
        run_task(prompt_generator, request_handler, file_writer, idx)
        for idx in range(N_DATA)
    ]

    _ = await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
