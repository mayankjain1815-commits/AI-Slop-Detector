import random
import re

import aiohttp

from utils import ErrorResponse
from rate_limiter import RateLimiter

SLOP_MODELS = [
    "openai/gpt-oss-20b:free",
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3-27b-it:free",
]

CLEAN_UP_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-3-27b-it:free",
]

SLOP_TEMPERATURE = 1.0
CLEAN_UP_TEMPERATURE = 1.0


class RequestHandler:
    def __init__(
        self,
        url: str,
        key: str,
        rate_limiter: RateLimiter,
        slop_models: list[str] = SLOP_MODELS,
        clean_up_models: list[str] = CLEAN_UP_MODELS,
        slop_temperature: float = SLOP_TEMPERATURE,
        clean_up_temperature: float = CLEAN_UP_TEMPERATURE,
    ):
        self.url: str = url
        self.key: str = key
        self.rate_limiter: RateLimiter = rate_limiter

        self.slop_models: list[str] = slop_models.copy()
        self.clean_up_models: list[str] = clean_up_models.copy()

        self.slop_temperature: float = slop_temperature
        self.clean_up_temperature: float = clean_up_temperature

    async def get_slop_post(self, slop_prompt: str) -> str | ErrorResponse:
        """Calls LLM API to generate a slop LinkedIn post"""
        model = random.choice(self.slop_models)
        return await self._get_post(model, slop_prompt, self.slop_temperature)

    async def get_clean_post(self, clean_up_prompt: str) -> str | ErrorResponse:
        """Calls LLM API to generate a cleaned up LinkedIn post"""
        model = random.choice(self.clean_up_models)
        return await self._get_post(model, clean_up_prompt, self.clean_up_temperature)

    async def _get_post(
        self, model: str, prompt: str, temperature: float
    ) -> str | ErrorResponse:
        """Calls LLM API to generate a LinkedIn post"""
        async with (
            aiohttp.ClientSession() as session,
            self.rate_limiter,
            session.post(
                url=self.url,
                headers=self._get_headers(),
                json=self._get_json_body(model, prompt, temperature),
            ) as response,
        ):
            try:
                data = await response.json()  # pyright:ignore[reportAny]
            except Exception:
                result = ErrorResponse("Could not get JSON response")
                return result

            try:
                text: str = data["choices"][0]["text"]  # pyright:ignore[reportAny]
            except Exception:
                result = ErrorResponse("Error response from OpenRouter")
                result.add_context(
                    [
                        f"Model: {model}",
                        f"Prompt: {prompt}",
                        f"Data: {data}",
                    ]
                )
                return result

            post = self._parse_response_text(text)
            if isinstance(post, ErrorResponse):
                post.add_context(
                    [f"Model: {model}", f"Prompt: {prompt}", f"Text: {text}"]
                )

            return post

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _get_json_body(
        self, model: str, prompt: str, temperature: float
    ) -> dict[str, str | float]:
        return {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }

    def _parse_response_text(self, response: str) -> str | ErrorResponse:
        """Parses a response for content between <POST> and </POST> tags"""
        matches: list[str] = re.findall(r"<POST>(.*?)</POST>", response, re.DOTALL)
        if matches:
            # Return last match in case model is weird and includes original post in response.
            return matches[-1]

        result = ErrorResponse("Failed to find <POST> tags")

        return result
