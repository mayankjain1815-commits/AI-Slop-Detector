import random
import re

import aiohttp

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

SLOP_TEMPERATURE = 1.25
CLEAN_UP_TEMPERATURE = 1.0


class RequestHandler:
    def __init__(
        self,
        url: str,
        key: str,
        slop_models: list[str] = SLOP_MODELS,
        clean_up_models: list[str] = CLEAN_UP_MODELS,
        slop_temperature: float = SLOP_TEMPERATURE,
        clean_up_temperature: float = CLEAN_UP_TEMPERATURE,
    ):
        self.url: str = url
        self.key: str = key

        self.slop_models: list[str] = slop_models.copy()
        self.clean_up_models: list[str] = clean_up_models.copy()

        self.slop_temperature: float = slop_temperature
        self.clean_up_temperature: float = clean_up_temperature

        self.session: aiohttp.ClientSession = aiohttp.ClientSession()

    async def get_slop_post(self, prompt: str) -> str | None:
        model = random.choice(self.slop_models)
        return await self._get_post(model, prompt, self.slop_temperature)

    async def get_clean_post(self, prompt: str) -> str | None:
        model = random.choice(self.clean_up_models)
        return await self._get_post(model, prompt, self.clean_up_temperature)

    async def _get_post(
        self, model: str, prompt: str, temperature: float
    ) -> str | None:
        """Returns a LinkedIn post"""
        async with self.session:
            async with self.session.post(
                url=self.url,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                },
            ) as response:
                data = await response.json()
                try:
                    text: str = data["choices"][0]["text"]
                except KeyError:
                    raise RuntimeError(f"Unexpected response from {self.url}:\n{data}")

                return self._parse_response_text(text)

    def _parse_response_text(self, response: str) -> str | None:
        """Parses a response for <POST> and </POST> tags"""
        matches: list[str] = re.findall(r"<POST>(.*?)</POST>", response, re.DOTALL)
        if matches:
            # Return last match in case model is weird and includes original post in response.
            return matches[-1]
