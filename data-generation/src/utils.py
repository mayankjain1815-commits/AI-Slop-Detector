from typing_extensions import override


class ErrorResponse:
    def __init__(self, reason: str):
        self.reason: str = reason
        self.contexts: list[str] = []

    def add_context(self, contexts: list[str]):
        self.contexts.extend(contexts)

    @override
    def __str__(self) -> str:
        result = f"[ERROR] {self.reason}"
        for context in self.contexts:
            result += f"- {context}\n"

        return result


def get_env() -> dict[str, str]:
    env: dict[str, str] = {}

    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()

    return env
