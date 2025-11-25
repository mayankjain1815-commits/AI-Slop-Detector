import openai

NEGATIVE_STATUSES = {"failed", "expired", "cancelling", "cancelled"}
NEUTRAL_STATUSES = {"validating", "in_progress", "finalizing"}
POSITIVE_STATUS = "completed"


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
