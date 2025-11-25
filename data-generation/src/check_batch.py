import time

import openai

from utils import NEGATIVE_STATUSES, NEUTRAL_STATUSES, POSITIVE_STATUS


def await_batch(
    batch_data: openai.types.Batch,
    check_interval: float,
    client: openai.OpenAI,
) -> openai.types.Batch | None:
    while True:
        batch_data = client.batches.retrieve(batch_data.id)

        if batch_data.status in NEGATIVE_STATUSES:
            print(f"[ERROR] Batch status is {batch_data.status}")
            return None
        elif batch_data.status in NEUTRAL_STATUSES:
            print(f"[INFO] {batch_data.status}")
        elif batch_data.status == POSITIVE_STATUS:
            print("[SUCCESS] Batch completed!")
            return batch_data

        time.sleep(check_interval)
