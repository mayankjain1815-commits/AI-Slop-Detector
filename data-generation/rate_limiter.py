import asyncio
import time
from types import TracebackType


class RateLimiter:
    def __init__(self, min_interval: float):
        """
        Args:
            min_interval: Minimum seconds between calls
        """
        self.min_interval: float = min_interval
        self.last_call: float = time.time() - min_interval
        self.lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_call = time.time()

        return self

    async def __aexit__(
        self,
        *exc: tuple[BaseException, BaseException, TracebackType] | None,
    ) -> None:
        return None
