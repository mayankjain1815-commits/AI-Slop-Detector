import asyncio
import json
import uuid

import aiofiles

PATH = "../data"


class FileWriter:
    def __init__(self, path: str = PATH):
        self.file_name: str = f"{PATH}/{uuid.uuid4()}.jsonl"
        self.lock: asyncio.Lock = asyncio.Lock()

    async def write(self, slop_post: str, clean_post: str) -> None:
        datum = {"slop": slop_post, "clean": clean_post}

        async with (
            self.lock,
            aiofiles.open(self.file_name, "a", encoding="utf-8") as file,
        ):
            _ = await file.write(json.dumps(datum, ensure_ascii=False))
            _ = await file.write("\n")
