import asyncio
import base64

import instructor
from supabase.client import AsyncClient as SupabaseAsyncClient


class SupabaseJPEGDownloader:
    def __init__(self, client: SupabaseAsyncClient, bucket_name: str):
        self.client = client
        self.bucket_name = bucket_name

    async def download_image(self, filename: str) -> bytes:
        return await self.client.storage.from_(id=self.bucket_name).download(
            path=filename
        )

    async def download_images(self, paths: list[str]) -> list[bytes]:
        tasks = [self.download_image(path) for path in paths]
        return await asyncio.gather(*tasks)

    async def download_instructor_images(
        self, filenames: list[str]
    ) -> list[instructor.Image]:
        images_bytes = await self.download_images(paths=filenames)
        return bytes_list_to_instructor_images(images_bytes=images_bytes)


def bytes_to_instructor_image(image_bytes: bytes) -> instructor.Image:
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return instructor.Image.from_raw_base64(base64_str)


def bytes_list_to_instructor_images(
    images_bytes: list[bytes],
) -> list[instructor.Image]:
    return [
        bytes_to_instructor_image(image_bytes=img_bytes)
        for img_bytes in images_bytes
    ]
