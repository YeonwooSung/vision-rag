import asyncio
from io import BytesIO

from loguru import logger
from PIL import Image
from pydantic import UUID4
from supabase.client import AsyncClient as SupabaseAsyncClient


class SupabaseJPEGUploader:
    def __init__(self, client: SupabaseAsyncClient, bucket_name: str):
        self.client = client
        self.bucket_name = bucket_name

    async def _upload_image(
        self, session_id: UUID4, file_name: str, page: int, image: Image.Image
    ):
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            await self.client.storage.from_(id=self.bucket_name).upload(
                path=f"{session_id}/{file_name}/{page}.jpeg",
                file=buffer.getvalue(),
                file_options={"content-type": "image/jpeg"},
            )

    async def upload_images(
        self,
        session_id: UUID4,
        file_name: str,
        images: list[Image.Image],
        start: int = 1,
    ):
        logger.info(
            "Attempting to upload {n} images for {f} in session {s}",
            n=len(images),
            f=file_name,
            s=session_id,
        )
        tasks = [
            self._upload_image(
                session_id=session_id,
                file_name=file_name,
                page=page,
                image=image,
            )
            for page, image in zip(range(start, start + len(images)), images)
        ]
        await asyncio.gather(*tasks)
        logger.success("Uploaded {n} images", n=len(images))
