from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import instructor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import FastAPI
from instructor import AsyncInstructor
from qdrant_client import AsyncQdrantClient

from vision_rag.app.api.state import (
    create_anthropic_client,
    create_qdrant_client,
    create_supabase_client,
)
from vision_rag.app.colpali.loaders import ColQwen2_5Loader
from vision_rag.app.services.img_downloader import SupabaseJPEGDownloader
from vision_rag.app.services.img_uploader import SupabaseJPEGUploader
from vision_rag.app.settings import get_settings


class State(TypedDict):
    model: ColQwen2_5
    processor: ColQwen2_5_Processor
    supabase_uploader: SupabaseJPEGUploader
    supabase_downloader: SupabaseJPEGDownloader
    instructor_client: AsyncInstructor
    qdrant_client: AsyncQdrantClient
    collection_name: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    settings = get_settings()
    qdrant_client = create_qdrant_client(settings=settings)
    anthropic_client = create_anthropic_client(settings=settings)
    instructor_client = instructor.from_anthropic(client=anthropic_client)
    supabase_client = create_supabase_client(settings=settings)
    supabase_uploader = SupabaseJPEGUploader(
        client=supabase_client, bucket_name=settings.supabase.bucket
    )
    supabase_downloader = SupabaseJPEGDownloader(
        client=supabase_client, bucket_name=settings.supabase.bucket
    )
    loader = ColQwen2_5Loader(model_name=settings.colpali.colpali_model_name)
    model, processor = loader.load()

    yield {
        "model": model,
        "processor": processor,
        "supabase_uploader": supabase_uploader,
        "supabase_downloader": supabase_downloader,
        "instructor_client": instructor_client,
        "qdrant_client": qdrant_client,
        "collection_name": settings.qdrant.collection_name,
    }

    await qdrant_client.close()
    await anthropic_client.close()
