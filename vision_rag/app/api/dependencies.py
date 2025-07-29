from functools import lru_cache

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import Request
from instructor import AsyncInstructor
from qdrant_client import AsyncQdrantClient

from vision_rag.app.services.img_downloader import SupabaseJPEGDownloader
from vision_rag.app.services.img_uploader import SupabaseJPEGUploader
from vision_rag.app.utils.prompt_utils import read_prompt_from_plain_file


async def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.state.qdrant_client


async def get_colpali_model(request: Request) -> ColQwen2_5:
    return request.state.model


async def get_colpali_processor(request: Request) -> ColQwen2_5_Processor:
    return request.state.processor


async def get_supabase_uploader(request: Request) -> SupabaseJPEGUploader:
    return request.state.supabase_uploader


async def get_supabase_downloader(request: Request) -> SupabaseJPEGDownloader:
    return request.state.supabase_downloader


async def get_collection_name(request: Request) -> str:
    return request.state.collection_name


async def get_instructor_client(request: Request) -> AsyncInstructor:
    return request.state.instructor_client


@lru_cache(maxsize=1)
def get_prompts():
    prompt1 = read_prompt_from_plain_file("prompts/response_1")
    prompt2 = read_prompt_from_plain_file("prompts/response_2")
    return {"prompt1": prompt1, "prompt2": prompt2}
