from anthropic import AsyncAnthropic
from qdrant_client import AsyncQdrantClient
from supabase.client import AsyncClient as SupabaseAsyncClient

from vision_rag.app.settings import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant.qdrant_url,
        api_key=settings.qdrant.qdrant_api_key,
    )


def create_supabase_client(settings: Settings) -> SupabaseAsyncClient:
    return SupabaseAsyncClient(
        supabase_key=settings.supabase.supabase_key,
        supabase_url=settings.supabase.supabase_url,
    )


def create_anthropic_client(settings: Settings) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=settings.anthropic.api_key)
