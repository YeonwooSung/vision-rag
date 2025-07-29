from qdrant_client import AsyncQdrantClient, models
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def upsert_with_retry(
    qdrant_client: AsyncQdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
) -> None:
    await qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
