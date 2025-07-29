import asyncio

from loguru import logger
from qdrant_client import models

from vision_rag.app.api.state import create_qdrant_client
from vision_rag.app.settings import Settings


async def main(settings: Settings):
    qdrant_client = create_qdrant_client(settings=settings)
    COLLECTION_NAME = settings.qdrant.collection_name

    collections_response = await qdrant_client.get_collections()
    collections = [
        collection.name for collection in collections_response.collections
    ]
    if COLLECTION_NAME in collections:
        logger.warning("Collection {c} already exists", c=COLLECTION_NAME)
        raise ValueError(f"Collection {COLLECTION_NAME} already exists")

    await qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            on_disk=False,
        ),
        on_disk_payload=False,
    )

    await qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


if __name__ == "__main__":
    from vision_rag.app.settings import get_settings

    settings = get_settings()
    asyncio.run(main(settings=settings))
