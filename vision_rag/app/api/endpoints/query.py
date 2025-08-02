from typing import Annotated, Any, AsyncIterator

import numpy as np
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from instructor import AsyncInstructor
from pydantic import UUID4
from qdrant_client import AsyncQdrantClient, models

from vision_rag.app.api.dependencies import (
    get_collection_name,
    get_colpali_model,
    get_colpali_processor,
    get_instructor_client,
    get_prompts,
    get_qdrant_client,
    get_supabase_downloader,
)
from vision_rag.app.models.query_response import FinalResponse
from vision_rag.app.services.img_downloader import SupabaseJPEGDownloader
from vision_rag.lib.muvera.fde_generator import (
    EncodingType,
    FixedDimensionalEncodingConfig,
    generate_query_fde,
)


router = APIRouter()


class QueryController:
    def __init__(
        self,
        model: ColQwen2_5,
        processor: ColQwen2_5_Processor,
        downloader: SupabaseJPEGDownloader,
        instructor_client: AsyncInstructor,
        qdrant_client: AsyncQdrantClient,
        collection_name: str,
        prompts: dict[str, str],
    ) -> None:
        self.model = model
        self.processor = processor
        self.downloader = downloader
        self.instructor_client = instructor_client
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.prompts = prompts

        # configurations for MUVERA
        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=7,
            seed=42,
            fill_empty_partitions=False,  # Config for query
            encoding_type=EncodingType.DEFAULT_SUM
        )


    async def query(
        self, query: str, top_k: int, session_id: UUID4
    ) -> AsyncIterator[Any]:
        with torch.inference_mode():
            processed_queries = self.processor.process_queries(
                queries=[query]
            ).to(self.model.device)
            query_embeddings = self.model(**processed_queries)

        # MUVERA -> multi-vector to fixed dimensional encoding
        # convert query_embeddings (multi-vector) to fixed dimensional encoding
        query_vectors = query_embeddings[0].cpu().numpy()
        query_vectors = generate_query_fde(query_vectors, self.doc_config)
        query_vector = query_vectors.astype(np.float32).tolist()

        search_results = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=str(session_id)),
                    )
                ]
            ),
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
        )

        points = [point.payload for point in search_results.points]
        filenames = [
            f"{point['session_id']}/{point['document']}/{point['page']}.jpeg"
            for point in points
            if point
        ]
        instructor_images = await self.downloader.download_instructor_images(
            filenames=filenames
        )

        prompt_1 = self.prompts["prompt1"]
        prompt_2 = self.prompts["prompt2"]

        query_content: list[str | object] = [prompt_1]
        for filename, image in zip(filenames, instructor_images):
            query_content.extend(
                [f'\t<image file="{filename}">', image, "\t</image>"]
            )
        query_content.append(prompt_2)

        stream = self.instructor_client.completions.create_partial(
            model="claude-3-7-sonnet-latest",
            response_model=FinalResponse,
            messages=[{"role": "user", "content": query_content}],  # type: ignore
            context={"query": query},
            temperature=0.0,
            max_tokens=8192,
            max_retries=3,
        )

        async for partial in stream:
            yield partial.model_dump_json() + "\n"


@router.post("/query/")
async def query_endpoint(
    query: str,
    top_k: int,
    session_id: UUID4,
    model: Annotated[ColQwen2_5, Depends(get_colpali_model)],
    processor: Annotated[ColQwen2_5_Processor, Depends(get_colpali_processor)],
    downloader: Annotated[
        SupabaseJPEGDownloader, Depends(get_supabase_downloader)
    ],
    instructor_client: Annotated[
        AsyncInstructor, Depends(get_instructor_client)
    ],
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    collection_name: Annotated[str, Depends(get_collection_name)],
    prompts: Annotated[dict[str, str], Depends(get_prompts)],
):
    controller = QueryController(
        model=model,
        processor=processor,
        downloader=downloader,
        instructor_client=instructor_client,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        prompts=prompts,
    )
    return StreamingResponse(
        controller.query(query, top_k, session_id),
        media_type="text/event-stream",
    )
