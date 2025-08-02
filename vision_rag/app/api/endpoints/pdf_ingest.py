from typing import Annotated
from uuid import uuid4

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from fastapi import APIRouter, Depends, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pdf2image import convert_from_bytes
from pydantic import UUID4
from qdrant_client import AsyncQdrantClient, models

from vision_rag.app.api.dependencies import (
    get_collection_name,
    get_colpali_model,
    get_colpali_processor,
    get_qdrant_client,
    get_supabase_uploader,
)
from vision_rag.app.services.img_uploader import SupabaseJPEGUploader
from vision_rag.app.utils.qdrant_utils import upsert_with_retry
from vision_rag.lib.muvera.fde_generator import (
    EncodingType,
    FixedDimensionalEncodingConfig,
    generate_document_fde_batch,
)


router = APIRouter()


class PDFIngestController:
    def __init__(
        self,
        model: ColQwen2_5,
        processor: ColQwen2_5_Processor,
        uploader: SupabaseJPEGUploader,
        qdrant_client: AsyncQdrantClient,
        collection_name: str,
    ):
        self.model = model
        self.processor = processor
        self.uploader = uploader
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

        # configurations for MUVERA
        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=20,
            num_simhash_projections=7,
            seed=42,
            fill_empty_partitions=True,  # Config for documents
            encoding_type=EncodingType.DEFAULT_SUM,
        )


    async def ingest(
        self, files: list[UploadFile], session_id: UUID4
    ) -> dict[str, list[dict[str, str | int | None]]]:
        results = []
        batch_size = 1

        for file in files:
            try:
                pdf_bytes = await file.read()
                images = await run_in_threadpool(
                    convert_from_bytes,
                    pdf_file=pdf_bytes,
                    dpi=300,
                    thread_count=4,
                    fmt="jpeg",
                )
                num_images = len(images)
                total_batches = (num_images + batch_size - 1) // batch_size

                for batch_idx, start_idx in enumerate(
                    range(0, num_images, batch_size)
                ):
                    end_idx = start_idx + batch_size
                    batch = images[start_idx:end_idx]

                    with torch.inference_mode():
                        processed_images = self.processor.process_images(batch).to(self.model.device)
                        batch_embeddings = self.model(**processed_images)

                    # MUVERA -> multi-vector to fixed dimensional encoding
                    batch_vectors = batch_embeddings.cpu().numpy()
                    batch_vectors = generate_document_fde_batch(batch_vectors, self.doc_config)

                    points = []
                    for offset, (_, embedding) in enumerate(
                        zip(batch, batch_embeddings)
                    ):
                        page_number = start_idx + offset + 1
                        payload = {
                            "session_id": session_id,
                            "document": file.filename,
                            "page": page_number,
                        }
                        vector = embedding.cpu().float().numpy().tolist()
                        point = models.PointStruct(
                            id=str(uuid4()),
                            vector=vector,
                            payload=payload,
                        )
                        points.append(point)

                    await upsert_with_retry(
                        qdrant_client=self.qdrant_client,
                        collection_name=self.collection_name,
                        points=points,
                    )

                    await self.uploader.upload_images(
                        session_id=session_id,
                        file_name=file.filename or "unknown",
                        images=batch,
                        start=start_idx + 1,
                    )

                    logger.info(
                        "Processed batch {batch_num}/{total_batches} for file {filename}",
                        batch_num=batch_idx + 1,
                        total_batches=total_batches,
                        filename=file.filename,
                    )

                results.append(
                    {"filename": file.filename, "num_pages": num_images}
                )
            except Exception as e:
                logger.error(
                    "Error processing file {filename}: {error}",
                    filename=file.filename,
                    error=str(e),
                )
                results.append({"filename": file.filename, "error": str(e)})
        return {"results": results}


@router.post("/ingest-pdfs/")
async def ingest_pdf(
    files: list[UploadFile],
    session_id: UUID4,
    model: Annotated[ColQwen2_5, Depends(get_colpali_model)],
    processor: Annotated[ColQwen2_5_Processor, Depends(get_colpali_processor)],
    uploader: Annotated[SupabaseJPEGUploader, Depends(get_supabase_uploader)],
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    collection_name: Annotated[str, Depends(get_collection_name)],
):
    controller = PDFIngestController(
        model=model,
        processor=processor,
        uploader=uploader,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
    )
    return await controller.ingest(files=files, session_id=session_id)
