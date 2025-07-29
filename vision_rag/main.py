from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .app.api.endpoints import pdf_ingest, query
from .app.api.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(pdf_ingest.router)
app.include_router(query.router)
