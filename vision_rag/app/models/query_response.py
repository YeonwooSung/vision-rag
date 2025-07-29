from pydantic import BaseModel, Field


class Reference(BaseModel):
    id: int = Field(
        description="Sequential numeric identifier for the reference, starting from 1, used for citations in the answer"
    )
    title: str
    filename: str


class FinalResponse(BaseModel):
    references: list[Reference] = Field(
        description="List of unique reference entries indicating where the supporting information was found."
    )
    answer: str = Field(
        description="The complete answer text based solely on the provided context. The answer must include in-text citations in the format [id] corresponding to the references."
    )
