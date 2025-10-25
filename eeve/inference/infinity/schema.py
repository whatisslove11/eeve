from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: list[str]
    model: str
    user: str | None = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: dict
