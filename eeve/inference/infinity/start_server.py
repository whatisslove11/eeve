import os
import logging
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import AsyncGenerator

from infinity_emb import AsyncEmbeddingEngine
from schema import EmbeddingRequest, EmbeddingResponse, EmbeddingObject


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large-instruct")
        device = os.getenv("DEVICE", "cuda")
        engine_type = os.getenv("ENGINE", "torch")
        
        logger.info(f"Initializing embedding engine with model: {model_name}")
        logger.info(f"Using device: {device}, engine: {engine_type}")
        
        app.state.engine = AsyncEmbeddingEngine(
            model_name_or_path=model_name,
            engine=engine_type,
            device=device
        )

        await app.state.engine.astart()
        logger.info("Embedding engine started successfully")
        
        yield
        
    finally:
        logger.info("Stopping embedding engine")
        if hasattr(app.state, 'engine'):
            await app.state.engine.astop()
            logger.info("Embedding engine stopped")


app = FastAPI(
    title="Infinity Embeddings Server",
    description="OpenAI-compatible embedding API using Infinity library",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)


@app.post(
    "/embeddings", 
    response_model=EmbeddingResponse,
    summary="Generate embeddings",
    description="Generate text embeddings for input"
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding engine not initialized"
        )
    
    try:
        embeddings, usage = await app.state.engine.embed(request.input)
        
        data = [
            EmbeddingObject(embedding=embedding.tolist(), index=i)
            for i, embedding in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": usage,
                "total_tokens": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding processing error: {str(e)}"
        ) from e


@app.get(
    "/health", 
    summary="Server health check",
    description="Check server status and model information"
)
async def health_check() -> dict:
    status = {
        "status": "ok",
        "model": "unknown",
        "device": "unknown",
        "ready": False
    }
    
    if hasattr(app.state, 'engine') and app.state.engine is not None:
        status.update({
            "model": app.state.engine.model,
            "device": app.state.engine.device,
            "ready": True
        })
    
    return status


if __name__ == "__main__":
    host = '0.0.0.0'
    uvicorn.run(app, host=host, port=8888)