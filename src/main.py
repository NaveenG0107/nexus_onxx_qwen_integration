from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.config import get_logger
from src.generation.api import router as generation_router
from src.generation.service import get_model_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("light-weight-llm service starting up...")
    manager = get_model_manager()
    # Pre-warm the custom SLM on startup
    _ = manager.custom_service
    yield
    logger.info("light-weight-llm service shutting down")


app = FastAPI(
    title="light-weight-llm",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(generation_router)


@app.get("/health_check")
def health_check():
    return {"message": "light-weight-llm service is running"}
