"""FastAPI main application entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes_chat import router as chat_router
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_rag import router as rag_router
from backend.app.api.routes_tools import router as tools_router
from backend.app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("ai_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown hooks."""
    logger.info("Initializing Enterprise AI Assistant Engine...")
    logger.info("Configured Primary Provider: %s", settings.LLM_PROVIDER)
    logger.info("Configured Fallback Providers: %s", settings.FALLBACK_PROVIDERS)
    yield
    logger.info("Shutting down AI Assistant...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise AI Assistant with Applied AI & Engineering Systems architecture (RAG, Tool Calling, ONNX, Caching, Reliability).",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health_router)
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(rag_router, prefix=settings.API_V1_STR)
app.include_router(tools_router, prefix=settings.API_V1_STR)


@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} v{settings.VERSION}",
        "docs": "/docs",
        "health": "/healthz",
        "api_v1": settings.API_V1_STR,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
