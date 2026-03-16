"""
main.py — FastAPI application entry point.

Start the server:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via the helper script:
    python main.py
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from event_router.router import router as event_router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown hooks) ───────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Voice AI Agent — Nova Sonic backend")
    logger.info("AWS region        : %s", settings.aws_region)
    logger.info("Nova Sonic model  : %s", settings.nova_sonic_model_id)
    logger.info(
        "Bedrock KB        : %s",
        settings.bedrock_kb_id or "NOT CONFIGURED (local FAISS fallback)",
    )
    logger.info(
        "ironclad-runtime  : %s",
        "FOUND" if settings.ironclad_available else "NOT FOUND (native Python)",
    )
    logger.info("Vault path        : %s", settings.vault_path)
    yield
    logger.info("Shutting down Voice AI Agent")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice AI Agent — Nova Sonic Financial Research Tool",
    description=(
        "AWS Bedrock Nova Sonic + Polygon.io + SEC RAG + Monte Carlo compute. "
        "Speak a market question, get a spoken answer."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow browser clients on localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(event_router)


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
