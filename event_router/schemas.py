"""
schemas.py — Pydantic models for request/response shapes used by the
Event Router's HTTP endpoints and internal tool boundaries.
"""

from typing import Any
from pydantic import BaseModel, Field


# ── WebSocket / session ───────────────────────────────────────────────────────


class SessionStatus(BaseModel):
    session_id: str
    state: str


# ── Tool call shapes (mirroring Nova Sonic tool schemas) ─────────────────────


class MarketDataRequest(BaseModel):
    ticker: str = Field(..., description="Uppercase ticker symbol, e.g. 'NVDA'.")


class MarketDataResponse(BaseModel):
    ticker: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    change_pct: str
    summary: str
    data_source: str
    data_freshness: str


class SecRagRequest(BaseModel):
    company: str
    topic: str
    filing_type: str = "any"


class SecRagResponse(BaseModel):
    company: str
    topic: str
    passages: list[str]
    sources: list[str]
    summary: str


class QuantModelRequest(BaseModel):
    ticker: str
    days: int = Field(..., ge=1, le=252)
    simulations: int = Field(10_000, ge=100, le=100_000)


class QuantModelResponse(BaseModel):
    ticker: str
    days: int
    simulations: int
    current_price: float
    execution_mode: str
    calculation_engine: str
    simulation_time_seconds: float
    total_time_seconds: float
    p10: float
    p50: float
    p90: float
    mean: float
    summary: str


class VaultLogRequest(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)
    title: str = ""
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional session/tool context used to enrich the generated note.",
    )


class VaultLogResponse(BaseModel):
    saved: bool
    filepath: str
    message: str
    llm_provider: str = "none"
    llm_model: str = "none"


# ── Generic tool result envelope ─────────────────────────────────────────────


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
