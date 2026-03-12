"""
router.py — Event Router: maps Nova Sonic tool-call events to Python backends.

The Event Router is the single dispatch point between Nova Sonic and the
four financial tool backends.  It exposes:

  1. A WebSocket endpoint  (/ws/voice)  for browser audio streaming.
  2. A REST endpoint       (/vault/log) for direct vault writes (tool 4).
  3. An async dispatch()   method that NovaSonicSession calls when a tool
     event arrives from the model.

Tool routing table
──────────────────
  "query_live_market_data"    → tools.market_data.get_market_snapshot()
  "analyze_sec_filings_rag"   → tools.sec_rag.query_sec_filings()
  "execute_quantitative_model"→ tools.quant_model.run_monte_carlo()
  "log_research_insight"      → tools.vault_logger.log_insight()
"""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from event_router.schemas import VaultLogRequest, VaultLogResponse
from tools.market_data import get_market_snapshot
from tools.sec_rag import query_sec_filings
from tools.quant_model import run_monte_carlo
from tools.vault_logger import log_insight
from nova_sonic.session import NovaSonicSession

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Tool dispatch table ───────────────────────────────────────────────────────

TOOL_DISPATCH: dict[str, Any] = {
    "query_live_market_data": get_market_snapshot,
    "analyze_sec_filings_rag": query_sec_filings,
    "execute_quantitative_model": run_monte_carlo,
    "log_research_insight": log_insight,
}


async def dispatch(
    tool_name: str,
    tool_input: dict[str, Any],
    session_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Route a Nova Sonic tool call to the correct backend function.

    Parameters
    ----------
    tool_name  : Name from the tool schema (must match TOOL_DISPATCH keys).
    tool_input : Parsed JSON input from the model.

    Returns
    -------
    dict that Nova Sonic will receive as the tool result.
    """
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return {"error": f"Unknown tool: {tool_name}"}

    logger.info(
        "Dispatching tool=%s input=%s",
        tool_name,
        json.dumps(tool_input),
    )
    try:
        if tool_name == "log_research_insight":
            result = await handler(**tool_input, context=session_context or {})
        else:
            result = await handler(**tool_input)
        logger.info("Tool=%s result=%s", tool_name, json.dumps(result))
        return result
    except Exception as exc:
        logger.error("Tool=%s raised %s: %s", tool_name, type(exc).__name__, exc)
        return {"error": str(exc)}


# ── WebSocket: browser audio ↔ Nova Sonic ───────────────────────────────────


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    """
    Bidirectional audio WebSocket.

    Browser sends:  raw PCM-16 @ 16 kHz/mono in binary frames.
    Server sends:   raw PCM-16 @ 24 kHz/mono in binary frames (Nova Sonic TTS).

    The session is created per-connection and torn down on disconnect.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket connected | session_id=%s", session_id)

    session = NovaSonicSession(tool_handlers=dispatch)

    try:
        await session.start()

        # Run two coroutines concurrently:
        #   • receive_loop — forward browser audio → Nova Sonic
        #   • send_loop    — stream Nova Sonic audio → browser

        import asyncio

        async def receive_loop() -> None:
            async for message in websocket.iter_bytes():
                await session.send_audio_chunk(message)

        async def send_loop() -> None:
            while session.state.name != "CLOSED":
                try:
                    pcm_chunk = await asyncio.wait_for(
                        session.audio_output_queue.get(), timeout=0.05
                    )
                    await websocket.send_bytes(pcm_chunk)
                except asyncio.TimeoutError:
                    pass

        await asyncio.gather(receive_loop(), send_loop())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | session_id=%s", session_id)
    except Exception as exc:
        logger.error("WebSocket error | session_id=%s | %s", session_id, exc)
    finally:
        await session.close()


# ── REST: vault write (also callable directly, not just via Nova Sonic) ──────


@router.post("/vault/log", response_model=VaultLogResponse)
async def vault_log_endpoint(body: VaultLogRequest) -> VaultLogResponse:
    """
    Direct REST endpoint for vault writes.
    Nova Sonic calls this indirectly via the log_research_insight tool;
    you can also POST to it manually during development.
    """
    try:
        result = await log_insight(
            content=body.content,
            tags=body.tags,
            title=body.title or None,
            context=body.context,
        )
        return VaultLogResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Health / debug ────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "tools": list(TOOL_DISPATCH.keys())})
