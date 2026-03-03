"""
session.py — Nova Sonic audio session state machine.

Manages the lifecycle of a single voice conversation:
  IDLE → LISTENING → MODEL_THINKING → TOOL_EXECUTING → SPEAKING → IDLE

The session receives raw PCM audio bytes from the browser (via FastAPI
WebSocket), forwards them to Nova Sonic, listens for model events, and
fires tool callbacks into the Event Router when the model requests a tool.
"""

import asyncio
import base64
import json
import logging
import uuid
from enum import Enum, auto
from typing import Any, Callable, Awaitable

from nova_sonic.client import NovaSonicClient

logger = logging.getLogger(__name__)

# Type alias for tool handler callbacks registered by the Event Router
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    MODEL_THINKING = auto()
    TOOL_EXECUTING = auto()
    SPEAKING = auto()
    CLOSED = auto()


class NovaSonicSession:
    """
    Represents one active voice conversation with Nova Sonic.

    Usage
    -----
    session = NovaSonicSession(tool_handlers=router.dispatch)
    await session.start()
    await session.send_audio_chunk(pcm_bytes)
    # ... stream audio_output_chunk events back to browser ...
    await session.close()
    """

    def __init__(self, tool_handlers: ToolHandler) -> None:
        self._client = NovaSonicClient()
        self._tool_handlers = tool_handlers
        self._state = SessionState.IDLE
        self._stream: Any = None
        self._prompt_id: str = str(uuid.uuid4())
        self._content_id: str = str(uuid.uuid4())
        # Queue for audio output chunks to stream back to the browser
        self.audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open the bidirectional stream and send the sessionStart event."""
        if self._state != SessionState.IDLE:
            raise RuntimeError("Session already started.")

        logger.info("Opening Nova Sonic stream (prompt_id=%s)", self._prompt_id)
        self._stream = self._client.open_stream()

        # Send sessionStart
        start_event = self._client.build_session_start_event()
        await self._send_event(start_event)

        self._state = SessionState.LISTENING
        # Start background task to consume model output events
        asyncio.create_task(self._consume_output())

    async def close(self) -> None:
        """Gracefully close the stream."""
        if self._stream and self._state != SessionState.CLOSED:
            logger.info("Closing Nova Sonic stream (prompt_id=%s)", self._prompt_id)
            try:
                self._stream["body"].close()
            except Exception:
                pass
        self._state = SessionState.CLOSED

    # ── Audio I/O ─────────────────────────────────────────────────────────────

    async def send_audio_chunk(self, pcm_bytes: bytes) -> None:
        """
        Forward a raw PCM-16 audio chunk to Nova Sonic.
        Caller should chunk at ~100ms (3 200 bytes @ 16 kHz/16-bit/mono).
        """
        if self._state not in (SessionState.LISTENING, SessionState.SPEAKING):
            return  # Drop audio when model is processing a tool call

        audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        event = self._client.build_audio_chunk_event(
            self._prompt_id, self._content_id, audio_b64
        )
        await self._send_event(event)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Serialize and write one event into the bidirectional stream."""
        payload = json.dumps(event).encode("utf-8")
        try:
            self._stream["input_stream"].send({"chunk": {"bytes": payload}})
        except Exception as exc:
            logger.error("Error sending event to Nova Sonic: %s", exc)
            raise

    async def _consume_output(self) -> None:
        """
        Background task: read events from Nova Sonic output stream.

        Handles:
          - audioOutput  → enqueue PCM bytes for the browser
          - toolUse      → execute tool via Event Router, return result
          - contentBlockStop / generationComplete → state transitions
        """
        try:
            for raw_event in self._stream["body"]:
                if self._state == SessionState.CLOSED:
                    break

                chunk_bytes = raw_event.get("chunk", {}).get("bytes", b"")
                if not chunk_bytes:
                    continue

                event = json.loads(chunk_bytes.decode("utf-8"))
                await self._handle_output_event(event)

        except Exception as exc:
            logger.error("Output stream consumer error: %s", exc)
        finally:
            self._state = SessionState.CLOSED

    async def _handle_output_event(self, event: dict[str, Any]) -> None:
        """Dispatch a single Nova Sonic output event to the right handler."""

        if "audioOutput" in event:
            audio_b64: str = event["audioOutput"].get("content", "")
            pcm = base64.b64decode(audio_b64)
            await self.audio_output_queue.put(pcm)
            self._state = SessionState.SPEAKING

        elif "toolUse" in event:
            await self._handle_tool_use(event["toolUse"])

        elif "contentBlockStop" in event:
            logger.debug("contentBlockStop received")
            self._state = SessionState.LISTENING

        elif "generationComplete" in event:
            logger.debug("generationComplete received")
            self._state = SessionState.LISTENING

        elif "error" in event:
            logger.error("Nova Sonic error event: %s", event["error"])

    async def _handle_tool_use(self, tool_event: dict[str, Any]) -> None:
        """
        Called when Nova Sonic decides to invoke a tool.
        Delegates to the Event Router's dispatch function and returns the result.
        """
        tool_name: str = tool_event.get("name", "")
        tool_use_id: str = tool_event.get("toolUseId", "")
        tool_input: dict[str, Any] = tool_event.get("input", {})

        logger.info(
            "Tool requested: %s | id=%s | input=%s", tool_name, tool_use_id, tool_input
        )
        self._state = SessionState.TOOL_EXECUTING

        try:
            result = await self._tool_handlers(tool_name, tool_input)
        except Exception as exc:
            logger.error("Tool %s raised an exception: %s", tool_name, exc)
            result = {"error": str(exc)}

        result_event = self._client.build_tool_result_event(
            self._prompt_id, tool_use_id, result
        )
        await self._send_event(result_event)
        self._state = SessionState.LISTENING

    @property
    def state(self) -> SessionState:
        return self._state
