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
import threading
import uuid
from enum import Enum, auto
from datetime import datetime
from typing import Any, Callable, Awaitable

from nova_sonic.client import NovaSonicClient

logger = logging.getLogger(__name__)

# Type alias for tool handler callbacks registered by the Event Router
ToolHandler = Callable[
    [str, dict[str, Any], dict[str, Any] | None], Awaitable[dict[str, Any]]
]


class SessionState(Enum):
    IDLE = auto()  # 1
    LISTENING = auto()  # 2
    MODEL_THINKING = auto()  # 3
    TOOL_EXECUTING = auto()
    SPEAKING = auto()
    CLOSED = auto()  # 6


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
        # Background task reference (kept to prevent GC and allow cancellation)
        self._consumer_task: asyncio.Task | None = None
        self._tool_history: list[dict[str, Any]] = []
        self._session_context: dict[str, Any] = {
            "session_id": str(uuid.uuid4()),
            "prompt_id": self._prompt_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "last_user_summary": "",
            "tool_history": self._tool_history,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open the bidirectional stream and send the sessionStart event."""
        if self._state != SessionState.IDLE:
            raise RuntimeError("Session already started.")

        logger.info("Opening Nova Sonic stream (prompt_id=%s)", self._prompt_id)
        self._stream = self._client.open_stream()

        # Send sessionStart — configures inference settings, system prompt, and tools
        start_event = self._client.build_session_start_event()
        await self._send_event(start_event)

        # Send promptStart — configures audio input/output format for this prompt
        prompt_start_event = self._client.build_audio_input_start_event(
            self._prompt_id, self._content_id
        )
        await self._send_event(prompt_start_event)

        self._state = SessionState.LISTENING
        # Store task reference to prevent GC and allow cancellation on close()
        self._consumer_task = asyncio.create_task(self._consume_output())

    async def close(self) -> None:
        """Gracefully close the stream."""
        if self._stream and self._state != SessionState.CLOSED:
            logger.info("Closing Nova Sonic stream (prompt_id=%s)", self._prompt_id)
            try:
                self._stream["body"].close()
            except Exception:
                pass
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
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

        The boto3 response body is a synchronous iterator — running it directly
        in a coroutine would block the event loop and prevent audio chunks from
        being sent concurrently.  We push the blocking I/O into a daemon thread
        and forward parsed events back to the event loop via an asyncio.Queue.

        Handles:
          - audioOutput  → enqueue PCM bytes for the browser
          - toolUse      → execute tool via Event Router, return result
          - contentBlockStop / generationComplete → state transitions
        """
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def _blocking_reader() -> None:
            """Runs in a daemon thread — iterates the synchronous boto3 stream."""
            try:
                for raw_event in self._stream["body"]:
                    chunk_bytes = raw_event.get("chunk", {}).get("bytes", b"")
                    if chunk_bytes:
                        event = json.loads(chunk_bytes.decode("utf-8"))
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
            except Exception as exc:
                logger.error("Stream reader thread error: %s", exc)
            finally:
                # None sentinel tells the async consumer the stream is done
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        reader_thread = threading.Thread(target=_blocking_reader, daemon=True)
        reader_thread.start()

        try:
            while True:
                event = await event_queue.get()
                if event is None or self._state == SessionState.CLOSED:
                    break
                await self._handle_output_event(event)
        except asyncio.CancelledError:
            pass
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

        if tool_name == "log_research_insight":
            # The model-provided summary is useful context for final note generation.
            self._session_context["last_user_summary"] = str(tool_input.get("content", ""))

        tool_entry: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "input": tool_input,
            "invoked_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._tool_history.append(tool_entry)
        context_snapshot = dict(self._session_context)
        context_snapshot["tool_history"] = list(self._tool_history)
        context_snapshot["latest_tool_call"] = dict(tool_entry)

        try:
            result = await self._tool_handlers(tool_name, tool_input, context_snapshot)
        except Exception as exc:
            logger.error("Tool %s raised an exception: %s", tool_name, exc)
            result = {"error": str(exc)}

        tool_entry["result"] = result

        result_event = self._client.build_tool_result_event(
            self._prompt_id, tool_use_id, result
        )
        await self._send_event(result_event)
        self._state = SessionState.LISTENING

    @property
    def state(self) -> SessionState:
        return self._state
