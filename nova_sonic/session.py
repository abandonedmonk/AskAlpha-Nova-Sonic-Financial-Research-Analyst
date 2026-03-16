"""
session.py — Nova Sonic audio session state machine.
"""

import asyncio
import base64
import json
import logging
import uuid
from enum import Enum, auto
from datetime import datetime
from typing import Any, Callable, Awaitable

from nova_sonic.client import NovaSonicClient
from config import settings

logger = logging.getLogger(__name__)

ToolHandler = Callable[
    [str, dict[str, Any], dict[str, Any] | None], Awaitable[dict[str, Any]]
]


class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    MODEL_THINKING = auto()
    TOOL_EXECUTING = auto()
    SPEAKING = auto()
    CLOSED = auto()


class NovaSonicSession:

    def __init__(self, tool_handlers: ToolHandler) -> None:
        self._client = NovaSonicClient()
        self._tool_handlers = tool_handlers
        self._state = SessionState.IDLE
        self._stream: Any = None
        self._prompt_id: str = str(uuid.uuid4())
        self._content_id: str = str(uuid.uuid4())
        self.audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.metadata_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        self._tool_history: list[dict[str, Any]] = []
        self._current_block_role: str | None = None
        self._pending_tool_use: dict | None = None  # buffered until contentEnd
        self._user_utterance_parts: list[str] = []  # ← CAPTURE USER SPEECH
        self._audio_chunks_received: int = 0  # ← TRACK AUDIO FLOW
        self._session_context: dict[str, Any] = {
            "session_id": str(uuid.uuid4()),
            "prompt_id": self._prompt_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "last_user_summary": "",
            "tool_history": self._tool_history,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._state != SessionState.IDLE:
            raise RuntimeError("Session already started.")

        try:
            logger.info("=" * 80)
            logger.info("PREFLIGHT: Checking AWS credentials from config...")
            
            # Use pydantic settings (loaded from .env)
            try:
                access_key = settings.aws_access_key_id
                secret_key = settings.aws_secret_access_key
                region = settings.aws_region
                
                if not access_key:
                    logger.error("✗ AWS_ACCESS_KEY_ID not configured!")
                    raise RuntimeError("AWS_ACCESS_KEY_ID not in .env or environment")
                if not secret_key:
                    logger.error("✗ AWS_SECRET_ACCESS_KEY not configured!")
                    raise RuntimeError("AWS_SECRET_ACCESS_KEY not in .env or environment")
                
                logger.info("✓ AWS credentials loaded from .env")
                logger.info("  Region: %s", region)
            except Exception as cred_err:
                logger.error("✗ Credentials check failed: %s", cred_err)
                raise
            
            logger.info("=" * 80)
            logger.info("STEP 1: Opening Nova Sonic stream...")
            logger.info("  AWS credentials: loaded from .env")
            logger.info("  AWS_ACCESS_KEY_ID: SET (from .env)")
            logger.info("  AWS_SECRET_ACCESS_KEY: SET (from .env)")
            logger.info("  AWS_REGION: %s", settings.aws_region)
            logger.info("  Calling invoke_model_with_bidirectional_stream()...")
            self._stream = await self._client.open_stream()
            logger.info("✓ Stream opened successfully!")
            logger.info("  Stream object type: %s", type(self._stream).__name__)
            logger.info("  Has input_stream: %s", hasattr(self._stream, 'input_stream'))
            logger.info("  Has output_stream: %s", hasattr(self._stream, 'output_stream'))

            # CRITICAL: consumer MUST start before any send_event() call.
            # input_stream.send() blocks until output_stream has an active reader.
            # await_output() is the correct AWS SDK v2 API — NOT async for on output_stream.
            logger.info("STEP 2: Starting consumer task...")
            self._consumer_task = asyncio.create_task(self._consume_output())
            logger.info("  Consumer task created, yielding...")
            await asyncio.sleep(0.1)  # Give consumer time to reach await_output()
            logger.info("  Checking consumer task status...")
            if self._consumer_task.done():
                logger.error("✗ Consumer task already done! Exception: %s", self._consumer_task.exception())
                raise RuntimeError(f"Consumer task crashed immediately: {self._consumer_task.exception()}")
            logger.info("✓ Consumer running and waiting for Nova events")

            logger.info("STEP 3: Sending sessionStart...")
            await self._send_event(self._client.build_session_start_event())
            logger.info("✓ sessionStart sent")

            logger.info("STEP 4: Sending promptStart...")
            await self._send_event(
                self._client.build_audio_input_start_event(self._prompt_id, self._content_id)
            )
            logger.info("✓ promptStart sent")

            logger.info("STEP 5: Sending system prompt...")
            sys_content_id = str(uuid.uuid4())
            system_prompt = (
                "You are a financial research assistant with access to four tools. "
                "You MUST use the provided tools whenever the answer requires market data, SEC filings, "
                "quantitative simulation, or saving notes. Do not claim you lack access when a tool can "
                "answer the request. Use query_live_market_data for stock prices and market moves. "
                "Use analyze_sec_filings_rag for questions about what a company said in a 10-K or 10-Q. "
                "Use execute_quantitative_model for Monte Carlo or forward price simulation requests. "
                "Use log_research_insight to save notes or summaries to the vault. "
                "After receiving tool results, answer concisely and naturally."
            )
            await self._send_event(self._client.build_system_prompt_start_event(self._prompt_id, sys_content_id))
            await self._send_event(self._client.build_system_prompt_text_event(self._prompt_id, sys_content_id, system_prompt))
            await self._send_event(self._client.build_content_end_event(self._prompt_id, sys_content_id))
            logger.info("✓ System prompt sent")

            await asyncio.sleep(0.05)
            self._state = SessionState.LISTENING
            logger.info("✓✓✓ SESSION FULLY INITIALIZED — State: LISTENING ✓✓✓")
            logger.info("=" * 80)

        except Exception as e:
            logger.error("")
            logger.error("="*80)
            logger.error("✗✗✗ SESSION INITIALIZATION FAILED ✗✗✗")
            logger.error("="*80)
            logger.error("Exception Type: %s", type(e).__name__)
            logger.error("Exception Message: %s", str(e))
            logger.error("")
            logger.error("DEBUGGING CHECKLIST:")
            logger.error("  1️⃣  AWS credentials configured?")
            logger.error("       Run: echo $AWS_ACCESS_KEY_ID; echo $AWS_SECRET_ACCESS_KEY")
            logger.error("  2️⃣  Credentials valid? (not expired)")
            logger.error("       Check AWS Management Console > Security Credentials")
            logger.error("  3️⃣  Region correct? (should be us-east-1)")
            logger.error("       Nova Sonic may not be available in all regions")
            logger.error("  4️⃣  Bedrock permissions? (need bedrock:* IAM permissions)")
            logger.error("       Check IAM policy for your access key")
            logger.error("  5️⃣  Model available? (amazon.nova-2-sonic-v1:0)")
            logger.error("       Check AWS Bedrock console for model access")
            logger.error("")
            logger.error("Full Traceback:")
            import traceback
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error("  %s", line)
            logger.error("="*80)
            logger.error("")
            self._state = SessionState.CLOSED
            raise

    async def close(self) -> None:
        if self._state == SessionState.CLOSED:
            return
        self._state = SessionState.CLOSED
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        if self._stream:
            try:
                await self._send_event({"event": {"sessionEnd": {}}})
                await self._stream.input_stream.close()
            except Exception as e:
                logger.debug("Error closing stream: %s", e)

    # ── Audio input lifecycle ─────────────────────────────────────────────────

    async def start_audio_input(self) -> None:
        self._user_utterance_parts = []  # ← RESET on new audio input
        audio_content_id = str(uuid.uuid4())
        self._audio_content_id = audio_content_id
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self._prompt_id,
                    "contentName": audio_content_id,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        })
        logger.info("✓ Audio contentStart sent (id=%s)", audio_content_id)

    async def send_audio_chunk(self, pcm_bytes: bytes) -> None:
        # Allow audio during TOOL_EXECUTING too (mic stays open while tool runs)
        if self._state not in (SessionState.LISTENING, SessionState.SPEAKING, 
                               SessionState.IDLE, SessionState.MODEL_THINKING,
                               SessionState.TOOL_EXECUTING):
            logger.warning("Blocked: state=%s", self._state.name)
            return
        if not hasattr(self, '_audio_content_id'):
            logger.warning("Blocked: no audio content block open yet")
            return
        audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        logger.debug("→ Sending %d byte audio chunk to Nova Sonic", len(pcm_bytes))
        await self._send_event({
            "event": {
                "audioInput": {
                    "promptName": self._prompt_id,
                    "contentName": self._audio_content_id,
                    "content": audio_b64,
                }
            }
        })

    async def end_audio_input(self) -> None:
        if not hasattr(self, '_audio_content_id'):
            logger.warning("end_audio_input: no active audio block")
            return
        await self._send_event({
            "event": {
                "contentEnd": {
                    "promptName": self._prompt_id,
                    "contentName": self._audio_content_id,
                }
            }
        })
        logger.info("✓ Audio contentEnd sent")
        await self._send_event({
            "event": {
                "promptEnd": {"promptName": self._prompt_id}
            }
        })
        logger.info("✓ promptEnd sent — Nova Sonic will now respond")
        del self._audio_content_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._client.send_event(self._stream, event)

    async def _consume_output(self) -> None:
        """
        Read events from Nova Sonic.

        ROOT CAUSE FIX: The previous code used
            async for event_chunk in self._stream.output_stream
        which is NOT a valid iterator on the AWS SDK v2 — it never yields,
        so input_stream.send() blocked forever waiting for a reader, causing
        the stream to time out after ~42 seconds on every single connection.

        The correct AWS SDK v2 pattern (from every AWS reference, the medium
        article, and test_audio_with_tools.py) is:
            output = await self._stream.await_output()
            result = await output[1].receive()
        """
        logger.info("CONSUMER: started — using await_output() (correct SDK v2 API)")
        event_count = 0
        consecutive_errors = 0

        try:
            logger.info("CONSUMER: Waiting for first event from Nova Sonic (timeout=30s)...")
            while self._state != SessionState.CLOSED:
                try:
                    logger.debug("CONSUMER: Calling await_output()...")
                    try:
                        output = await asyncio.wait_for(self._stream.await_output(), timeout=30.0)
                        logger.debug("CONSUMER: await_output() succeeded, type=%s", type(output))
                    except asyncio.TimeoutError:
                        logger.error("✗ CONSUMER: await_output() timed out after 30 seconds!")
                        logger.error("  This usually means: AWS credentials invalid, region wrong, or model unavailable")
                        raise RuntimeError("Stream timeout: AWS not responding after 30s. Check credentials and model access.")
                    
                    logger.debug("CONSUMER: Calling receive()...")
                    result = await output[1].receive()

                    if not result.value or not result.value.bytes_:
                        logger.debug("Received empty result, continuing...")
                        consecutive_errors = 0
                        continue

                    event_count += 1
                    consecutive_errors = 0
                    
                    try:
                        raw = json.loads(result.value.bytes_.decode("utf-8"))
                    except json.JSONDecodeError as je:
                        logger.error("JSON decode error on event #%d: %s", event_count, je)
                        continue

                    if "event" not in raw:
                        logger.debug("No event key in raw: %s", list(raw.keys()))
                        continue

                    event_keys = list(raw["event"].keys())
                    logger.info("EVENT #%d: %s", event_count, event_keys)
                    
                    try:
                        await self._handle_output_event(raw["event"])
                    except Exception as e:
                        logger.error("Error handling event #%d (%s): %s", event_count, event_keys, e, exc_info=True)
                        # Don't crash on handler errors, continue processing

                except asyncio.CancelledError:
                    logger.info("Consumer cancelled")
                    raise
                except StopAsyncIteration:
                    logger.info("Stream ended (StopAsyncIteration)")
                    break
                except Exception as e:
                    if self._state == SessionState.CLOSED:
                        logger.info("Session already closed, consumer exiting")
                        break
                    consecutive_errors += 1
                    logger.error(
                        "Event read error #%d (consecutive_error=%d/%d): %s",
                        event_count, consecutive_errors, 5, e, exc_info=True
                    )
                    # Give up after 5 consecutive errors
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive errors, closing session")
                        self._state = SessionState.CLOSED
                        break
                    # Brief backoff before retrying
                    await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.info("Consumer cancelled at top level")
        except Exception as exc:
            logger.error("Consumer fatal error: %s", exc, exc_info=True)
        finally:
            logger.info("Consumer done after %d events", event_count)
            if self._state != SessionState.CLOSED:
                logger.info("Consumer exiting but session still active, allowing graceful close")
            else:
                logger.info("Session already marked closed")
            self._state = SessionState.CLOSED

    async def _handle_output_event(self, event: dict[str, Any]) -> None:

        if "audioOutput" in event:
            audio_b64: str = event["audioOutput"].get("content", "")
            pcm = base64.b64decode(audio_b64)
            self._audio_chunks_received += 1  # ← TRACK AUDIO
            if self._audio_chunks_received == 1:
                logger.info("")
                logger.info("="*80)
                logger.info("🔊 [AUDIO OUTPUT STARTED] Nova Sonic is synthesizing response audio...")
                logger.info("="*80)
                logger.info("")
            logger.info("  📻 [AUDIO] response chunk #%d: %d bytes", self._audio_chunks_received, len(pcm))
            await self.audio_output_queue.put(pcm)
            self._state = SessionState.SPEAKING

        elif "textOutput" in event:
            text: str = event["textOutput"].get("content", "")
            role = self._current_block_role or "ASSISTANT"
            logger.info("  [TEXT] role=%s: %s", role, text[:100])
            if role == "USER":
                # ← CAPTURE USER UTTERANCE for tool enhancement
                self._user_utterance_parts.append(text)
                if text.strip():
                    await self.metadata_queue.put({"type": "user_transcript", "text": text})
            else:
                # Emit as streaming chunk — frontend accumulates and commits on response_complete
                await self.metadata_queue.put({"type": "transcript", "text": text})

        elif "inputTranscription" in event:
            transcription: str = event["inputTranscription"].get("content", "")
            if transcription.strip():
                self._user_utterance_parts.append(transcription)
                logger.info("")
                logger.info("🎙️  [SPEECH-TO-TEXT DECODED] Nova heard: \"%s\"", transcription)
                logger.info("    Total utterance parts captured: %d", len(self._user_utterance_parts))
                logger.info("")
                await self.metadata_queue.put({"type": "user_transcript", "text": transcription})

        elif "contentStart" in event:
            self._current_block_role = event["contentStart"].get("role", "ASSISTANT")
            block_type = event["contentStart"].get("type", "TEXT")
            logger.info("  [FLOW] contentStart role=%s type=%s", self._current_block_role, block_type)
            if self._current_block_role == "ASSISTANT":
                self._state = SessionState.MODEL_THINKING

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                chunk = delta["text"]
                role = self._current_block_role or "ASSISTANT"
                if role == "USER":
                    await self.metadata_queue.put({"type": "user_transcript", "text": chunk})
                else:
                    await self.metadata_queue.put({"type": "transcript", "text": chunk})

        elif "toolUse" in event:
            # Buffer the toolUse payload — Nova Sonic sends name+input here but we
            # must wait for the matching contentEnd before the payload is complete.
            self._pending_tool_use = event["toolUse"]
            logger.info("  [TOOL] toolUse buffered: keys=%s", list(event["toolUse"].keys()))

        elif "contentBlockStop" in event:
            logger.info("  [FLOW] contentBlockStop")
            self._state = SessionState.LISTENING
            self._current_block_role = None

        elif "contentEnd" in event:
            role = self._current_block_role
            block_type = event["contentEnd"].get("type", "") if isinstance(event.get("contentEnd"), dict) else ""
            logger.info("  [FLOW] contentEnd role=%s", role)

            if role == "TOOL" and self._pending_tool_use is not None:
                # Now the full toolUse payload is committed — execute the tool
                tool_event = self._pending_tool_use
                self._pending_tool_use = None
                self._current_block_role = None
                self._state = SessionState.LISTENING
                await self._handle_tool_use(tool_event)
            else:
                prev_role = self._current_block_role
                self._current_block_role = None
                self._state = SessionState.LISTENING
                # Nova Sonic does NOT send generationComplete — the turn ends with
                # the last ASSISTANT TEXT contentEnd. Emit response_complete here so
                # the frontend commits the full turn to the chat panel.
                if prev_role == "ASSISTANT":
                    logger.info("  [FLOW] ASSISTANT TEXT ended → emitting response_complete")
                    await self.metadata_queue.put({"type": "response_complete"})

        elif "generationComplete" in event:
            logger.info("  [FLOW] generationComplete")
            await self.metadata_queue.put({"type": "response_complete"})
            self._state = SessionState.LISTENING

        elif "promptEnd" in event:
            logger.info("  [FLOW] promptEnd from server")

        elif "sessionEnd" in event:
            logger.info("  [FLOW] sessionEnd")
            self._state = SessionState.CLOSED

        elif "error" in event:
            logger.error("  [ERROR] %s", event["error"])

        else:
            logger.debug("  [?] unhandled: %s", list(event.keys()))

    async def _handle_tool_use(self, tool_event: dict[str, Any]) -> None:
        # Extract tool name: check both "name" and "toolName" (SDK variations)
        tool_name: str = tool_event.get("name") or tool_event.get("toolName") or ""
        tool_use_id: str = tool_event.get("toolUseId", "")

        # Nova Sonic sends `input` as a JSON-encoded STRING, not a dict.
        # The test harness handles this with json.loads(); we must do the same.
        raw_input = tool_event.get("input") or tool_event.get("content", {})
        if isinstance(raw_input, str):
            try:
                tool_input: dict[str, Any] = json.loads(raw_input)
                logger.info("✓ Parsed tool input from JSON string: %s", tool_input)
            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse tool input JSON string: %r", raw_input)
                tool_input = {"raw_input": raw_input}
        elif isinstance(raw_input, dict):
            tool_input = raw_input
        else:
            tool_input = {}

        # Log full event structure for debugging
        logger.info("toolUse event keys: %s", list(tool_event.keys()))
        logger.info("TOOL: %s id=%s input=%s", tool_name, tool_use_id, tool_input)
        
        # Safety check: if tool_name is still empty, log warning and extract from event dict
        if not tool_name and tool_event:
            logger.warning("⚠️ tool_name is empty! Event keys: %s, Event: %s", 
                          list(tool_event.keys()), json.dumps(tool_event, default=str))
        
        self._state = SessionState.TOOL_EXECUTING

        await self.metadata_queue.put({"type": "tool_call", "tool_name": tool_name, "input": tool_input})

        if tool_name == "log_research_insight":
            self._session_context["last_user_summary"] = str(tool_input.get("content", ""))

        # ← ENHANCE TOOL INPUT using captured user speech (new)
        if tool_name == "analyze_sec_filings_rag":
            transcript = " ".join(self._user_utterance_parts).lower()
            company = str(tool_input.get("company", "")).strip()
            filing_type = str(tool_input.get("filing_type", "any")).strip() or "any"

            # Normalize filing type from transcript
            if filing_type.lower() == "any":
                if any(x in transcript for x in ["10-k", "10 k", "10k", "ten k", "tenk", "ten-k"]):
                    filing_type = "10-K"
                elif any(x in transcript for x in ["10-q", "10 q", "10q", "ten q", "tenq", "ten-q"]):
                    filing_type = "10-Q"

            # Normalize company name using transcript
            if company:
                known_companies = ["nvidia", "amd", "apple", "microsoft", "amazon", "google", "alphabet", "meta", "tesla"]
                transcript_company = next((c for c in known_companies if c in transcript), None)
                if transcript_company and transcript_company.lower() != company.lower():
                    company = transcript_company.title()

            tool_input["company"] = company or tool_input.get("company", "")
            tool_input["filing_type"] = filing_type
            logger.info("✓ Enhanced tool input from user transcript: company=%s filing_type=%s", 
                       tool_input["company"], tool_input["filing_type"])

        tool_entry: dict[str, Any] = {
            "tool_name": tool_name, "tool_use_id": tool_use_id,
            "input": tool_input, "invoked_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._tool_history.append(tool_entry)
        context_snapshot = {
            **self._session_context,
            "tool_history": list(self._tool_history),
            "latest_tool_call": dict(tool_entry),
        }

        try:
            logger.info("[TOOL EXECUTE] Calling %r tool_handlers...", tool_name)
            result = await self._tool_handlers(tool_name, tool_input, context_snapshot)
            logger.info("[TOOL SUCCESS] %r returned %d result keys", tool_name, len(result) if isinstance(result, dict) else 0)
        except Exception as exc:
            logger.error("[TOOL ERROR] %r raised %s: %s", tool_name, type(exc).__name__, exc, exc_info=True)
            result = {"error": str(exc)}

        tool_entry["result"] = result
        await self.metadata_queue.put({"type": "tool_result", "tool_name": tool_name, "result": result})

        # 3-part tool result protocol: contentStart(TOOL) → toolResult → contentEnd
        tool_content_name = str(uuid.uuid4())
        logger.info("[TOOL RESULT PROTOCOL] Sending 3-part response for %r (tool_use_id=%s)", tool_name, tool_use_id)
        
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self._prompt_id,
                    "contentName": tool_content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        })
        logger.info("  [1/3] Sent contentStart(TOOL)")
        
        result_json = json.dumps(result)
        logger.info("  [2/3] Sending toolResult with %d bytes of JSON", len(result_json))
        await self._send_event({
            "event": {
                "toolResult": {
                    "promptName": self._prompt_id,
                    "contentName": tool_content_name,
                    "content": result_json,
                }
            }
        })
        
        await self._send_event({
            "event": {
                "contentEnd": {
                    "promptName": self._prompt_id,
                    "contentName": tool_content_name,
                }
            }
        })
        logger.info("  [3/3] Sent contentEnd")
        logger.info("✓ [TOOL COMPLETE] %r → Nova will generate response with audio", tool_name)
        self._state = SessionState.LISTENING

    @property
    def state(self) -> SessionState:
        return self._state