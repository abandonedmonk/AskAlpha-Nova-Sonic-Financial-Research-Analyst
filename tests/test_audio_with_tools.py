"""
End-to-end integration test for Nova Sonic 2 with financial tools and bidirectional audio streaming.

Verifies that voice commands trigger tool execution and return synthesized audio responses.

Run: python tests/test_audio_with_tools.py
"""

import asyncio
import base64
import json
import sys
import uuid
from pathlib import Path

import pyaudio
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from nova_sonic.tool_schemas import ALL_TOOLS
from event_router.router import dispatch

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Audio configuration constants
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 15


class AudioToolsTest:
    """Integration test for Nova Sonic 2 with tool calling and bidirectional audio."""

    def __init__(self, model_id="amazon.nova-2-sonic-v1:0", region="us-east-1"):
        self.model_id = model_id
        self.region = region
        self.client = None
        self.stream = None
        self.is_active = False

        # Session identifiers
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        # State tracking
        self.audio_queue = asyncio.Queue()
        self.tools_called = []
        self.current_role = None
        self.pending_tool_use = None
        self.audio_chunks_received = 0
        self.user_utterance_parts = []
        self.tool_requested = asyncio.Event()
        self.response_completed = asyncio.Event()
        self.tool_result_sent = asyncio.Event()
        self.pending_tool_task = None
        self.last_event_ts = 0.0
        
    def _initialize_client(self):
        """Initialize Bedrock Runtime client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
    
    async def send_event(self, event_json: str) -> None:
        """Send JSON event to bidirectional stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)
    
    async def start_session(self) -> None:
        """Initialize bidirectional stream with Nova Sonic 2 and configure tools + audio."""
        if not self.client:
            self._initialize_client()

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        self.last_event_ts = asyncio.get_event_loop().time()

        # Step 1: Session initialization
        session_start = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    }
                }
            }
        }
        await self.send_event(json.dumps(session_start))

        # Step 2: Prompt start with tool and audio configuration
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

        prompt_start = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {
                        "tools": ALL_TOOLS,
                        "toolChoice": {"auto": {}},
                    },
                }
            }
        }
        await self.send_event(json.dumps(prompt_start))

        # Step 3: Send system prompt as first content block (required by API protocol)
        system_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }
        await self.send_event(json.dumps(system_content_start))

        system_text_input = {
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": system_prompt,
                }
            }
        }
        await self.send_event(json.dumps(system_text_input))

        system_content_end = {
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.content_name}}
        }
        await self.send_event(json.dumps(system_content_end))

        print("\n" + "=" * 80)
        print("SESSION READY - AWAITING USER INPUT")
        print("=" * 80)

        asyncio.create_task(self._process_responses())
    
    async def start_audio_input(self) -> None:
        """Begin audio input stream from microphone."""
        self.user_utterance_parts = []

        audio_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }
        await self.send_event(json.dumps(audio_content_start))

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Send audio chunk to stream."""

        if not self.is_active:
            return

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_event = {
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": audio_b64,
                }
            }
        }
        await self.send_event(json.dumps(audio_event))
    
    async def end_audio_input(self) -> None:
        """Complete audio input and signal Nova to process."""

        audio_content_end = {
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}}
        }
        await self.send_event(json.dumps(audio_content_end))

        if self.tool_requested.is_set():
            if not self.tool_result_sent.is_set():
                try:
                    await asyncio.wait_for(self.tool_result_sent.wait(), timeout=20)
                except asyncio.TimeoutError:
                    pass
            return

        # Trigger response generation
        prompt_end = {"event": {"promptEnd": {"promptName": self.prompt_name}}}
        await self.send_event(json.dumps(prompt_end))
    
    async def _process_responses(self) -> None:
        """Process streaming events from Nova Sonic."""
        
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()

                if not result.value or not result.value.bytes_:
                    continue

                response_data = result.value.bytes_.decode("utf-8")
                json_data = json.loads(response_data)
                self.last_event_ts = asyncio.get_event_loop().time()

                if "event" not in json_data:
                    continue

                event_dict = json_data["event"]
                if not event_dict:
                    continue

                # Tool invocation
                if "toolUse" in event_dict:
                    self.pending_tool_use = event_dict["toolUse"]
                    self.tool_requested.set()

                # User transcript
                elif "textOutput" in event_dict and self.current_role == "USER":
                    self.user_utterance_parts.append(event_dict["textOutput"]["content"])

                # Assistant response text
                elif "textOutput" in event_dict and self.current_role == "ASSISTANT":
                    text = event_dict["textOutput"]["content"]
                    print(f"[RESPONSE] {text}")

                # Role change tracking
                elif "contentStart" in event_dict:
                    self.current_role = event_dict["contentStart"].get("role")

                # Tool result handling
                elif "contentEnd" in event_dict:
                    if self.current_role == "TOOL" and self.pending_tool_use:
                        self.tool_result_sent.clear()
                        self.pending_tool_task = asyncio.create_task(
                            self._handle_tool_use(self.pending_tool_use)
                        )
                        self.pending_tool_use = None

                # Audio output
                elif "audioOutput" in event_dict:
                    audio_content = event_dict["audioOutput"]["content"]
                    audio_bytes = base64.b64decode(audio_content)
                    self.audio_chunks_received += 1
                    if self.audio_chunks_received == 1:
                        print(f"[AUDIO] Generating synthesized speech...")
                    await self.audio_queue.put(audio_bytes)

                # Completion markers
                elif "generationComplete" in event_dict or "completionEnd" in event_dict:
                    self.response_completed.set()

        except Exception as e:
            if self.is_active:
                print(f"Error processing responses: {e}")
                import traceback

                traceback.print_exc()
    
    async def _handle_tool_use(self, tool_event: dict) -> None:
        """Execute tool and send result back to Nova."""
        tool_name = tool_event.get("name") or tool_event.get("toolName")
        tool_use_id = tool_event["toolUseId"]

        raw_tool_input = tool_event.get("input") or tool_event.get("content", {})
        if isinstance(raw_tool_input, str):
            try:
                tool_input = json.loads(raw_tool_input)
            except json.JSONDecodeError:
                tool_input = {"raw_input": raw_tool_input}
        elif isinstance(raw_tool_input, dict):
            tool_input = raw_tool_input
        else:
            tool_input = {}

        if not tool_name:
            raise ValueError(f"Unsupported toolUse payload: {json.dumps(tool_event)}")

        # Enhance tool 2 input quality using transcript
        if tool_name == "analyze_sec_filings_rag":
            transcript = " ".join(self.user_utterance_parts).lower()
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

        # Execute tool
        try:
            result = await dispatch(tool_name, tool_input)
        except Exception as e:
            result = {"error": str(e)}

        # Log tool execution
        print(f"\n[TOOL] {tool_name}")
        print(f"  Input: {json.dumps(tool_input, indent=2)}")
        self.tools_called.append(tool_name)

        # Send result back: contentStart -> toolResult -> contentEnd
        tool_content_name = str(uuid.uuid4())

        tool_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
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
        }
        await self.send_event(json.dumps(tool_content_start))

        # Log retrieved data
        if "error" not in result:
            print(f"  Retrieved: {json.dumps(result, indent=2)[:500]}")
            if len(json.dumps(result, indent=2)) > 500:
                print(f"  ... (truncated)")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

        tool_result = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": tool_content_name,
                    "content": json.dumps(result),
                }
            }
        }
        await self.send_event(json.dumps(tool_result))

        tool_content_end = {
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": tool_content_name}}
        }
        await self.send_event(json.dumps(tool_content_end))
        self.tool_result_sent.set()

        # Wait for response generation
        await asyncio.sleep(0.2)
        loop = asyncio.get_event_loop()
        start_wait = loop.time()
        while loop.time() - start_wait < 15:
            if self.response_completed.is_set() or self.audio_chunks_received > 0:
                return
            await asyncio.sleep(0.1)

    async def wait_for_turn_end(self, timeout: int = 60, quiet_seconds: float = 2.0) -> bool:
        """Wait for turn completion by event completion or stream quiescence."""
        loop = asyncio.get_event_loop()
        start = loop.time()

        while loop.time() - start < timeout:
            if self.response_completed.is_set():
                return True

            no_pending_tool = self.pending_tool_task is None or self.pending_tool_task.done()
            no_pending_audio = self.audio_queue.empty()
            quiet = (loop.time() - self.last_event_ts) >= quiet_seconds

            if self.tool_requested.is_set() and no_pending_tool and no_pending_audio and quiet:
                return True

            await asyncio.sleep(0.2)

        return False
    
    async def play_audio(self) -> None:
        """Play audiooutput from queue through speakers."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
        )

        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                stream.write(audio_data)
        except Exception as e:
            if self.is_active:
                print(f"Playback error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def capture_audio(self) -> None:
        """Capture audio from microphone and send to stream."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        await self.start_audio_input()

        try:
            chunks_to_record = int(INPUT_SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)
            for i in range(chunks_to_record):
                if self.tool_requested.is_set():
                    break

                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.001)

        finally:
            await self.end_audio_input()
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    async def close(self) -> None:
        """Cleanup and close session."""
        if not self.is_active:
            return

        self.is_active = False

        try:
            session_end = {"event": {"sessionEnd": {}}}
            await self.send_event(json.dumps(session_end))
            await self.stream.input_stream.close()
        except Exception:
            pass

        # Log summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Tools called: {len(self.tools_called)}")
        print(f"Audio chunks received: {self.audio_chunks_received}")
        if self.tools_called:
            for tool in self.tools_called:
                print(f"   {tool}")
        else:
            print("  (No tools were called)")


async def main() -> None:
    """Run integration test for Nova Sonic 2 tool calling with audio."""
    test = AudioToolsTest()

    try:
        await test.start_session()

        # Run audio capture and playback concurrently
        capture_task = asyncio.create_task(test.capture_audio())
        play_task = asyncio.create_task(test.play_audio())

        # Wait for user input to complete
        await capture_task

        # Wait for response generation
        await test.wait_for_turn_end(timeout=60, quiet_seconds=2.0)

        await test.close()

        play_task.cancel()
        try:
            await play_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        await test.close()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())