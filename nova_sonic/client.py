"""
client.py — Low-level AWS Bedrock / Nova Sonic client wrapper.

Uses the generated AWS SDK v2 (aws-sdk-bedrock-runtime) for bidirectional
streaming via InvokeModelWithBidirectionalStream.

Reference:
  https://docs.aws.amazon.com/bedrock/latest/userguide/nova-sonic-overview.html
"""

import json
import logging
import os
from typing import Any

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from config import settings
from nova_sonic.tool_schemas import ALL_TOOLS

DEFAULT_NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"

logger = logging.getLogger(__name__)


class NovaSonicClient:
    """AWS SDK v2 wrapper for Nova Sonic bidirectional streaming."""

    def __init__(self) -> None:
        # Set environment variables from pydantic settings so AWS SDK can find them
        # (pydantic-settings loads .env into settings object, not os.environ)
        os.environ['AWS_ACCESS_KEY_ID'] = settings.aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = settings.aws_secret_access_key
        os.environ['AWS_REGION'] = settings.aws_region
        
        logger.info("✓ AWS credentials set in environment from .env")
        
        # Configure the generated SDK client
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{settings.aws_region}.amazonaws.com",
            region=settings.aws_region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self._client = BedrockRuntimeClient(config=config)
        self.model_id = getattr(settings, 'nova_sonic_model_id', DEFAULT_NOVA_SONIC_MODEL_ID)

    # ── Session start payload ─────────────────────────────────────────────────

    def build_session_start_event(
        self,
        system_prompt: str = (
            "You are a financial research assistant with access to four tools. "
            "You MUST use the provided tools whenever the answer requires market data, SEC filings, "
            "quantitative simulation, or saving notes. Do not claim you lack access when a tool can "
            "answer the request. Use query_live_market_data for stock prices and market moves. "
            "Use analyze_sec_filings_rag for questions about what a company said in a 10-K or 10-Q. "
            "Use execute_quantitative_model for Monte Carlo or forward price simulation requests. "
            "Use log_research_insight to save notes or summaries to the vault. "
            "After receiving tool results, answer concisely and naturally."
        ),
    ) -> dict[str, Any]:
        """
        Construct the sessionStart event that initialises a Nova Sonic stream.
        Injects all four financial tool schemas so the model can use them.
        """
        return {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    },
                    "systemPrompt": {"text": system_prompt},
                    "toolConfiguration": {"tools": ALL_TOOLS},
                }
            }
        }

    # ── Audio prompt helpers ──────────────────────────────────────────────────

    @staticmethod
    def build_audio_input_start_event(
        prompt_name: str, content_name: str
    ) -> dict[str, Any]:
        """Build promptStart event with correct API format (matches AWS test examples)."""
        return {
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "inputTranscriptionConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "toolConfiguration": {
                        "tools": ALL_TOOLS,
                        "toolChoice": {"auto": {}},
                    },
                }
            }
        }

    @staticmethod
    def build_audio_chunk_event(
        prompt_name: str, content_name: str, audio_b64: str
    ) -> dict[str, Any]:
        """Build event for a chunk of audio input."""
        return {
            "event": {
                "audioInput": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "content": audio_b64,
                }
            }
        }

    @staticmethod
    def build_tool_result_event(
        prompt_name: str, tool_use_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a tool result back into the active Nova Sonic stream."""
        return {
            "event": {
                "toolResult": {
                    "promptName": prompt_name,
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(result)}],
                    "status": "success",
                }
            }
        }

    # ── System prompt content blocks ──────────────────────────────────────────

    @staticmethod
    def build_system_prompt_start_event(
        prompt_name: str, content_name: str
    ) -> dict[str, Any]:
        """Start a system prompt content block."""
        return {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }

    @staticmethod
    def build_system_prompt_text_event(
        prompt_name: str, content_name: str, text: str
    ) -> dict[str, Any]:
        """Send system prompt text."""
        return {
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "content": text,
                }
            }
        }

    @staticmethod
    def build_content_end_event(
        prompt_name: str, content_name: str
    ) -> dict[str, Any]:
        """End a content block."""
        return {
            "event": {
                "contentEnd": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                }
            }
        }

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    async def open_stream(self) -> Any:
        """
        Open a bidirectional stream to Nova Sonic.
        Returns the stream object for async event handling.
        
        The caller is responsible for sending sessionStart event first.
        """
        try:
            from aws_sdk_bedrock_runtime.models import (
                InvokeModelWithBidirectionalStreamOperationInput,
            )
            logger.info("")
            logger.info("CLIENT: Invoking bidirectional model...")
            logger.info("  Model ID: %s", self.model_id)
            logger.info("  Region: us-east-1")
            
            stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            
            logger.info("")
            logger.info("✓ Bidirectional stream opened successfully!")
            logger.info("  Stream type: %s", type(stream).__name__)
            logger.info("  Stream.input_stream: %s", type(stream.input_stream).__name__ if hasattr(stream, 'input_stream') else 'N/A')
            logger.info("  Stream.output_stream: %s", type(stream.output_stream).__name__ if hasattr(stream, 'output_stream') else 'N/A')
            logger.info("")
            
            return stream
        except Exception as exc:
            logger.error("")
            logger.error("=" * 80)
            logger.error("✗ Failed to open Nova Sonic stream!")
            logger.error("=" * 80)
            logger.error("Exception type: %s", type(exc).__name__)
            logger.error("Exception message: %s", exc)
            import traceback
            logger.error("Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error("  %s", line)
            logger.error("=" * 80)
            logger.error("")
            raise

    async def send_event(self, stream: Any, event: dict[str, Any]) -> None:
        """Send a JSON event into the bidirectional stream."""
        try:
            event_json = json.dumps(event).encode("utf-8")
            payload_part = BidirectionalInputPayloadPart(bytes_=event_json)
            chunk = InvokeModelWithBidirectionalStreamInputChunk(value=payload_part)
            logger.debug("Sending event: %s (size=%d bytes)", list(event.get("event", {}).keys()), len(event_json))
            await stream.input_stream.send(chunk)
            logger.debug("✓ Event sent successfully")
        except Exception as exc:
            logger.error("Error sending event to Nova Sonic: %s | event_keys=%s", exc, list(event.get("event", {}).keys()))
            import traceback
            logger.error("Send event traceback: %s", traceback.format_exc())
            raise
