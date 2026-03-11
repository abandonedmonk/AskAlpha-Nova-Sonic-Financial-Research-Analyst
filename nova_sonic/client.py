"""
client.py — Low-level AWS Bedrock / Nova Sonic client wrapper.

Wraps boto3's Bedrock Runtime client and exposes the bidirectional
streaming invoke used by Nova Sonic (InvokeModelWithBidirectionalStream).

Reference:
  https://docs.aws.amazon.com/bedrock/latest/userguide/nova-sonic-overview.html
"""

import json
import logging
from typing import Any, AsyncGenerator

import boto3
from botocore.exceptions import ClientError

from config import settings
from nova_sonic.tool_schemas import ALL_TOOLS

# Nova 2 Sonic model ID (supports tool calling and improved speech quality)
_DEFAULT_NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"

logger = logging.getLogger(__name__)


class NovaSonicClient:
    """Thin wrapper around boto3's Bedrock Runtime for Nova Sonic streams."""

    def __init__(self) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        # Use Nova 2 Sonic if available (released Dec 2, 2025), else fall back to v1
        self.model_id = getattr(settings, 'nova_sonic_model_id', _DEFAULT_NOVA_SONIC_MODEL_ID)

    # ── Session start payload ─────────────────────────────────────────────────

    def build_session_start_event(
        self,
        system_prompt: str = (
            "You are an expert voice-driven financial research assistant. "
        "Answer concisely. When you call a tool, wait for the result before speaking."
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
        prompt_id: str, content_id: str
    ) -> dict[str, Any]:
        return {
            "event": {
                "promptStart": {
                    "promptId": prompt_id,
                    "inputConfiguration": {
                        "audio": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        }
                    },
                    "outputConfiguration": {
                        "audio": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 24000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": "matthew",  # change to preferred Nova Sonic voice
                        }
                    },
                }
            }
        }

    @staticmethod
    def build_audio_chunk_event(
        prompt_id: str, content_id: str, audio_b64: str
    ) -> dict[str, Any]:
        return {
            "event": {
                "audioInput": {
                    "promptId": prompt_id,
                    "contentId": content_id,
                    "content": audio_b64,
                }
            }
        }

    @staticmethod
    def build_tool_result_event(
        prompt_id: str, tool_use_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a tool result back into the active Nova Sonic stream."""
        return {
            "event": {
                "toolResult": {
                    "promptId": prompt_id,
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(result)}],
                    "status": "success",
                }
            }
        }

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    def open_stream(self) -> Any:
        """
        Open a bidirectional stream to Nova Sonic.
        Returns the raw boto3 stream handler — the session manager owns it.
        
        Note: The sessionStart event (sent first by the caller) configures
        inference parameters, system prompt, and tool schemas.
        """
        try:
            response = self._client.invoke_model_with_bidirectional_stream(
                modelId=self.model_id
            )
            return response
        except ClientError as exc:
            logger.error("Failed to open Nova Sonic stream: %s", exc)
            raise
