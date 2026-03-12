"""
test_nova_sonic_client.py - Unit tests for nova_sonic/client.py event builders.

Run: pytest tests/test_nova_sonic_client.py -v
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = MagicMock()
    sys.modules["boto3"] = boto3_stub

if "botocore.exceptions" not in sys.modules:
    botocore_stub = types.ModuleType("botocore")
    botocore_exceptions_stub = types.ModuleType("botocore.exceptions")

    class _ClientError(Exception):
        pass

    botocore_exceptions_stub.ClientError = _ClientError
    sys.modules["botocore"] = botocore_stub
    sys.modules["botocore.exceptions"] = botocore_exceptions_stub

if "config" not in sys.modules:
    config_stub = types.ModuleType("config")

    class _SettingsStub:
        aws_access_key_id = "test-key"
        aws_secret_access_key = "test-secret"
        aws_region = "us-east-1"
        nova_sonic_model_id = "amazon.nova-sonic-v1:0"

    config_stub.settings = _SettingsStub()
    sys.modules["config"] = config_stub

from nova_sonic.client import NovaSonicClient
from nova_sonic.tool_schemas import ALL_TOOLS


def _build_client() -> NovaSonicClient:
    """Create client with boto3 mocked to avoid real AWS setup in unit tests."""
    with patch("nova_sonic.client.boto3.client", return_value=MagicMock()):
        return NovaSonicClient()


def test_build_session_start_event_contains_expected_sections() -> None:
    client = _build_client()

    event = client.build_session_start_event(system_prompt="Test prompt")

    session_start = event["event"]["sessionStart"]
    assert session_start["systemPrompt"]["text"] == "Test prompt"
    assert session_start["toolConfiguration"]["tools"] == ALL_TOOLS

    inference = session_start["inferenceConfiguration"]
    assert inference["maxTokens"] == 1024
    assert inference["topP"] == 0.9
    assert inference["temperature"] == 0.7


def test_build_audio_input_start_event_uses_expected_audio_formats() -> None:
    event = NovaSonicClient.build_audio_input_start_event("prompt-1", "content-1")

    prompt_start = event["event"]["promptStart"]
    assert prompt_start["promptId"] == "prompt-1"

    in_audio = prompt_start["inputConfiguration"]["audio"]
    assert in_audio["mediaType"] == "audio/lpcm"
    assert in_audio["sampleRateHertz"] == 16000
    assert in_audio["sampleSizeBits"] == 16
    assert in_audio["channelCount"] == 1
    assert in_audio["audioType"] == "SPEECH"
    assert in_audio["encoding"] == "base64"

    out_audio = prompt_start["outputConfiguration"]["audio"]
    assert out_audio["mediaType"] == "audio/lpcm"
    assert out_audio["sampleRateHertz"] == 24000
    assert out_audio["sampleSizeBits"] == 16
    assert out_audio["channelCount"] == 1
    assert out_audio["voiceId"] == "matthew"


def test_build_audio_chunk_event_contains_prompt_content_and_payload() -> None:
    event = NovaSonicClient.build_audio_chunk_event("prompt-2", "content-2", "YWJj")

    audio_input = event["event"]["audioInput"]
    assert audio_input["promptId"] == "prompt-2"
    assert audio_input["contentId"] == "content-2"
    assert audio_input["content"] == "YWJj"


def test_build_tool_result_event_serializes_result_json() -> None:
    result = {"ticker": "NVDA", "price": 100.5}

    event = NovaSonicClient.build_tool_result_event("prompt-3", "tool-123", result)

    tool_result = event["event"]["toolResult"]
    assert tool_result["promptId"] == "prompt-3"
    assert tool_result["toolUseId"] == "tool-123"
    assert tool_result["status"] == "success"

    payload_text = tool_result["content"][0]["text"]
    assert json.loads(payload_text) == result
