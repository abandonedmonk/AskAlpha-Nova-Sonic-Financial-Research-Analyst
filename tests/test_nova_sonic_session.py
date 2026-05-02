"""
test_nova_sonic_session.py - Unit tests for nova_sonic/session.py state machine.

Run: pytest tests/test_nova_sonic_session.py -v
"""

import asyncio
import base64
import json
import sys
import types
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub heavy AWS SDK dependencies BEFORE nova_sonic imports happen.
# ---------------------------------------------------------------------------

if "aws_sdk_bedrock_runtime" not in sys.modules:
    _aws_mod = types.ModuleType("aws_sdk_bedrock_runtime")
    _aws_client_mod = types.ModuleType("aws_sdk_bedrock_runtime.client")
    _aws_models_mod = types.ModuleType("aws_sdk_bedrock_runtime.models")
    _aws_config_mod = types.ModuleType("aws_sdk_bedrock_runtime.config")
    _aws_client_mod.BedrockRuntimeClient = MagicMock()
    _aws_models_mod.BidirectionalInputPayloadPart = MagicMock()
    _aws_models_mod.InvokeModelWithBidirectionalStreamInputChunk = MagicMock()
    _aws_models_mod.InvokeModelWithBidirectionalStreamOperationInput = MagicMock()
    _aws_config_mod.Config = MagicMock()
    sys.modules["aws_sdk_bedrock_runtime"] = _aws_mod
    sys.modules["aws_sdk_bedrock_runtime.client"] = _aws_client_mod
    sys.modules["aws_sdk_bedrock_runtime.models"] = _aws_models_mod
    sys.modules["aws_sdk_bedrock_runtime.config"] = _aws_config_mod

if "smithy_aws_core" not in sys.modules:
    _smithy_mod = types.ModuleType("smithy_aws_core")
    _smithy_id_mod = types.ModuleType("smithy_aws_core.identity")
    _smithy_env_mod = types.ModuleType("smithy_aws_core.identity.environment")
    _smithy_env_mod.EnvironmentCredentialsResolver = MagicMock()
    sys.modules["smithy_aws_core"] = _smithy_mod
    sys.modules["smithy_aws_core.identity"] = _smithy_id_mod
    sys.modules["smithy_aws_core.identity.environment"] = _smithy_env_mod

if "config" not in sys.modules:
    _config_stub = types.ModuleType("config")

    class _SettingsStub:
        aws_access_key_id = "test-key"
        aws_secret_access_key = "test-secret"
        aws_region = "us-east-1"
        nova_sonic_model_id = "amazon.nova-sonic-v1:0"
        finnhub_api_key = ""
        polygon_api_key = ""
        tiingo_api_key = ""
        bedrock_kb_id = ""
        bedrock_kb_model_arn = ""
        vault_path = Path("./vault")
        log_level = "INFO"
        app_host = "0.0.0.0"
        app_port = 8000

        @property
        def ironclad_available(self) -> bool:
            return False

    _config_stub.settings = _SettingsStub()
    sys.modules["config"] = _config_stub

from nova_sonic.session import NovaSonicSession, SessionState  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def session_with_mocks():
    """Build a session with a fully mocked NovaSonicClient."""
    mock_client = MagicMock()

    # open_stream returns an object with input_stream and await_output()
    mock_stream = MagicMock()
    mock_stream.input_stream = AsyncMock()
    mock_client.open_stream = AsyncMock(return_value=mock_stream)

    # Builder stubs — use promptName (current AWS SDK v2 naming)
    mock_client.build_session_start_event.return_value = {
        "event": {"sessionStart": {}}
    }
    mock_client.build_audio_input_start_event.return_value = {
        "event": {"promptStart": {"promptName": "prompt-id"}}
    }
    mock_client.build_audio_chunk_event.return_value = {
        "event": {"audioInput": {"content": "audio-b64"}}
    }
    mock_client.build_system_prompt_start_event.return_value = {
        "event": {"contentStart": {}}
    }
    mock_client.build_system_prompt_text_event.return_value = {
        "event": {"textInput": {}}
    }
    mock_client.build_content_end_event.return_value = {
        "event": {"contentEnd": {}}
    }

    # build_tool_result_event kept for reference (not called by _handle_tool_use anymore)
    mock_client.build_tool_result_event.side_effect = (
        lambda prompt_name, tool_use_id, result: {
            "event": {
                "toolResult": {
                    "promptName": prompt_name,
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(result)}],
                    "status": "success",
                }
            }
        }
    )

    tool_handler = AsyncMock(return_value={"summary": "ok"})

    with patch("nova_sonic.session.NovaSonicClient", return_value=mock_client):
        session = NovaSonicSession(tool_handlers=tool_handler)

    return session, mock_client, tool_handler


def _run(coro):
    """Run async session APIs in plain pytest (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_start_transitions_to_listening_and_sends_start_events(session_with_mocks):
    """
    session.start() must:
      1. open the stream
      2. start the consumer task
      3. send sessionStart + promptStart + 3 system-prompt events (5 total)
      4. transition to LISTENING
    """
    session, mock_client, _ = session_with_mocks
    session._send_event = AsyncMock()

    fake_task = MagicMock()
    fake_task.done.return_value = False

    def _fake_create_task(coro):
        coro.close()
        return fake_task

    with patch("nova_sonic.session.asyncio.create_task", side_effect=_fake_create_task):
        _run(session.start())

    assert session.state == SessionState.LISTENING
    # sessionStart + promptStart + sys_contentStart + sys_textInput + sys_contentEnd = 5 events
    assert session._send_event.await_count == 5
    session._send_event.assert_any_await(
        mock_client.build_session_start_event.return_value
    )
    session._send_event.assert_any_await(
        mock_client.build_audio_input_start_event.return_value
    )


def test_send_audio_chunk_drops_audio_while_tool_executing(session_with_mocks):
    """Audio is blocked when state is TOOL_EXECUTING AND _audio_content_id is absent."""
    session, mock_client, _ = session_with_mocks
    session._state = SessionState.TOOL_EXECUTING
    session._send_event = AsyncMock()
    # Do NOT set _audio_content_id — the guard requires it to exist

    _run(session.send_audio_chunk(b"\x00\x01\x02"))

    session._send_event.assert_not_awaited()
    mock_client.build_audio_chunk_event.assert_not_called()


def test_send_audio_chunk_sends_event_while_listening(session_with_mocks):
    """Audio chunk is encoded and sent inline (not via build_audio_chunk_event)."""
    session, mock_client, _ = session_with_mocks
    session._state = SessionState.LISTENING
    # session.py sends the event inline using promptName/contentName directly
    audio_content_id = "test-audio-content-id"
    session._audio_content_id = audio_content_id
    session._send_event = AsyncMock()

    _run(session.send_audio_chunk(b"abc"))

    expected_b64 = base64.b64encode(b"abc").decode("utf-8")
    session._send_event.assert_awaited_once_with({
        "event": {
            "audioInput": {
                "promptName": session._prompt_id,
                "contentName": audio_content_id,
                "content": expected_b64,
            }
        }
    })


def test_handle_output_event_audio_output_enqueues_pcm_and_sets_speaking(
    session_with_mocks,
):
    session, _, _ = session_with_mocks
    pcm = b"pcm-bytes"
    audio_b64 = base64.b64encode(pcm).decode("utf-8")

    _run(session._handle_output_event({"audioOutput": {"content": audio_b64}}))

    assert _run(session.audio_output_queue.get()) == pcm
    assert session.state == SessionState.SPEAKING


def test_handle_output_event_generation_complete_sets_listening(session_with_mocks):
    session, _, _ = session_with_mocks
    session._state = SessionState.SPEAKING

    _run(session._handle_output_event({"generationComplete": {}}))

    assert session.state == SessionState.LISTENING


def test_handle_tool_use_success_sends_tool_result_and_returns_to_listening(
    session_with_mocks,
):
    """
    _handle_tool_use must call tool_handler, send the 3-part protocol
    (contentStart + toolResult + contentEnd) and return to LISTENING.
    """
    session, mock_client, tool_handler = session_with_mocks
    session._send_event = AsyncMock()

    _run(
        session._handle_tool_use(
            {
                "name": "query_live_market_data",
                "toolUseId": "tool-1",
                "input": {"ticker": "NVDA"},
            }
        )
    )

    tool_handler.assert_awaited_once_with(
        "query_live_market_data",
        {"ticker": "NVDA"},
        ANY,
    )
    # session.py now sends 3 events inline (contentStart, toolResult, contentEnd)
    assert session._send_event.await_count == 3
    assert session.state == SessionState.LISTENING


def test_handle_tool_use_exception_returns_error_payload(session_with_mocks):
    """On tool failure, an error dict is sent as the tool result."""
    session, mock_client, tool_handler = session_with_mocks
    tool_handler.side_effect = RuntimeError("tool exploded")
    session._send_event = AsyncMock()

    _run(
        session._handle_tool_use(
            {
                "name": "execute_quantitative_model",
                "toolUseId": "tool-2",
                "input": {"ticker": "NVDA"},
            }
        )
    )

    # The toolResult event (2nd of 3) contains the error as JSON in "content"
    calls = session._send_event.await_args_list
    assert len(calls) == 3
    tool_result_event = calls[1].args[0]["event"]["toolResult"]
    result_payload = json.loads(tool_result_event["content"])
    assert "error" in result_payload
    assert "tool exploded" in result_payload["error"]
    assert session.state == SessionState.LISTENING


def test_close_closes_stream_and_cancels_consumer_task(session_with_mocks):
    """close() must cancel the consumer task and close input_stream."""
    session, _, _ = session_with_mocks

    session._stream = MagicMock()
    session._stream.input_stream.close = AsyncMock()
    session._send_event = AsyncMock()

    async def _test_close():
        # Create a real asyncio.Future to emulate the consumer task.
        # Must be created inside an async context so it attaches to the loop.
        fake_task = asyncio.Future()
        session._consumer_task = fake_task
        session._state = SessionState.LISTENING

        await session.close()

        assert fake_task.cancelled() is True
        assert session.state == SessionState.CLOSED

    _run(_test_close())


def test_consume_output_dispatches_events_from_stream_chunks(session_with_mocks):
    """
    _consume_output reads via await_output() and dispatches each decoded event
    to _handle_output_event.
    """
    session, _, _ = session_with_mocks

    inner_event = {"generationComplete": {}}
    full_event = {"event": inner_event}
    event_bytes = json.dumps(full_event).encode("utf-8")

    # Build the AWS SDK v2 await_output() chain:
    # stream.await_output() -> (_, receiver); receiver.receive() -> result
    mock_result = MagicMock()
    mock_result.value = MagicMock()
    mock_result.value.bytes_ = event_bytes

    mock_receiver = AsyncMock()
    mock_receiver.receive = AsyncMock(return_value=mock_result)

    async def _fake_await_output():
        return (None, mock_receiver)

    session._stream = MagicMock()
    session._stream.await_output = _fake_await_output
    session._state = SessionState.LISTENING

    # After the first event is dispatched, flip state to CLOSED so the
    # `while self._state != SessionState.CLOSED` guard exits cleanly.
    # This avoids relying on CancelledError propagation through asyncio.wait_for
    # which behaves differently across Python versions.
    async def _handle_and_close(ev):
        session._state = SessionState.CLOSED

    session._handle_output_event = AsyncMock(side_effect=_handle_and_close)

    _run(session._consume_output())

    session._handle_output_event.assert_awaited_once_with(inner_event)
