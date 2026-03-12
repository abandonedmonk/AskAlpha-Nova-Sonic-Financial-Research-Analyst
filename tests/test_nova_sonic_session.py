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
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

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

    config_stub.settings = _SettingsStub()
    sys.modules["config"] = config_stub

from nova_sonic.session import NovaSonicSession, SessionState


@pytest.fixture
def session_with_mocks():
    """Build a session instance with a fully mocked NovaSonicClient."""
    mock_client = MagicMock()
    mock_client.open_stream.return_value = {
        "input_stream": MagicMock(),
        "body": MagicMock(),
    }
    mock_client.build_session_start_event.return_value = {"event": {"sessionStart": {}}}
    mock_client.build_audio_input_start_event.return_value = {
        "event": {"promptStart": {"promptId": "prompt-id"}}
    }
    mock_client.build_audio_chunk_event.return_value = {
        "event": {"audioInput": {"content": "audio-b64"}}
    }
    mock_client.build_tool_result_event.side_effect = (
        lambda prompt_id, tool_use_id, result: {
            "event": {
                "toolResult": {
                    "promptId": prompt_id,
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
    """Run async session APIs in plain pytest without pytest-asyncio."""
    return asyncio.run(coro)


def test_start_transitions_to_listening_and_sends_start_events(
    session_with_mocks,
):
    session, mock_client, _ = session_with_mocks
    session._send_event = AsyncMock()

    fake_task = MagicMock()
    fake_task.done.return_value = False

    def _fake_create_task(coro):
        # Prevent "coroutine was never awaited" warnings in this unit test.
        coro.close()
        return fake_task

    with patch(
        "nova_sonic.session.asyncio.create_task", side_effect=_fake_create_task
    ) as create_task:
        _run(session.start())

    assert session.state == SessionState.LISTENING
    assert session._send_event.await_count == 2
    session._send_event.assert_any_await(
        mock_client.build_session_start_event.return_value
    )
    session._send_event.assert_any_await(
        mock_client.build_audio_input_start_event.return_value
    )
    create_task.assert_called_once()


def test_send_audio_chunk_drops_audio_while_tool_executing(session_with_mocks):
    session, mock_client, _ = session_with_mocks
    session._state = SessionState.TOOL_EXECUTING
    session._send_event = AsyncMock()

    _run(session.send_audio_chunk(b"\x00\x01\x02"))

    session._send_event.assert_not_awaited()
    mock_client.build_audio_chunk_event.assert_not_called()


def test_send_audio_chunk_sends_event_while_listening(session_with_mocks):
    session, mock_client, _ = session_with_mocks
    session._state = SessionState.LISTENING
    session._send_event = AsyncMock()

    _run(session.send_audio_chunk(b"abc"))

    mock_client.build_audio_chunk_event.assert_called_once_with(
        session._prompt_id,
        session._content_id,
        base64.b64encode(b"abc").decode("utf-8"),
    )
    session._send_event.assert_awaited_once_with(
        mock_client.build_audio_chunk_event.return_value
    )


def test_handle_output_event_audio_output_enqueues_pcm_and_sets_speaking(
    session_with_mocks,
):
    session, _, _ = session_with_mocks
    pcm = b"pcm-bytes"
    audio_b64 = base64.b64encode(pcm).decode("utf-8")

    _run(session._handle_output_event({"audioOutput": {"content": audio_b64}}))

    assert _run(session.audio_output_queue.get()) == pcm
    assert session.state == SessionState.SPEAKING


def test_handle_output_event_generation_complete_sets_listening(
    session_with_mocks,
):
    session, _, _ = session_with_mocks
    session._state = SessionState.SPEAKING

    _run(session._handle_output_event({"generationComplete": {}}))

    assert session.state == SessionState.LISTENING


def test_handle_tool_use_success_sends_tool_result_and_returns_to_listening(
    session_with_mocks,
):
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
    mock_client.build_tool_result_event.assert_called_once()
    session._send_event.assert_awaited_once()
    assert session.state == SessionState.LISTENING


def test_handle_tool_use_exception_returns_error_payload(session_with_mocks):
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

    _, _, result_arg = mock_client.build_tool_result_event.call_args.args
    assert "error" in result_arg
    assert "tool exploded" in result_arg["error"]
    session._send_event.assert_awaited_once()
    assert session.state == SessionState.LISTENING


def test_close_closes_stream_and_cancels_consumer_task(session_with_mocks):
    session, _, _ = session_with_mocks

    body_mock = MagicMock()
    session._stream = {"input_stream": MagicMock(), "body": body_mock}

    fake_task = MagicMock()
    fake_task.done.return_value = False
    session._consumer_task = fake_task
    session._state = SessionState.LISTENING

    _run(session.close())

    body_mock.close.assert_called_once()
    fake_task.cancel.assert_called_once()
    assert session.state == SessionState.CLOSED


def test_consume_output_dispatches_events_from_stream_chunks(session_with_mocks):
    session, _, _ = session_with_mocks

    event = {"generationComplete": {}}
    session._stream = {
        "input_stream": MagicMock(),
        "body": [{"chunk": {"bytes": json.dumps(event).encode("utf-8")}}],
    }
    session._state = SessionState.LISTENING
    session._handle_output_event = AsyncMock()

    _run(session._consume_output())

    session._handle_output_event.assert_awaited_once_with(event)
    assert session.state == SessionState.CLOSED
