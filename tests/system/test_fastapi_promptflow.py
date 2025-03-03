from promptflow.core._serving.app import create_app
from dotenv import load_dotenv, find_dotenv
import os

import pytest
import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from typing import AsyncGenerator

from promptflow_tool_semantic_kernel.tools.semantic_kernel_tool import semantic_kernel_chat
import promptflow.tracing._tracer as tracer


@pytest.fixture
def client():
    load_dotenv(find_dotenv())
    flow_file_path = os.path.dirname(__file__)
    app = create_app(engine="fastapi", flow_file_path=flow_file_path)
    return TestClient(app)


class MockContentChunk:

    def __init__(self, content):
        self.content = content


async def mock_stream_generator() -> AsyncGenerator[MockContentChunk, None]:
    """Mock async generator to simulate streaming response."""
    chunks = [
        "", "Hello", "!", " How", " can", " I", " assist", " you", " today",
        " ?", ""
    ]
    for chunk in chunks:
        yield MockContentChunk(chunk)


def test_streaming_response(client):
    """Test the streaming response functionality of the FastAPI endpoint."""
    # Create a mock for the underlying AzureChatCompletion streaming method
    with patch(
            'semantic_kernel.connectors.ai.open_ai.AzureChatCompletion.get_streaming_chat_message_content',
            return_value=mock_stream_generator()):
        # Send request with streaming header
        response = client.post("/score",
                               json={
                                   "question": "Hello",
                                   "chat_history": []
                               },
                               headers={"Accept": "text/event-stream"})

        # Verify response status and headers
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["Content-Type"]

        # Parse the streaming response content
        content = response.content.decode('utf-8')

        # Process the Server-Sent Events format
        chunks = []
        for line in content.split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    chunks.append(data)
                except json.JSONDecodeError:
                    pass

        # Verify we received the expected streaming chunks
        expected_chunks = [{
            "answer": ""
        }, {
            "answer": "Hello"
        }, {
            "answer": "!"
        }, {
            "answer": " How"
        }, {
            "answer": " can"
        }, {
            "answer": " I"
        }, {
            "answer": " assist"
        }, {
            "answer": " you"
        }, {
            "answer": " today"
        }, {
            "answer": " ?"
        }, {
            "answer": ""
        }]

        assert chunks == expected_chunks


def test_streaming_response_with_tracing_enabled(client):
    """Test the streaming response functionality with tracing enabled."""
    # Patch the tracing function directly to enable tracing
    with patch('promptflow.tracing._utils.is_tracing_disabled', return_value=False), \
         patch('promptflow.tracing._trace.Tracer.push') as mock_push, \
         patch('promptflow.tracing._trace.start_as_current_span') as mock_span, \
         patch('promptflow.tracing._trace.format_trace_id', return_value='1234567890abcdef'):

        # Create a mock for the underlying AzureChatCompletion streaming method
        with patch(
                'semantic_kernel.connectors.ai.open_ai.AzureChatCompletion.get_streaming_chat_message_content',
                return_value=mock_stream_generator()):

            # Send request with streaming header
            response = client.post("/score",
                                   json={
                                       "question": "Hello",
                                       "chat_history": []
                                   },
                                   headers={"Accept": "text/event-stream"})

            # Verify response status and headers
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["Content-Type"]

            # Verify tracing was enabled
            assert mock_push.called, "Tracer.push was not called, which means tracing didn't work"
            assert mock_span.called, "start_as_current_span was not called, which means spans weren't created"

            # Parse and verify the streaming response content as before
            content = response.content.decode('utf-8')
            chunks = []
            for line in content.split('\n'):
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        chunks.append(data)
                    except json.JSONDecodeError:
                        pass

            expected_chunks = [{
                "answer": ""
            }, {
                "answer": "Hello"
            }, {
                "answer": "!"
            }, {
                "answer": " How"
            }, {
                "answer": " can"
            }, {
                "answer": " I"
            }, {
                "answer": " assist"
            }, {
                "answer": " you"
            }, {
                "answer": " today"
            }, {
                "answer": " ?"
            }, {
                "answer": ""
            }]

            assert chunks == expected_chunks
