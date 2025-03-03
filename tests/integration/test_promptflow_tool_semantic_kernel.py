import os
import pytest
from unittest.mock import patch, MagicMock
import asyncio
from typing import AsyncGenerator
from promptflow.contracts.types import PromptTemplate
from promptflow.connections import CustomConnection
from promptflow_tool_semantic_kernel.tools.semantic_kernel_tool import semantic_kernel_chat
from promptflow_tool_semantic_kernel.tools.semantic_kernel_tool import semantic_kernel_chat


# Mock streaming response generator
async def mock_streaming_generator() -> AsyncGenerator[str, None]:
    chunks = ["Hello, ", "I am ", "a mock ", "streaming ", "response!"]
    for chunk in chunks:
        await asyncio.sleep(0.01)  # Small delay to simulate real streaming
        yield chunk


@pytest.mark.asyncio
@patch(
    "promptflow_tool_semantic_kernel.tools.kernel_factory.KernelFactory.create_kernel"
)
@patch(
    "promptflow_tool_semantic_kernel.tools.response_strategy.ResponseStrategy.get_streaming_response"
)
async def test_integration(mock_get_streaming, mock_create_kernel):
    # Import here to avoid circular imports in test discovery

    # Setup mocks
    mock_kernel = MagicMock()
    mock_chat_completion = MagicMock()
    mock_create_kernel.return_value = (mock_kernel, mock_chat_completion)
    mock_get_streaming.return_value = mock_streaming_generator()

    # Create test inputs
    connection = CustomConnection(secrets={"api_key": "test_api_key"},
                                  configs={
                                      "base_url":
                                      "https://test.openai.azure.com/",
                                      "api_type": "azure",
                                      "api_version": "2023-07-01-preview"
                                  })

    chat_history = [{
        "role": "user",
        "content": "Hello"
    }, {
        "role": "assistant",
        "content": "Hi there!"
    }]

    prompt = PromptTemplate("Tell me about {{topic}}")

    # Call the tool with tracing enabled
    result = semantic_kernel_chat(connection=connection,
                                  deployment_name="test-deployment",
                                  chat_history=chat_history,
                                  prompt=prompt,
                                  streaming=True,
                                  topic="AI")

    # Collect streaming chunks
    response_chunks = []
    async for chunk in result:
        response_chunks.append(chunk)

    # Verify results
    assert len(response_chunks) == 5
    assert "".join(response_chunks) == "Hello, I am a mock streaming response!"
    mock_create_kernel.assert_called_once()
    mock_get_streaming.assert_called_once()
