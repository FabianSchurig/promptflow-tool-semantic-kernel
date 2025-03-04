import promptflow_tool_semantic_kernel.tools.semantic_kernel_tool as semantic_kernel_tool
from promptflow.connections import CustomConnection
from dotenv import load_dotenv
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging
from unittest.mock import patch
from promptflow.contracts.types import PromptTemplate
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion


@pytest.fixture
def mock_connection():
    return CustomConnection({
        "api_key": "test_api_key",
        "api_base": "https://test.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    })


@pytest.fixture
def mock_chat_history():
    return [{
        "role": "user",
        "content": "Hello"
    }, {
        "role": "assistant",
        "content": "Hi there!"
    }]


@pytest.fixture
def mock_prompt():
    return PromptTemplate("Tell me about {{topic}}")


@pytest.mark.asyncio
@patch(
    "promptflow_tool_semantic_kernel.tools.kernel_factory.KernelFactory.create_kernel"
)
@patch(
    "promptflow_tool_semantic_kernel.tools.response_strategy.ResponseStrategy.get_complete_response"
)
async def test_semantic_kernel_chat_non_streaming(mock_get_response,
                                                  mock_create_kernel,
                                                  mock_connection,
                                                  mock_chat_history,
                                                  mock_prompt):
    # Setup mocks
    mock_kernel = MagicMock()
    mock_chat_completion = MagicMock()
    mock_create_kernel.return_value = (mock_kernel, mock_chat_completion)
    mock_get_response.return_value = "Test response"

    # Call function
    result = semantic_kernel_tool.semantic_kernel_chat(
        connection=mock_connection,
        deployment_name="test-deployment",
        chat_history=mock_chat_history,
        prompt=mock_prompt,
        plugins=[],
        streaming=False,
        topic="AI")

    response = [r async for r in result]

    # Assertions
    assert response[0] == "Test response"
    mock_create_kernel.assert_called_once()
    mock_get_response.assert_called_once()


@pytest.mark.asyncio
@patch(
    "promptflow_tool_semantic_kernel.tools.kernel_factory.KernelFactory.create_kernel"
)
@patch(
    "promptflow_tool_semantic_kernel.tools.response_strategy.ResponseStrategy.get_streaming_response"
)
async def test_semantic_kernel_chat_streaming(mock_get_streaming,
                                              mock_create_kernel,
                                              mock_connection,
                                              mock_chat_history, mock_prompt):
    # Setup mocks
    mock_kernel = MagicMock()
    mock_chat_completion = MagicMock()
    mock_create_kernel.return_value = (mock_kernel, mock_chat_completion)

    # Create an async generator function
    async def mock_streaming_gen():
        yield "Streaming response"

    mock_get_streaming.return_value = mock_streaming_gen()

    # Call function
    result = semantic_kernel_tool.semantic_kernel_chat(
        connection=mock_connection,
        deployment_name="test-deployment",
        chat_history=mock_chat_history,
        prompt=mock_prompt,
        plugins=[],
        streaming=True,
        topic="AI")

    response = [r async for r in result]

    # Assertions
    assert response[0] == "Streaming response"
    mock_create_kernel.assert_called_once()
    mock_get_streaming.assert_called_once()


@pytest.mark.asyncio
@patch(
    "promptflow_tool_semantic_kernel.tools.kernel_factory.KernelFactory.create_kernel"
)
async def test_semantic_kernel_chat_exception_handling(mock_create_kernel,
                                                       mock_connection,
                                                       mock_chat_history,
                                                       mock_prompt):
    # Setup mock to raise exception
    mock_create_kernel.side_effect = Exception("Test error")

    # Call function
    result = semantic_kernel_tool.semantic_kernel_chat(
        connection=mock_connection,
        deployment_name="test-deployment",
        chat_history=mock_chat_history,
        prompt=mock_prompt,
        plugins=[],
        topic="AI")

    response = [r async for r in result]

    # Assertions
    assert "An error occurred: Test error" in response[0]


@pytest.mark.asyncio
@patch(
    "promptflow_tool_semantic_kernel.tools.kernel_factory.KernelFactory.create_kernel"
)
@patch(
    "promptflow_tool_semantic_kernel.tools.chat_history_processor.ChatHistoryProcessor.build_history"
)
async def test_chat_history_processing(mock_build_history, mock_create_kernel,
                                       mock_connection, mock_chat_history,
                                       mock_prompt):
    # Setup mocks
    mock_kernel = MagicMock()
    mock_chat_completion = MagicMock()
    mock_create_kernel.return_value = (mock_kernel, mock_chat_completion)
    mock_chat_history_obj = MagicMock()
    mock_chat_history_obj.messages = []
    mock_build_history.return_value = mock_chat_history_obj

    # Patch the response strategy to avoid full execution
    async def mock_streaming_gen():
        yield "Streaming response"

    with patch(
            "promptflow_tool_semantic_kernel.tools.response_strategy.ResponseStrategy.get_streaming_response",
            return_value=mock_streaming_gen()):
        # Call function
        result = semantic_kernel_tool.semantic_kernel_chat(
            connection=mock_connection,
            deployment_name="test-deployment",
            chat_history=mock_chat_history,
            prompt=mock_prompt,
            plugins=[],
            streaming=True,
            topic="AI")

        response = [r async for r in result]

        # Assertions
        assert response[0] == "Streaming response"
        mock_build_history.assert_called_once()
