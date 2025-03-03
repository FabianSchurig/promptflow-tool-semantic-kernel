import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from collections.abc import AsyncGenerator
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from promptflow_tool_semantic_kernel.tools.response_strategy import ResponseStrategy


class TestResponseStrategy:

    @pytest.fixture
    def kernel(self):
        return MagicMock(spec=Kernel)

    @pytest.fixture
    def chat_completion(self):
        return AsyncMock(spec=AzureChatCompletion)

    @pytest.fixture
    def chat_history(self):
        return MagicMock(spec=ChatHistory)

    @pytest.fixture
    def settings(self):
        return MagicMock(spec=AzureChatPromptExecutionSettings)

    @pytest.mark.asyncio
    async def test_get_streaming_response_success(self, kernel,
                                                  chat_completion,
                                                  chat_history, settings):
        # Mock the streaming response
        async def mock_streaming():
            for content in ["Hello", " world", "!"]:
                chunk = MagicMock()
                chunk.content = content
                yield chunk

        chat_completion.get_streaming_chat_message_content.return_value = mock_streaming(
        )

        # Collect the streaming response
        result = []
        async for chunk in ResponseStrategy.get_streaming_response(
                chat_completion, chat_history, settings, kernel):
            result.append(chunk)

        # Verify the response was correctly streamed
        assert result == ["Hello", " world", "!"]
        chat_completion.get_streaming_chat_message_content.assert_called_once_with(
            chat_history=chat_history, settings=settings, kernel=kernel)

    @pytest.mark.asyncio
    async def test_get_streaming_response_exception(self, kernel,
                                                    chat_completion,
                                                    chat_history, settings):
        # Mock an exception during streaming
        chat_completion.get_streaming_chat_message_content.side_effect = Exception(
            "Test error")

        # Collect the error response
        result = []
        async for chunk in ResponseStrategy.get_streaming_response(
                chat_completion, chat_history, settings, kernel):
            result.append(chunk)

        # Verify error message is yielded
        assert len(result) == 1
        assert "Error retrieving response: Test error" in result[0]

    @pytest.mark.asyncio
    async def test_get_complete_response_success(self, kernel, chat_completion,
                                                 chat_history, settings):
        # Mock the complete response
        response = MagicMock()
        response.content = "Complete response content"
        chat_completion.get_chat_message_content.return_value = response

        # Get the complete response
        result = await ResponseStrategy.get_complete_response(
            chat_completion, chat_history, settings, kernel)

        # Verify the response was correctly returned
        assert result == "Complete response content"
        chat_completion.get_chat_message_content.assert_called_once_with(
            chat_history=chat_history, settings=settings, kernel=kernel)

    @pytest.mark.asyncio
    async def test_get_complete_response_exception(self, kernel,
                                                   chat_completion,
                                                   chat_history, settings):
        # Mock an exception during complete response
        chat_completion.get_chat_message_content.side_effect = Exception(
            "Test error")

        # Get the error response
        result = await ResponseStrategy.get_complete_response(
            chat_completion, chat_history, settings, kernel)

        # Verify error message is returned
        assert "Error retrieving response: Test error" in result
