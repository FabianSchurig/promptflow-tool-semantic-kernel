import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from promptflow_tool_semantic_kernel.tools.response_processor import HistoryObserver, ResponseProcessor


class TestHistoryObserver:

    def test_init(self):
        history = ChatHistory()
        observer = HistoryObserver(history)

        assert observer.history == history
        assert observer.last_observed_count == 0

    def test_get_new_messages_no_changes(self):
        history = ChatHistory()
        observer = HistoryObserver(history)

        new_messages = observer.get_new_messages()
        assert new_messages == []
        assert observer.last_observed_count == 0

    def test_get_new_messages_with_changes(self):
        history = ChatHistory()
        observer = HistoryObserver(history)

        # Add messages
        history.add_user_message("User message 1")
        history.add_assistant_message("Assistant message 1")

        # Get new messages
        new_messages = observer.get_new_messages()

        assert len(new_messages) == 2
        assert new_messages[0].content == "User message 1"
        assert new_messages[1].content == "Assistant message 1"
        assert observer.last_observed_count == 2

    def test_get_new_messages_sequential_calls(self):
        history = ChatHistory()
        observer = HistoryObserver(history)

        # Add messages
        history.add_user_message("User message 1")

        # Get new messages
        new_messages = observer.get_new_messages()
        assert len(new_messages) == 1
        assert new_messages[0].content == "User message 1"
        assert observer.last_observed_count == 1

        # Add more messages
        history.add_assistant_message("Assistant message 1")

        # Get new messages again
        new_messages = observer.get_new_messages()
        assert len(new_messages) == 1
        assert new_messages[0].content == "Assistant message 1"
        assert observer.last_observed_count == 2


class TestResponseProcessor:

    def setup_method(self):
        self.history = ChatHistory()
        self.observer = HistoryObserver(self.history)
        self.processor = ResponseProcessor(self.observer)

    def test_init(self):
        assert self.processor.history_observer == self.observer

    def test_yield_tool_messages(self):
        # Mock the get_new_messages method
        message = MagicMock(spec=ChatMessageContent)
        message.to_dict.return_value = {
            "role": "tool",
            "content": "Tool message"
        }

        self.observer.get_new_messages = MagicMock(return_value=[message])

        # Collect the yielded messages
        messages = list(self.processor.yield_tool_messages())

        assert len(messages) == 1
        expected = json.dumps({
            "role": "tool",
            "content": "Tool message"
        }) + "\n \n"
        assert messages[0] == expected
        assert self.observer.get_new_messages.called

    @pytest.mark.asyncio
    async def test_process_streaming(self):
        # Mock the yield_tool_messages method
        self.processor.yield_tool_messages = MagicMock(
            return_value=["Tool message"])

        # Mock the content generator
        async def mock_generator():
            yield "Chunk 1"
            yield "Chunk 2"

        content_gen = mock_generator()

        # Collect the results
        results = []
        async for chunk in self.processor.process_streaming(content_gen):
            results.append(chunk)

        assert results == [
            "Tool message", "Chunk 1", "Tool message", "Chunk 2"
        ]
        assert self.processor.yield_tool_messages.call_count == 2

    @pytest.mark.asyncio
    async def test_process_complete(self):
        # Mock the yield_tool_messages method
        self.processor.yield_tool_messages = MagicMock(
            return_value=["Tool message"])

        # Mock the content generator
        async def mock_generator():
            return "Complete response"

        content_gen = mock_generator()

        # Collect the results
        results = []
        async for chunk in self.processor.process_complete(content_gen):
            results.append(chunk)

        assert results == ["Tool message", "Complete response"]
        assert self.processor.yield_tool_messages.call_count == 1

    @pytest.mark.asyncio
    async def test_process(self):
        # Mock the process_streaming and process_complete methods
        async def mock_process_streaming(content_gen):
            yield "Streaming output"

        self.processor.process_streaming = mock_process_streaming

        async def mock_process_complete(content_gen):
            yield "Complete output"

        self.processor.process_complete = mock_process_complete

        # Test streaming process
        content_gen = AsyncMock()
        results = []
        async for output in self.processor.process(content_gen,
                                                   is_streaming=True):
            results.append(output)

        assert results == ["Streaming output"]

        # Test complete process
        results = []
        async for output in self.processor.process(content_gen,
                                                   is_streaming=False):
            results.append(output)

        assert results == ["Complete output"]
