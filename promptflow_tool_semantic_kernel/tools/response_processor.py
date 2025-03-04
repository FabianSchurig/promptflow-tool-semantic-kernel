import json
from typing import List

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent


class HistoryObserver:
    """Observes a ChatHistory object for new messages."""

    def __init__(self, history: ChatHistory):
        """Initialize the observer with a chat history."""
        self.history: ChatHistory = history
        self.last_observed_count: int = len(history.messages)

    def get_new_messages(self) -> List[ChatMessageContent]:
        """Return any new messages added to the history since last check."""
        current_count = len(self.history.messages)
        if current_count > self.last_observed_count:
            new_messages: list[ChatMessageContent] = self.history.messages[
                self.last_observed_count:current_count]
            self.last_observed_count = current_count
            return new_messages
        return []


class ResponseProcessor:
    """Handles processing of response content from chat completions."""

    def __init__(self, history_observer):
        self.history_observer = history_observer

    def yield_tool_messages(self):
        """Yield tool messages from the history observer."""
        for message in self.history_observer.get_new_messages():
            yield json.dumps(message.to_dict()) + "\n \n"

    async def process_streaming(self, content_generator):
        """Process a streaming response."""
        async for chunk in content_generator:
            for message in self.yield_tool_messages():
                yield message
            yield chunk

    async def process_complete(self, content_generator):
        """Process a complete (non-streaming) response."""
        response = await content_generator
        for message in self.yield_tool_messages():
            yield message
        yield response

    async def process(self, content_generator, is_streaming=True):
        """Process the response using the appropriate strategy."""
        if is_streaming:
            async for output in self.process_streaming(content_generator):
                yield output
        else:
            async for output in self.process_complete(content_generator):
                yield output
