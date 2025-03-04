from typing import List, Dict, Any
from semantic_kernel.contents.chat_history import ChatHistory

from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory


class ChatHistoryProcessor:

    @staticmethod
    def build_history(chat_history: List[Dict[str, Any]],
                      current_prompt: str) -> ChatHistory:
        """Process chat history and add the current prompt"""
        history = ChatHistory()

        try:
            for entry in chat_history:
                if isinstance(entry, dict):
                    user_message = entry.get("inputs", {}).get("question", "")
                    answer = entry.get("outputs", {}).get("answer", {})

                    # Handle the case where "answer" is a string
                    assistant_message = ""
                    if isinstance(answer, dict):
                        assistant_message = answer.get("content", "")
                    elif isinstance(answer, str):
                        assistant_message = answer

                    if user_message:
                        history.add_user_message(user_message)
                    if assistant_message:
                        history.add_assistant_message(assistant_message)
                elif isinstance(entry, str):
                    # Handle string entries by adding them as user messages
                    history.add_user_message(entry)

            # Add current prompt as the latest user message
            history.add_user_message(current_prompt)
            return history
        except Exception as e:
            logger = LoggerFactory.create_logger("chat-history")
            logger.error(f"Failed to build chat history: {str(e)}")
            raise
