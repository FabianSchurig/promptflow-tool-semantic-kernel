from collections.abc import AsyncGenerator

from promptflow._utils.logger_utils import LoggerFactory

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, )

from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory
import traceback


class ResponseStrategy:

    @staticmethod
    async def get_streaming_response(
            chat_completion: AzureChatCompletion, history: ChatHistory,
            settings: AzureChatPromptExecutionSettings,
            kernel: Kernel) -> AsyncGenerator[str, None]:
        """Handle streaming response strategy"""
        logger = LoggerFactory.create_logger("response-strategy")

        try:
            streaming_response = chat_completion.get_streaming_chat_message_content(
                chat_history=history,
                settings=settings,
                kernel=kernel,
            )
            async for chunk in streaming_response:
                yield chunk.content
        except Exception as e:
            logger.error(
                f"Error in streaming response: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            yield f"Error retrieving response: {str(e)}"

    @staticmethod
    async def get_complete_response(chat_completion: AzureChatCompletion,
                                    history: ChatHistory,
                                    settings: AzureChatPromptExecutionSettings,
                                    kernel: Kernel) -> str:
        """Handle complete (non-streaming) response strategy"""
        logger = LoggerFactory.create_logger("response-strategy")

        try:
            response = await chat_completion.get_chat_message_content(
                chat_history=history,
                settings=settings,
                kernel=kernel,
            )
            logger.debug(f"Response type: {type(response)}")
            return response.content
        except Exception as e:
            logger.error(f"Error in complete response: {str(e)}")
            return f"Error retrieving response: {str(e)}"
