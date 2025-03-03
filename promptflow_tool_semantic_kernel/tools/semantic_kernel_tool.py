from collections.abc import AsyncGenerator
import logging
from typing import Dict, List, Any, Union
from jinja2 import Template

from promptflow.core import tool
from promptflow.connections import CustomConnection
from promptflow.contracts.types import PromptTemplate
from promptflow._utils.logger_utils import LoggerFactory

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, )

from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory
from promptflow_tool_semantic_kernel.tools.kernel_factory import KernelFactory
from promptflow_tool_semantic_kernel.tools.chat_history_processor import ChatHistoryProcessor
from promptflow_tool_semantic_kernel.tools.response_strategy import ResponseStrategy

import logging


@tool(streaming_option_parameter="streaming")
async def semantic_kernel_chat(
        connection: CustomConnection,
        deployment_name: str,
        chat_history: List[Dict[str, Any]],
        prompt: PromptTemplate,
        streaming: bool = True,
        **kwargs) -> Union[str, AsyncGenerator[str, None]]:
    """
    Process chat interactions using Semantic Kernel.
    
    Args:
        connection: Azure OpenAI connection details
        deployment_name: The deployment name to use
        chat_history: Previous chat interactions
        prompt: The prompt template to use
        streaming: Whether to stream the response
        **kwargs: Additional parameters for prompt rendering
    """
    logger = LoggerFactory.create_logger("semantic-kernel-tool", logging.INFO)

    try:
        # Render the prompt with provided parameters
        rendered_prompt = Template(str(prompt),
                                   trim_blocks=True,
                                   keep_trailing_newline=True).render(**kwargs)

        logger.info(f"Processing prompt: {rendered_prompt[:50]}...")

        # Create and configure the kernel
        kernel, chat_completion = KernelFactory.create_kernel(
            connection, deployment_name)

        # Process chat history
        history = ChatHistoryProcessor.build_history(chat_history,
                                                     rendered_prompt)

        # Configure execution settings
        execution_settings = AzureChatPromptExecutionSettings()
        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        )

        # Get response using appropriate strategy
        if streaming:

            # TODO: Investigate why this patch is necessary, and if it can be avoided
            # Currently it would wrap the openai.AsyncStream object with the TracedAsyncIterator,
            # then FastAPI does not return any response
            import promptflow.tracing._trace

            original_handle_output = promptflow.tracing._trace.handle_output

            def patched_handle_output(span, inputs, output, trace_type):
                return output

            promptflow.tracing._trace.handle_output = patched_handle_output

            # At the beginning of your function

            logger.debug("Using streaming response strategy")
            response_strategy = ResponseStrategy.get_streaming_response(
                chat_completion, history, execution_settings, kernel)

            async for chunk in response_strategy:
                logger.debug(f"Streaming chunk: {chunk}")
                yield chunk

            promptflow.tracing._trace.handle_output = original_handle_output
        else:
            logger.debug("Using complete response strategy")
            response = await ResponseStrategy.get_complete_response(
                chat_completion, history, execution_settings, kernel)
            yield response

    except Exception as e:
        logger.error(f"Semantic kernel processing failed: {str(e)}",
                     exc_info=True)
        yield f"An error occurred: {str(e)}"
