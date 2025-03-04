from collections.abc import AsyncGenerator
import json
import logging
from typing import Dict, List, Any, Union
from jinja2 import Template

from promptflow.core import tool
from promptflow.connections import CustomConnection, AzureOpenAIConnection, OpenAIConnection
from promptflow.contracts.types import PromptTemplate
from promptflow._utils.logger_utils import LoggerFactory

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory

from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory
from promptflow_tool_semantic_kernel.tools.kernel_factory import KernelFactory
from promptflow_tool_semantic_kernel.tools.chat_history_processor import ChatHistoryProcessor
from promptflow_tool_semantic_kernel.tools.response_strategy import ResponseStrategy
from promptflow_tool_semantic_kernel.tools.tracing_disabler import TracingDisabler
from promptflow_tool_semantic_kernel.tools.plugin_manager import PluginManager
from promptflow_tool_semantic_kernel.tools.response_processor import HistoryObserver, ResponseProcessor

import logging


@tool(streaming_option_parameter="streaming")
async def semantic_kernel_chat(
        connection: CustomConnection | AzureOpenAIConnection
    | OpenAIConnection,
        deployment_name: str,
        chat_history: List[Dict[str, Any]],
        prompt: PromptTemplate,
        plugins: List[Dict[str, Any]],
        streaming: bool = True,
        **kwargs) -> Union[str, AsyncGenerator[str, None]]:
    """
    Process chat interactions using Semantic Kernel.

    Args:
    connection: Azure OpenAI connection details
    deployment_name: The deployment name to use
    chat_history: Previous chat interactions
    prompt: The prompt template to use
    plugins: List of plugins to register with the kernel. Each plugin should be a dict with 
        'instance' (the plugin instance) and 'name' (optional plugin name)
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

        # Register plugins using the plugin manager
        plugin_manager = PluginManager(kernel, logger)
        plugin_manager.register_plugins(plugins)

        # Process chat history
        history: ChatHistory = ChatHistoryProcessor.build_history(
            chat_history, rendered_prompt)

        # Configure execution settings
        # Get execution settings from the kernel factory
        execution_settings = KernelFactory.get_execution_settings(connection)
        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        )

        # Get response using appropriate strategy
        # Create the history observer and response processor
        history_observer = HistoryObserver(history)
        processor = ResponseProcessor(history_observer)

        # For streaming response
        if streaming:
            with TracingDisabler():
                logger.debug("Using streaming response strategy")
                content_generator = ResponseStrategy.get_streaming_response(
                    chat_completion, history, execution_settings, kernel)
                async for output in processor.process(content_generator,
                                                      is_streaming=True):
                    yield output
        # For complete response
        else:
            logger.debug("Using complete response strategy")
            content_generator = ResponseStrategy.get_complete_response(
                chat_completion, history, execution_settings, kernel)
            async for output in processor.process(content_generator,
                                                  is_streaming=False):
                yield output

    except Exception as e:
        logger.error(f"Semantic kernel processing failed: {str(e)}",
                     exc_info=True)
        yield f"An error occurred: {str(e)}"
