from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory


class KernelFactory:

    @staticmethod
    def create_kernel(connection: Any, model_or_deployment: str) -> tuple:
        """Create and configure a Semantic Kernel instance based on connection type
        
        Args:
            connection: The connection object (CustomConnection, OpenAIConnection, or AzureOpenAIConnection)
            model_or_deployment: The model ID (OpenAI) or deployment name (Azure OpenAI)
            
        Returns:
            tuple: (kernel, chat_completion_service)
        """
        logger = LoggerFactory.create_logger("kernel-factory")

        try:
            kernel = Kernel()

            # Create the appropriate chat completion service
            if KernelFactory._is_azure_connection(connection):
                chat_completion = KernelFactory._create_azure_chat_completion(
                    connection, model_or_deployment)
            else:
                chat_completion = KernelFactory._create_openai_chat_completion(
                    connection, model_or_deployment)

            kernel.add_service(chat_completion)
            return kernel, chat_completion
        except Exception as e:
            logger.error(f"Failed to create kernel: {str(e)}")
            raise

    @staticmethod
    def _is_azure_connection(connection: Any) -> bool:
        """Determine if the connection is for Azure OpenAI"""
        connection_class = connection.__class__.__name__

        if connection_class == "AzureOpenAIConnection":
            return True
        elif connection_class == "OpenAIConnection":
            return False
        elif connection_class == "CustomConnection":
            # Check CustomConnection configs
            configs = getattr(connection, "configs", {})
            return configs.get("api_type") == "azure" or "base_url" in configs
        else:
            # Default to Azure if we can't determine
            return True

    @staticmethod
    def _create_azure_chat_completion(connection: Any, deployment_name: str):
        """Create Azure OpenAI chat completion service"""

        connection_class = connection.__class__.__name__

        if connection_class == "AzureOpenAIConnection":
            base_url = connection.api_base
            api_key = connection.secrets.get("api_key")
        else:  # CustomConnection or other
            base_url = getattr(connection, "configs", {}).get("base_url")
            api_key = getattr(connection, "secrets", {}).get("api_key")

        return AzureChatCompletion(api_key=api_key,
                                   deployment_name=deployment_name,
                                   base_url=base_url)

    @staticmethod
    def _create_openai_chat_completion(connection: Any, model_id: str):
        """Create OpenAI chat completion service"""

        connection_class = connection.__class__.__name__

        if connection_class == "OpenAIConnection":
            api_key = connection.secrets.get("api_key")
            org_id = connection.organization
        else:  # CustomConnection or other
            api_key = getattr(connection, "secrets", {}).get("api_key")
            org_id = getattr(connection, "configs", {}).get("organization")

        return OpenAIChatCompletion(api_key=api_key,
                                    ai_model_id=model_id,
                                    org_id=org_id)
