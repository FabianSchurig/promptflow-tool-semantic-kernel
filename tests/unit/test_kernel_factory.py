import pytest
from unittest.mock import MagicMock, patch
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from promptflow_tool_semantic_kernel.tools.kernel_factory import KernelFactory


class TestKernelFactory:

    def test_create_kernel_with_azure_connection(self):
        # Mock AzureOpenAIConnection
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "AzureOpenAIConnection"
        mock_connection.api_base = "https://test.openai.azure.com"
        mock_connection.secrets = {"api_key": "test-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
             patch('promptflow_tool_semantic_kernel.tools.kernel_factory.AzureChatCompletion', autospec=True) as mock_azure_chat:
            mock_kernel = MagicMock()
            mock_kernel.add_service.return_value = None
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_azure_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, "test-deployment")

            assert kernel == mock_kernel
            # Rather than comparing objects directly, verify the mock was used
            mock_azure_chat.assert_called_once_with(
                api_key="test-key",
                deployment_name="test-deployment",
                base_url="https://test.openai.azure.com")
            mock_kernel.add_service.assert_called_once_with(mock_chat)

    def test_create_kernel_with_openai_connection(self):
        # Mock OpenAIConnection
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "OpenAIConnection"
        mock_connection.secrets = {"api_key": "test-key"}
        mock_connection.organization = "test-org"

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
             patch('promptflow_tool_semantic_kernel.tools.kernel_factory.OpenAIChatCompletion', autospec=True) as mock_openai_chat:
            mock_kernel = MagicMock()
            mock_kernel.add_service.return_value = None
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_openai_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, "gpt-4")

            assert kernel == mock_kernel
            assert chat_completion == mock_chat
            mock_openai_chat.assert_called_once_with(api_key="test-key",
                                                     ai_model_id="gpt-4",
                                                     org_id="test-org")
            mock_kernel.add_service.assert_called_once_with(mock_chat)

    def test_create_kernel_with_custom_connection_azure(self):
        # Mock CustomConnection with Azure settings
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {
            "api_type": "azure",
            "base_url": "https://custom.azure.com"
        }
        mock_connection.secrets = {"api_key": "custom-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
             patch('promptflow_tool_semantic_kernel.tools.kernel_factory.AzureChatCompletion', autospec=True) as mock_azure_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_azure_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, "deployment-name")

            mock_azure_chat.assert_called_once_with(
                api_key="custom-key",
                deployment_name="deployment-name",
                base_url="https://custom.azure.com")

    def test_create_kernel_with_custom_connection_openai(self):
        # Mock CustomConnection with OpenAI settings
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {"organization": "custom-org"}
        mock_connection.secrets = {"api_key": "custom-key"}

        with patch.object(KernelFactory, '_is_azure_connection', return_value=False), \
             patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
             patch('promptflow_tool_semantic_kernel.tools.kernel_factory.OpenAIChatCompletion', autospec=True) as mock_openai_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_openai_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, "gpt-4")

            mock_openai_chat.assert_called_once_with(api_key="custom-key",
                                                     ai_model_id="gpt-4",
                                                     org_id="custom-org")

    def test_is_azure_connection(self):
        # Test AzureOpenAIConnection
        azure_connection = MagicMock()
        azure_connection.__class__.__name__ = "AzureOpenAIConnection"
        assert KernelFactory._is_azure_connection(azure_connection) == True

        # Test OpenAIConnection
        openai_connection = MagicMock()
        openai_connection.__class__.__name__ = "OpenAIConnection"
        assert KernelFactory._is_azure_connection(openai_connection) == False

        # Test CustomConnection with Azure
        custom_azure = MagicMock()
        custom_azure.__class__.__name__ = "CustomConnection"
        custom_azure.configs = {"api_type": "azure"}
        assert KernelFactory._is_azure_connection(custom_azure) == True

        # Test CustomConnection with base_url
        custom_with_base_url = MagicMock()
        custom_with_base_url.__class__.__name__ = "CustomConnection"
        custom_with_base_url.configs = {"base_url": "https://example.com"}
        assert KernelFactory._is_azure_connection(custom_with_base_url) == True

        # Test CustomConnection with non-Azure
        custom_non_azure = MagicMock()
        custom_non_azure.__class__.__name__ = "CustomConnection"
        custom_non_azure.configs = {"api_type": "openai"}
        assert KernelFactory._is_azure_connection(custom_non_azure) == False

        # Test unknown connection
        unknown = MagicMock()
        unknown.__class__.__name__ = "UnknownConnection"
        assert KernelFactory._is_azure_connection(unknown) == True

    def test_create_kernel_exception_handling(self):
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "OpenAIConnection"

        expected_error = "Failed to create OpenAI settings."
        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.OpenAIChatCompletion',
                  side_effect=Exception(expected_error)), \
             patch('promptflow_tool_semantic_kernel.tools.logger_factory.LoggerFactory.create_logger') as mock_logger_factory:
            mock_logger = MagicMock()
            mock_logger_factory.return_value = mock_logger

            with pytest.raises(Exception, match=expected_error):
                KernelFactory.create_kernel(mock_connection, "gpt-4")

            mock_logger.error.assert_called_once_with(
                f"Failed to create kernel: {expected_error}")
