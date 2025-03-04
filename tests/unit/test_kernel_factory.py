import pytest
from unittest.mock import MagicMock, patch
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from promptflow_tool_semantic_kernel.tools.kernel_factory import KernelFactory

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, )
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings, )
from semantic_kernel.connectors.ai.google.google_ai.google_ai_prompt_execution_settings import GoogleAIChatPromptExecutionSettings


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
            assert chat_completion == mock_chat
            # Rather than comparing objects directly, verify the mock was used
            mock_azure_chat.assert_called_once_with(
                api_key="test-key",
                deployment_name="test-deployment",
                base_url="https://test.openai.azure.com/openai/")
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
        assert KernelFactory._is_azure_connection(unknown) == False

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

    def test_azure_base_url_formatting(self):
        # Test that base_url gets properly formatted with /openai/ suffix
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "AzureOpenAIConnection"
        mock_connection.api_base = "https://test.openai.azure.com"  # No trailing /openai/
        mock_connection.secrets = {"api_key": "test-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel'), \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.AzureChatCompletion', autospec=True) as mock_azure_chat:
            KernelFactory.create_kernel(mock_connection, "test-deployment")
            mock_azure_chat.assert_called_once_with(
                api_key="test-key",
                deployment_name="test-deployment",
                base_url="https://test.openai.azure.com/openai/")

    def test_azure_base_url_already_has_openai_suffix(self):
        # Test that base_url is not modified if it already has /openai/ suffix
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "AzureOpenAIConnection"
        mock_connection.api_base = "https://test.openai.azure.com/openai/"  # Already has /openai/
        mock_connection.secrets = {"api_key": "test-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel'), \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.AzureChatCompletion', autospec=True) as mock_azure_chat:
            KernelFactory.create_kernel(mock_connection, "test-deployment")
            mock_azure_chat.assert_called_once_with(
                api_key="test-key",
                deployment_name="test-deployment",
                base_url="https://test.openai.azure.com/openai/")

    def test_custom_connection_azure_base_url_formatting(self):
        # Test CustomConnection with Azure base_url formatting
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {
            "base_url": "https://custom.azure.com"
        }  # No trailing /openai/
        mock_connection.secrets = {"api_key": "custom-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel'), \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.AzureChatCompletion', autospec=True) as mock_azure_chat:
            KernelFactory.create_kernel(mock_connection, "deployment-name")
            mock_azure_chat.assert_called_once_with(
                api_key="custom-key",
                deployment_name="deployment-name",
                # Custom connection doesn't get /openai/ appended
                base_url="https://custom.azure.com")

    def test_create_kernel_returns_correct_tuple(self):
        # Test that create_kernel returns the expected tuple
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "OpenAIConnection"
        mock_connection.secrets = {"api_key": "test-key"}
        mock_connection.organization = "test-org"

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.OpenAIChatCompletion', autospec=True) as mock_openai_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_openai_chat.return_value = mock_chat

            result = KernelFactory.create_kernel(mock_connection, "gpt-4")

            # Verify the return value is a tuple with kernel and chat completion
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == mock_kernel
            assert result[1] == mock_chat

    def test_is_google_ai_connection(self):
        # Test CustomConnection with Google AI settings
        custom_google = MagicMock()
        custom_google.__class__.__name__ = "CustomConnection"
        custom_google.configs = {"api_type": "google"}
        assert KernelFactory._is_google_ai_connection(custom_google) == True

        # Test CustomConnection with gemini model_id
        custom_gemini = MagicMock()
        custom_gemini.__class__.__name__ = "CustomConnection"
        custom_gemini.configs = {"model_id": "gemini-2.0"}
        assert KernelFactory._is_google_ai_connection(custom_gemini) == True

        # Test CustomConnection with non-Google AI
        custom_non_google = MagicMock()
        custom_non_google.__class__.__name__ = "CustomConnection"
        custom_non_google.configs = {"api_type": "openai"}
        assert KernelFactory._is_google_ai_connection(
            custom_non_google) == False

        # Test unknown connection
        unknown = MagicMock()
        unknown.__class__.__name__ = "UnknownConnection"
        assert KernelFactory._is_google_ai_connection(unknown) == False

    def test_create_google_ai_chat_completion(self):
        # Mock CustomConnection with Google AI settings
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {"model_id": "gemini-2.0"}
        mock_connection.secrets = {"api_key": "google-key"}

        with patch(
                'promptflow_tool_semantic_kernel.tools.kernel_factory.GoogleAIChatCompletion',
                autospec=True) as mock_google_chat:
            mock_chat = MagicMock()
            mock_google_chat.return_value = mock_chat

            chat_completion = KernelFactory._create_google_ai_chat_completion(
                mock_connection, "gemini-2.0")

            assert chat_completion == mock_chat
            mock_google_chat.assert_called_once_with(
                gemini_model_id="gemini-2.0", api_key="google-key")

    def test_get_execution_settings(self):
        # Test AzureOpenAIConnection
        azure_connection = MagicMock()
        azure_connection.__class__.__name__ = "AzureOpenAIConnection"
        assert isinstance(
            KernelFactory.get_execution_settings(azure_connection),
            AzureChatPromptExecutionSettings)

        # Test OpenAIConnection
        openai_connection = MagicMock()
        openai_connection.__class__.__name__ = "OpenAIConnection"
        assert isinstance(
            KernelFactory.get_execution_settings(openai_connection),
            OpenAIChatPromptExecutionSettings)

        # Test CustomConnection with Google AI
        custom_google = MagicMock()
        custom_google.__class__.__name__ = "CustomConnection"
        custom_google.configs = {"api_type": "google"}
        assert isinstance(KernelFactory.get_execution_settings(custom_google),
                          GoogleAIChatPromptExecutionSettings)

        # Test CustomConnection with Azure
        custom_azure = MagicMock()
        custom_azure.__class__.__name__ = "CustomConnection"
        custom_azure.configs = {"api_type": "azure"}
        assert isinstance(KernelFactory.get_execution_settings(custom_azure),
                          AzureChatPromptExecutionSettings)

        # Test CustomConnection with OpenAI
        custom_openai = MagicMock()
        custom_openai.__class__.__name__ = "CustomConnection"
        custom_openai.configs = {"api_type": "openai"}
        assert isinstance(KernelFactory.get_execution_settings(custom_openai),
                          OpenAIChatPromptExecutionSettings)

    def test_create_kernel_with_invalid_connection(self):
        # Test with an invalid connection type
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "InvalidConnection"

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.OpenAIChatCompletion', autospec=True) as mock_openai_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_openai_chat.return_value = mock_chat

            with patch.object(mock_connection.secrets, 'get', return_value=None), \
                    patch.object(mock_connection.configs, 'get', return_value=None):
                kernel, chat_completion = KernelFactory.create_kernel(
                    mock_connection, "gpt-4")

                assert kernel == mock_kernel
                assert chat_completion == mock_chat
                mock_openai_chat.assert_called_once_with(api_key=None,
                                                         ai_model_id="gpt-4",
                                                         org_id=None)
                mock_kernel.add_service.assert_called_once_with(mock_chat)

    def test_create_kernel_with_google_ai_connection(self):
        # Mock CustomConnection with Google AI settings
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {"api_type": "google"}
        mock_connection.secrets = {"api_key": "google-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.GoogleAIChatCompletion', autospec=True) as mock_google_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_google_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, "gemini-2.0")

            assert kernel == mock_kernel
            assert chat_completion == mock_chat
            mock_google_chat.assert_called_once_with(
                gemini_model_id="gemini-2.0", api_key="google-key")
            mock_kernel.add_service.assert_called_once_with(mock_chat)

    def test_create_kernel_with_custom_connection_google_default_model(self):
        # Mock CustomConnection with Google AI settings
        mock_connection = MagicMock()
        mock_connection.__class__.__name__ = "CustomConnection"
        mock_connection.configs = {"api_type": "google"}
        mock_connection.secrets = {"api_key": "google-key"}

        with patch('promptflow_tool_semantic_kernel.tools.kernel_factory.Kernel') as mock_kernel_class, \
                patch('promptflow_tool_semantic_kernel.tools.kernel_factory.GoogleAIChatCompletion', autospec=True) as mock_google_chat:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_chat = MagicMock()
            mock_google_chat.return_value = mock_chat

            kernel, chat_completion = KernelFactory.create_kernel(
                mock_connection, None)

            assert kernel == mock_kernel
            assert chat_completion == mock_chat
            mock_google_chat.assert_called_once_with(
                gemini_model_id="gemini-2.0-flash", api_key="google-key")
            mock_kernel.add_service.assert_called_once_with(mock_chat)
