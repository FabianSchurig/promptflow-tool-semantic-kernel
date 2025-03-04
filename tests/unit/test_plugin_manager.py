import pytest
import logging
from unittest.mock import MagicMock, patch
from semantic_kernel import Kernel
from promptflow_tool_semantic_kernel.tools.plugin_manager import PluginManager


class TestPluginManager:

    @pytest.fixture
    def mock_kernel(self):
        return MagicMock(spec=Kernel)

    @pytest.fixture
    def mock_logger(self):
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def plugin_manager(self, mock_kernel, mock_logger):
        return PluginManager(mock_kernel, mock_logger)

    def test_register_valid_plugin(self, plugin_manager, mock_kernel,
                                   mock_logger):
        # Arrange
        mock_plugin = MagicMock()

        with patch('importlib.import_module') as mock_import, \
             patch('builtins.getattr') as mock_getattr:
            # Setup mocks
            mock_import.return_value = MagicMock()
            mock_getattr.return_value = MagicMock(return_value=mock_plugin)

            # Reset mock to clear any previous calls
            mock_import.reset_mock()

            # Act
            plugin_manager.register_plugins([{
                "name":
                "lights",
                "class":
                "LightsPlugin",
                "module":
                "promptflow_tool_semantic_kernel.tools.lights_plugin"
            }])

            # Assert
            mock_import.assert_called_once_with(
                "promptflow_tool_semantic_kernel.tools.lights_plugin")
            mock_kernel.add_plugin.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Registered plugin 'lights'")

    def test_register_multiple_plugins(self, plugin_manager, mock_kernel,
                                       mock_logger):
        # Arrange
        with patch('importlib.import_module') as mock_import, \
             patch('builtins.getattr') as mock_getattr:
            # Setup mocks
            mock_import.return_value = MagicMock()
            mock_getattr.return_value = MagicMock(return_value=MagicMock())

            # Reset mocks to clear any previous calls
            mock_import.reset_mock()
            mock_getattr.reset_mock()
            mock_kernel.reset_mock()
            mock_logger.reset_mock()

            # Act
            plugin_manager.register_plugins([{
                "name":
                "lights",
                "class":
                "LightsPlugin",
                "module":
                "promptflow_tool_semantic_kernel.tools.lights_plugin"
            }, {
                "name":
                "weather",
                "class":
                "WeatherPlugin",
                "module":
                "promptflow_tool_semantic_kernel.tools.weather_plugin"
            }])

            # Assert
            assert mock_import.call_count == 2
            assert mock_kernel.add_plugin.call_count == 2
            assert mock_logger.error.call_count == 0

    def test_register_invalid_plugin_missing_fields(self, plugin_manager,
                                                    mock_kernel, mock_logger):
        # Act
        plugin_manager.register_plugins([{
            "name": "lights",
            # Missing class and module
        }])

        # Assert
        mock_logger.error.assert_called()
        mock_kernel.add_plugin.assert_not_called()

    def test_register_plugin_import_error(self, plugin_manager, mock_kernel,
                                          mock_logger):
        # Arrange
        with patch('importlib.import_module') as mock_import:
            # Setup mock to raise ImportError
            mock_import.side_effect = ImportError("Module not found")

            # Act
            plugin_manager.register_plugins([{
                "name": "lights",
                "class": "LightsPlugin",
                "module": "nonexistent_module"
            }])

            # Assert
            mock_logger.error.assert_called_once()
            mock_kernel.add_plugin.assert_not_called()

    def test_empty_plugin_list(self, plugin_manager, mock_kernel, mock_logger):
        # Act
        plugin_manager.register_plugins([])

        # Assert
        mock_kernel.add_plugin.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.info.assert_not_called()

    def test_register_plugin_with_none_in_list(self, plugin_manager,
                                               mock_kernel, mock_logger):
        # Act
        plugin_manager.register_plugins([None])

        # Assert
        mock_logger.error.assert_called_once()
        mock_kernel.add_plugin.assert_not_called()

    def test_register_plugin_with_non_dict_item(self, plugin_manager,
                                                mock_kernel, mock_logger):
        # Act
        plugin_manager.register_plugins(["not_a_dict"])

        # Assert
        mock_logger.error.assert_called_once()
        mock_kernel.add_plugin.assert_not_called()

    def test_register_plugin_kernel_error(self, plugin_manager, mock_kernel,
                                          mock_logger):
        # Arrange
        with patch('importlib.import_module') as mock_import, \
             patch('builtins.getattr') as mock_getattr:
            # Setup mocks
            mock_import.return_value = MagicMock()
            mock_class = MagicMock(return_value=MagicMock())
            mock_getattr.return_value = mock_class

            # Make kernel.add_plugin raise an exception
            mock_kernel.add_plugin.side_effect = Exception("Kernel error")

            # Act
            plugin_manager.register_plugins([{
                "name":
                "lights",
                "class":
                "LightsPlugin",
                "module":
                "promptflow_tool_semantic_kernel.tools.lights_plugin"
            }])

            # Assert
            mock_logger.error.assert_called_once()
