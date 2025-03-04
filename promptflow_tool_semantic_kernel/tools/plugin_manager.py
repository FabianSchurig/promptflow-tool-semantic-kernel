import importlib
from typing import List, Dict, Any
import logging

from semantic_kernel import Kernel


class PluginManager:
    """
    Manages the registration of plugins to a semantic kernel.
    """

    def __init__(self, kernel: Kernel, logger: logging.Logger):
        self.kernel: Kernel = kernel
        self.logger: logging.Logger = logger

    def register_plugins(self, plugins: List[Dict[str, Any]]) -> None:
        """Register multiple plugins with the semantic kernel.

        Parameters
        ----------
        plugins : List[Dict[str, Any]]
            A list of plugin definitions, where each plugin is defined as a dictionary with:
            - "name": The name of the plugin
            - "class": The class name of the plugin
            - "module": The full module path where the plugin class is defined

        Example
        -------
        ```python
        plugins = [
            {
            "name": "lights",
            "class": "LightsPlugin",
            "module": "promptflow_tool_semantic_kernel.tools.lights_plugin"
            },
            {
            "name": "weather",
            "class": "WeatherPlugin",
            "module": "promptflow_tool_semantic_kernel.tools.weather_plugin"
            }
        ]
        plugin_manager.register_plugins(plugins)
        ```
        """
        for plugin in plugins:
            if not plugin or not isinstance(plugin, dict):
                self.logger.error(f"Invalid plugin definition: {plugin}")
                continue

            plugin_name = plugin.get("name")
            plugin_class = plugin.get("class")
            plugin_module = plugin.get("module")
            plugin_parameters = plugin.get("parameters", {})

            if not plugin_name or not plugin_class or not plugin_module:
                self.logger.error(f"Invalid plugin definition: {plugin}")
                continue

            try:
                # Import the plugin module
                module = importlib.import_module(plugin_module)
                plugin_class_obj = getattr(module, plugin_class)

                # Create an instance of the plugin
                instance = plugin_class_obj(**plugin_parameters)

                # Register the plugin with the kernel
                self.kernel.add_plugin(instance, plugin_name=plugin_name)
                self.logger.info(f"Registered plugin '{plugin_name}'")
            except ImportError as e:
                self.logger.error(
                    f"Failed to register plugin '{plugin_name}': {str(e)}")
            except Exception as e:
                self.logger.error(
                    f"Failed to register plugin '{plugin_name}': {str(e)}")
