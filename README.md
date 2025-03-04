# Promptflow Tool Semantic Kernel

A Python package that integrates [Semantic Kernel](https://github.com/microsoft/semantic-kernel) with [Azure Prompt Flow](https://github.com/microsoft/promptflow), enabling efficient LLM application development.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=FabianSchurig_promptflow-tool-semantic-kernel&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=FabianSchurig_promptflow-tool-semantic-kernel)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=FabianSchurig_promptflow-tool-semantic-kernel&metric=coverage)](https://sonarcloud.io/summary/new_code?id=FabianSchurig_promptflow-tool-semantic-kernel)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=FabianSchurig_promptflow-tool-semantic-kernel&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=FabianSchurig_promptflow-tool-semantic-kernel)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=FabianSchurig_promptflow-tool-semantic-kernel&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=FabianSchurig_promptflow-tool-semantic-kernel)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=FabianSchurig_promptflow-tool-semantic-kernel&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=FabianSchurig_promptflow-tool-semantic-kernel)

## Why?

This tool bridges the powerful execution flow of Promptflow with the advanced ReAct capabilities of Semantic Kernel, offering several advantages:

- Easily pre-process or post-process data from your main assistant with minimal configuration.
- Leverage Semantic Kernel's planning and reasoning capabilities within your Prompt Flow applications **by just providing configuration.**
- Connect to a variety of LLM providers beyond OpenAI, including **Anthropic Claude, Amazon Bedrock, Llama, and more** through Semantic Kernel's connectors.
- Access Semantic Kernel's growing plugin ecosystem to extend functionality without writing custom code.
- Use Promptflow's UI and batch evaluation with your semantic kernel assistant.

The integration creates a best-of-both-worlds solution, combining Promptflow's orchestration capabilities with Semantic Kernel's flexibility and plugin architecture.


## Installation

Install the package from PyPI:

```bash
pip install promptflow-tool-semantic-kernel
```

You can find the package on [PyPI](https://pypi.org/project/promptflow-tool-semantic-kernel/).

## Usage

### In VSCode Promptflow

Once installed, the Semantic Kernel tool will be available in your Promptflow tools collection:

![New Tool in Sidebar](https://github.com/FabianSchurig/promptflow-tool-semantic-kernel/blob/e083a336c1c587b12c632f97365f01c0f0a9faa3/docs/promptflow_tools.png)

1. Create a new promptflow in VSCode
2. Add a custom LLM tool node
3. Select "Semantic Kernel LLM Tool" from the tool list
4. Configure the following parameters:
    - Connection (Azure OpenAI or OpenAI)
    - Deployment name (model name for OpenAI or deployment name for Azure)
    - Chat history (optional)
    - Plugins (optional)
    - Customize your prompt as needed  
      
  
![Semantic Kernel Chat](https://github.com/FabianSchurig/promptflow-tool-semantic-kernel/blob/e083a336c1c587b12c632f97365f01c0f0a9faa3/docs/vscode.png)

## Running the Demo

The package includes a simple demo script:

```bash
# Set up environment variables
export AZURE_OPENAI_API_KEY=your_api_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
export AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Run the demo
python -m scripts.main
```

## Using different connections

### Google Gemini

Add a CustomConnection via promptflow in vscode as follows, important is that `api_type: "google"`:

```yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/CustomConnection.schema.json
name: "google_gemini"
type: custom
configs:
  api_type: "google"
  model_id: "gemini-2.0-flash"
secrets:
  # Use'<no-change>' to keep original value or '<user-input>' to update it when the application runs.
  api_key: "<user-input>"
```

## Adding Custom Plugins

Semantic Kernel allows you to easily extend functionality through plugins. [Learn more about creating a native plugin](https://learn.microsoft.com/en-us/semantic-kernel/get-started/quick-start-guide?pivots=programming-language-python#create-a-native-plugin).

Here's how to use plugins with this tool:

### Built-in Plugins

The tool comes with a built-in `LightsPlugin` for demonstration:

```python
# Default plugin configuration
plugins = [
     {
          "name": "lights",
          "class": "LightsPlugin",
          "module": "promptflow_tool_semantic_kernel.tools.lights_plugin"
     }
]
```

## Configuring with flow.dag.yaml

You can also configure the tool using a `flow.dag.yaml` file. This file defines the flow and its components, including the `semantic_kernel_chat` tool and its plugins. Here is an example configuration:

```yaml
# filepath: /workspaces/promptflow-tool-semantic-kernel/tests/system/flow.dag.yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
environment:
     python_requirements_txt: requirements.txt
environment_variables:
     PROMPTFLOW_SERVING_ENGINE: fastapi
     PF_DISABLE_TRACING: "false"
inputs:
     chat_history:
          type: list
          is_chat_history: true
          default: []
     question:
          type: string
          is_chat_input: true
outputs:
     answer:
          type: string
          reference: ${semantic_kernel_chat.output}
          is_chat_output: true
nodes:
- name: semantic_kernel_chat
     type: custom_llm
     source:
          type: package_with_prompt
          tool: promptflow_tool_semantic_kernel.tools.semantic_kernel_tool.semantic_kernel_chat
          path: semantic_kernel_chat.jinja2
     inputs:
          connection: open_ai_connection
          deployment_name: gpt-4
          chat_history: ${inputs.chat_history}
          question: ${inputs.question}
          plugins: |
               [
               {
                    "name": "lights",
                    "class": "LightsPlugin",
                    "module": "promptflow_tool_semantic_kernel.tools.lights_plugin"
               }
               ]
```

This configuration allows you to leverage the power of plugins within your flow. You can define multiple plugins to extend the functionality of the `semantic_kernel_chat` tool. Each plugin is specified with its name, class, and module, making it easy to integrate and customize as needed.

## Development

### Setup

1. Clone the repository
     ```bash
     git clone git@github.com:FabianSchurig/promptflow-tool-semantic-kernel.git
     cd promptflow-tool-semantic-kernel
     cp .devcontainer/devcontainer.env.example .devcontainer/devcontainer.env
     ```
2. Start the devcontainer with vs code
3. Install development dependencies (should automatically run):
     ```bash
     poetry install
     ```
4. Activate the environment
     ```bash
     eval $(poetry env activate)
     which python
     uvicorn tests.system.api:app --workers 1 --port 5000
     ```

### Testing

Run the tests with pytest:

```bash
poetry run pytest
poetry run pytest --cov-report xml:coverage.xml --cov-report term --cov=promptflow_tool_semantic_kernel --cov-config=.coveragerc tests/
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the LICENSE file for details.