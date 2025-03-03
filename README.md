# Promptflow Tool Semantic Kernel

A Python package that integrates [Semantic Kernel](https://github.com/microsoft/semantic-kernel) with [Azure Prompt Flow](https://github.com/microsoft/promptflow), enabling efficient LLM application development.

## Features

- Ready-to-use custom LLM tool that leverages Semantic Kernel in Prompt Flow
- Supports both streaming and non-streaming responses
- Compatible with Azure OpenAI and OpenAI connections
- Customizable prompt templates
- Built-in chat history processing for conversational applications

## Installation

Install the package from PyPI:

```bash
pip install promptflow-tool-semantic-kernel
```

## Usage

### In Azure Prompt Flow

Once installed, the Semantic Kernel tool will be available in your Prompt Flow tools collection:

1. Create a new flow in Azure Prompt Flow
2. Add a custom LLM tool node
3. Select "Semantic Kernel LLM Tool" from the tool list
4. Configure the following parameters:
    - Connection (Azure OpenAI or OpenAI)
    - Deployment name (model name for OpenAI or deployment name for Azure)
    - Chat history (optional)
    - Customize your prompt as needed

### Sample Code

```python
from promptflow.connections import CustomConnection
from promptflow.contracts.types import PromptTemplate
from promptflow_tool_semantic_kernel.tools.semantic_kernel_tool import semantic_kernel_chat

# Create a connection
connection = CustomConnection(
     secrets={"api_key": "your_api_key"},
     configs={"base_url": "https://your-endpoint.openai.azure.com/", "api_type": "azure"}
)

# Define chat history
chat_history = [{
     "inputs": {"question": "Hello"},
     "outputs": {"answer": {"content": "Hi there! How can I help you?"}}
}]

# Define prompt template
prompt = PromptTemplate("Tell me about {{topic}}")

# Call the tool
async for chunk in semantic_kernel_chat(
     connection=connection,
     deployment_name="gpt-4",
     chat_history=chat_history,
     prompt=prompt,
     streaming=True,
     topic="Azure Prompt Flow"
):
     print(chunk, end="", flush=True)
```

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

## Development

### Setup

1. Clone the repository
2. Install Poetry
3. Install development dependencies:
    ```bash
    poetry install
    ```

### Testing

Run the tests with pytest:

```bash
poetry run pytest
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the LICENSE file for details.