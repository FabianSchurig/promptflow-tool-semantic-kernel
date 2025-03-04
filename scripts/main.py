import os
import asyncio
from dotenv import load_dotenv
from promptflow_tool_semantic_kernel.tools.semantic_kernel_tool import semantic_kernel_chat
from promptflow.connections import CustomConnection

if __name__ == "__main__":

    async def run_demo():
        # Load environment variables from .env file
        load_dotenv()

        # Create a mock connection with Azure OpenAI credentials
        mock_connection = CustomConnection(
            secrets={
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                "api_type": "azure"
            },
            configs={"base_url": os.environ.get("AZURE_OPENAI_ENDPOINT")})

        # Sample chat history
        sample_history = [{
            "inputs": {
                "question": "Hello, how are you?"
            },
            "outputs": {
                "answer": {
                    "content":
                    "I'm doing well, thank you! How can I help you today?"
                }
            }
        }]

        # Sample prompt
        sample_prompt = "Tell me about semantic kernel in 2-3 sentences."

        print("Running semantic kernel tool...")

        # Run the tool and handle streaming output
        async for chunk in semantic_kernel_chat(
            connection=mock_connection,
            deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME",
                                           "gpt-35-turbo"),
            chat_history=sample_history,
            prompt=sample_prompt,
            plugins=[],
            streaming=True):
            if chunk is not None:
                print(chunk, end="", flush=True)

        print("\nDemo completed!")

    # Run the async demo
    asyncio.run(run_demo())
