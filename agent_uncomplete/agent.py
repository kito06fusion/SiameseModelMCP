from __future__ import annotations

import asyncio
import os

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

# TODO: Add the MCP server URL.
MCP_URL = ...

# TODO: Write the agent instructions.
INSTRUCTIONS = ...


async def main() -> None:
    async with (
        AzureCliCredential() as credential,
        MCPStreamableHTTPTool(
            name=...,  # TODO: Add the MCP tool name.
            url=MCP_URL,
        ) as mcp_server,
        Agent(
            client=AzureOpenAIResponsesClient(
                project_endpoint=...,  # TODO: Add the Azure AI project endpoint.
                deployment_name=...,  # TODO: Add the Azure OpenAI deployment name.
                credential=...,  # TODO: Add the Azure credential object.
            ),
            name=...,  # TODO: Add the agent name.
            instructions= TODO: Add the agent instructions,
        ) as agent,
    ):
        print("Face Recognition Agent  (type 'quit' to exit)")
        print(f"MCP server: {MCP_URL}")
        print()

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            result = await agent.run(user_input, tools=mcp_server)
            print(f"Agent: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())