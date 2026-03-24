from __future__ import annotations

import asyncio
import os

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

MCP_URL = os.getenv("SIAMESE_MCP_URL", "http://127.0.0.1:8000/mcp")

INSTRUCTIONS = (
    "You are a face recognition assistant backed by a PostgreSQL database. "
    "You have two tools: register_face_file to add a face image, and search_face_file to look one up. "
    "Image files are stored in the /shared/ directory on the server. "
    "When the user provides a filename or a local path, always translate it to /shared/<filename> before calling a tool. "
    "When given an image path, pick the right tool and report the result clearly. "
    "If the conversation context includes a most recent valid image path, reuse it for follow-up "
    "requests such as 'search for a match' or 'register this face' unless the user provides a new path. "
    "Always return the full result from the tool call. Do not shorten no-match results. "
    "For every search_face_file response, always include these sections in this order: "
    "Request Details, Config, Summary, Best Match Details, Matches, Raw Search Result, Warnings, Conclusion. "
    "If no match is found, still include every section and explicitly say: "
    "Best Match Details: None found, Matches: [], Raw Search Result: none returned, and list any warnings. "
    "If there is a best match, include all available fields from the tool result, including distance, threshold, "
    "confidence, model, detector backend, distance metric, reference id, and raw search result details. "
    "For register_face_file responses, always include the full registration summary and any warnings."
)


async def main() -> None:
    async with (
        AzureCliCredential() as credential,
        MCPStreamableHTTPTool(
            name="Siamese Face MCP",
            url=MCP_URL,
        ) as mcp_server,
        Agent(
            client=AzureOpenAIResponsesClient(
                project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
                deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                credential=credential,
            ),
            name="FaceAgent",
            instructions=INSTRUCTIONS,
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