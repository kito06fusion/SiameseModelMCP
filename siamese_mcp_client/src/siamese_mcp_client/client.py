from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from .models import CompareFaceResponse, RegistryInfo

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"
COMPARE_TOOL_NAME = "compare_face_against_registry"
REGISTRY_RESOURCE_URI = "registry://faces"


class SiameseMcpClientError(RuntimeError):
    """Base error raised by the Siamese MCP client."""


class SiameseMcpToolError(SiameseMcpClientError):
    """Raised when the MCP server returns a tool error."""


class SiameseMcpClient:
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.server_url = server_url
        self.headers = headers
        self._session: ClientSession | None = None
        self._session_id_callback: Any | None = None
        self._stack: AsyncExitStack | None = None
        self._initialize_result: mcp_types.InitializeResult | None = None

    async def __aenter__(self) -> SiameseMcpClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    @property
    def session_id(self) -> str | None:
        if self._session_id_callback is None:
            return None
        return self._session_id_callback()

    @property
    def server_info(self) -> mcp_types.Implementation | None:
        if self._initialize_result is None:
            return None
        return self._initialize_result.serverInfo

    async def connect(self) -> None:
        if self._session is not None:
            return

        stack = AsyncExitStack()
        try:
            http_client = create_mcp_http_client(headers=self.headers)
            await stack.enter_async_context(http_client)
            read_stream, write_stream, session_id_callback = await stack.enter_async_context(
                streamable_http_client(self.server_url, http_client=http_client)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            initialize_result = await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self._session = session
        self._session_id_callback = session_id_callback
        self._initialize_result = initialize_result

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._session_id_callback = None
        self._initialize_result = None

    async def list_tools(self) -> list[mcp_types.Tool]:
        session = self._require_session()
        result = await session.list_tools()
        return result.tools

    async def get_registry_info(self) -> RegistryInfo:
        session = self._require_session()
        result = await session.read_resource(REGISTRY_RESOURCE_URI)
        payload = self._extract_json_from_resource(result)
        return RegistryInfo.model_validate(payload)

    async def compare_face(
        self,
        *,
        image_path: str,
        name: str,
        registry_path: str | None = None,
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
    ) -> CompareFaceResponse:
        session = self._require_session()
        arguments: dict[str, Any] = {
            "image_path": image_path,
            "name": name,
            "enforce_detection": enforce_detection,
            "align": align,
            "expand_percentage": expand_percentage,
        }
        if registry_path is not None:
            arguments["registry_path"] = registry_path

        result = await session.call_tool(COMPARE_TOOL_NAME, arguments=arguments)
        if result.isError:
            raise SiameseMcpToolError(self._tool_error_message(result))

        payload = self._extract_json_from_tool_result(result)
        return CompareFaceResponse.model_validate(payload)

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise SiameseMcpClientError("Client is not connected. Use 'async with' or call connect() first.")
        return self._session

    def _extract_json_from_tool_result(self, result: mcp_types.CallToolResult) -> dict[str, Any]:
        if result.structuredContent is not None:
            return result.structuredContent

        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                try:
                    parsed = json.loads(block.text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        raise SiameseMcpClientError("Tool result did not contain structured JSON content.")

    def _extract_json_from_resource(self, result: mcp_types.ReadResourceResult) -> dict[str, Any]:
        for content in result.contents:
            if isinstance(content, mcp_types.TextResourceContents):
                parsed = json.loads(content.text)
                if isinstance(parsed, dict):
                    return parsed
        raise SiameseMcpClientError("Resource did not contain JSON text content.")

    def _tool_error_message(self, result: mcp_types.CallToolResult) -> str:
        text_parts = [
            block.text
            for block in result.content
            if isinstance(block, mcp_types.TextContent) and block.text.strip()
        ]
        if text_parts:
            return "\n".join(text_parts)
        return "The MCP server returned an error for the tool call."
