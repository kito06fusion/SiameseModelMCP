from __future__ import annotations

import base64
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from .models import RegisterFaceResponse, SearchFaceResponse, ServiceInfo

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/mcp"
REGISTER_TOOL_NAME = "register_face_image"
SEARCH_TOOL_NAME = "search_face_image"
SERVICE_RESOURCE_URI = "service://face-recognition"


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

        # TODO: Open the HTTP client, create the streamable HTTP transport,
        # create a ClientSession, initialize it, and store the results on self.
        raise NotImplementedError("Workshop exercise: implement connect().")

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._session_id_callback = None
        self._initialize_result = None

    async def list_tools(self) -> list[mcp_types.Tool]:
        # TODO: Ask the active MCP session for the available tools and return them.
        raise NotImplementedError("Workshop exercise: implement list_tools().")

    async def get_service_info(self) -> ServiceInfo:
        # TODO: Read the service resource, extract its JSON payload,
        # and validate it into a ServiceInfo model.
        raise NotImplementedError("Workshop exercise: implement get_service_info().")

    async def register_face_image(
        self,
        *,
        filename: str,
        image_jpeg_base64: str,
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
    ) -> RegisterFaceResponse:
        # TODO: Build the tool arguments, call the register tool,
        # handle MCP errors, and validate the JSON response.
        raise NotImplementedError("Workshop exercise: implement register_face_image().")

    async def search_face_image(
        self,
        *,
        filename: str,
        image_jpeg_base64: str,
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
    ) -> SearchFaceResponse:
        # TODO: Build the tool arguments, call the search tool,
        # handle MCP errors, and validate the JSON response.
        raise NotImplementedError("Workshop exercise: implement search_face_image().")

    async def register_face_file(
        self,
        *,
        image_path: str,
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
    ) -> RegisterFaceResponse:
        return await self.register_face_image(
            filename=Path(image_path).name,
            image_jpeg_base64=_encode_image_file_to_base64(image_path),
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )

    async def search_face_file(
        self,
        *,
        image_path: str,
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
    ) -> SearchFaceResponse:
        return await self.search_face_image(
            filename=Path(image_path).name,
            image_jpeg_base64=_encode_image_file_to_base64(image_path),
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )

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


def _encode_image_file_to_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
