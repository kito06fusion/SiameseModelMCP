"""Client package for the Siamese Face MCP server."""

from .client import DEFAULT_SERVER_URL, SiameseMcpClient, SiameseMcpClientError, SiameseMcpToolError
from .models import RegisterFaceResponse, SearchFaceResponse, ServiceInfo

__all__ = [
    "DEFAULT_SERVER_URL",
    "RegisterFaceResponse",
    "SearchFaceResponse",
    "SiameseMcpClient",
    "SiameseMcpClientError",
    "SiameseMcpToolError",
    "ServiceInfo",
]
