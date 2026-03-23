"""Client package for the Siamese Face MCP server."""

from .client import DEFAULT_SERVER_URL, SiameseMcpClient, SiameseMcpClientError, SiameseMcpToolError
from .models import CompareFaceResponse, RegistryInfo

__all__ = [
    "CompareFaceResponse",
    "DEFAULT_SERVER_URL",
    "RegistryInfo",
    "SiameseMcpClient",
    "SiameseMcpClientError",
    "SiameseMcpToolError",
]
