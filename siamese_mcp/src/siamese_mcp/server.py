from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from starlette.applications import Starlette

from .face_service import register_face_image, search_face_image

HOST = os.getenv("SIAMESE_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("SIAMESE_MCP_PORT", "8000"))

mcp = FastMCP(
    name="Siamese Face MCP",
    instructions=(
        "Register and search JPEG face images using DeepFace backed by PostgreSQL. "
        "The person name is always derived from the JPEG filename stem, and ANN "
        "searches are only accepted when the matched identity equals that stem."
    ),
    host=HOST,
    port=PORT,
)



@mcp.tool(
    name="register_face_file",
    description="Register a JPEG image from a file path accessible to the server (e.g. /shared/person.jpg).",
)
async def register_face_file_tool(
    file_path: str,
    ctx: Context,
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    await ctx.info(f"Registering file '{file_path}' into the PostgreSQL-backed DeepFace registry.")
    path = Path(file_path)
    image_jpeg_base64 = base64.b64encode(path.read_bytes()).decode()
    return register_face_image(
        filename=path.name,
        image_jpeg_base64=image_jpeg_base64,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )


@mcp.tool(
    name="search_face_file",
    description="Search the DeepFace ANN index using a JPEG image from a file path accessible to the server (e.g. /shared/person.jpg).",
)
async def search_face_file_tool(
    file_path: str,
    ctx: Context,
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    await ctx.info(f"Searching the PostgreSQL-backed DeepFace ANN index using file '{file_path}'.")
    path = Path(file_path)
    image_jpeg_base64 = base64.b64encode(path.read_bytes()).decode()
    return search_face_image(
        filename=path.name,
        image_jpeg_base64=image_jpeg_base64,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )


@mcp.resource(
    "service://face-recognition",
    name="face_service_info",
    description="Transport and capability metadata for the Siamese Face MCP server.",
    mime_type="application/json",
)
def get_service_info() -> dict[str, Any]:
    return {
        "streamable_http_path": "/mcp",
        "transport": "streamable-http",
        "registry_backend": "postgres",
        "ann_enabled": True,
        "accepted_extensions": [".jpg", ".jpeg"],
        "image_transport": "base64_jpeg",
    }


def create_app() -> Starlette:
    return mcp.streamable_http_app()


app = create_app()


def main() -> None:
    uvicorn.run(
        "siamese_mcp.server:create_app",
        factory=True,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
