from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from starlette.applications import Starlette

from .face_service import compare_face_to_registry

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PACKAGE_ROOT / "data" / "faces.json"
HOST = os.getenv("SIAMESE_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("SIAMESE_MCP_PORT", "8000"))

mcp = FastMCP(
    name="Siamese Face MCP",
    instructions=(
        "Look up reference faces by name in a JSON-backed registry and compare the "
        "probe face image only against those matching entries using DeepFace with "
        "ArcFace, MTCNN, and cosine distance."
    ),
    host=HOST,
    port=PORT,
)


@mcp.tool(
    name="compare_face_against_registry",
    description="Find registry faces by name and compare the probe face image only against those matching entries.",
)
async def compare_face_against_registry(
    image_path: str,
    name: str,
    ctx: Context,
    registry_path: str = str(DEFAULT_REGISTRY_PATH),
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    await ctx.info(
        f"Comparing probe image '{image_path}' against registry '{registry_path}' for requested name '{name}'."
    )
    return compare_face_to_registry(
        probe_image_path=image_path,
        requested_name=name,
        registry_path=registry_path,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )


@mcp.resource(
    "registry://faces",
    name="face_registry_config",
    description="Path and metadata for the default face registry used by the Siamese Face MCP server.",
    mime_type="application/json",
)
def get_default_registry() -> dict[str, Any]:
    return {
        "default_registry_path": str(DEFAULT_REGISTRY_PATH),
        "streamable_http_path": "/mcp",
        "transport": "streamable-http",
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
