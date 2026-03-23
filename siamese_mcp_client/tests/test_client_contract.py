from __future__ import annotations

import unittest

import mcp.types as mcp_types

from siamese_mcp_client.client import SiameseMcpClient, SiameseMcpToolError


class FakeSession:
    def __init__(
        self,
        *,
        call_tool_result: mcp_types.CallToolResult | None = None,
        read_resource_result: mcp_types.ReadResourceResult | None = None,
    ) -> None:
        self._call_tool_result = call_tool_result
        self._read_resource_result = read_resource_result

    async def call_tool(self, name: str, arguments: dict[str, object] | None = None) -> mcp_types.CallToolResult:
        return self._call_tool_result  # type: ignore[return-value]

    async def read_resource(self, uri: str) -> mcp_types.ReadResourceResult:
        return self._read_resource_result  # type: ignore[return-value]


class SiameseMcpClientContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_face_parses_structured_content(self) -> None:
        client = SiameseMcpClient()
        client._session = FakeSession(  # type: ignore[assignment]
            call_tool_result=mcp_types.CallToolResult(
                content=[],
                structuredContent={
                    "request": {
                        "probe_image_path": "/tmp/probe.jpg",
                        "requested_name": "Alice",
                        "registry_path": "/tmp/faces.json",
                        "enforce_detection": True,
                        "align": True,
                        "expand_percentage": 0,
                    },
                    "config": {
                        "model_name": "ArcFace",
                        "detector_backend": "mtcnn",
                        "distance_metric": "cosine",
                        "normalization": "ArcFace",
                    },
                    "summary": {
                        "registry_entries": 3,
                        "candidate_entries": 1,
                        "successful_comparisons": 1,
                        "failed_comparisons": 0,
                        "requested_name_entries": 1,
                        "match_found": True,
                        "elapsed_seconds": 0.42,
                    },
                    "best_match": {
                        "registry_name": "Alice",
                        "reference_image_path": "/tmp/alice.jpg",
                        "status": "ok",
                        "verified": True,
                        "distance": 0.19,
                        "threshold": 0.68,
                    },
                    "matches": [
                        {
                            "registry_name": "Alice",
                            "reference_image_path": "/tmp/alice.jpg",
                            "status": "ok",
                            "verified": True,
                            "distance": 0.19,
                            "threshold": 0.68,
                        }
                    ],
                    "warnings": [],
                },
                isError=False,
            )
        )

        response = await client.compare_face(image_path="/tmp/probe.jpg", name="Alice")

        self.assertEqual(response.request.requested_name, "Alice")
        self.assertTrue(response.summary.match_found)
        self.assertEqual(response.best_match.registry_name, "Alice")
        self.assertAlmostEqual(response.best_match.distance, 0.19)

    async def test_compare_face_raises_on_tool_error(self) -> None:
        client = SiameseMcpClient()
        client._session = FakeSession(  # type: ignore[assignment]
            call_tool_result=mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text="bad request")],
                isError=True,
            )
        )

        with self.assertRaises(SiameseMcpToolError):
            await client.compare_face(image_path="/tmp/probe.jpg", name="Alice")

    async def test_get_registry_info_parses_resource_json(self) -> None:
        client = SiameseMcpClient()
        client._session = FakeSession(  # type: ignore[assignment]
            read_resource_result=mcp_types.ReadResourceResult(
                contents=[
                    mcp_types.TextResourceContents(
                        uri="registry://faces",
                        mimeType="application/json",
                        text=(
                            '{"default_registry_path":"/tmp/faces.json",'
                            '"streamable_http_path":"/mcp","transport":"streamable-http"}'
                        ),
                    )
                ]
            )
        )

        registry_info = await client.get_registry_info()

        self.assertEqual(registry_info.default_registry_path, "/tmp/faces.json")
        self.assertEqual(registry_info.streamable_http_path, "/mcp")
        self.assertEqual(registry_info.transport, "streamable-http")


if __name__ == "__main__":
    unittest.main()
