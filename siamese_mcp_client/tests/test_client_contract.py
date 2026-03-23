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
        self.last_tool_name: str | None = None
        self.last_arguments: dict[str, object] | None = None

    async def call_tool(self, name: str, arguments: dict[str, object] | None = None) -> mcp_types.CallToolResult:
        self.last_tool_name = name
        self.last_arguments = arguments
        return self._call_tool_result  # type: ignore[return-value]

    async def read_resource(self, uri: str) -> mcp_types.ReadResourceResult:
        return self._read_resource_result  # type: ignore[return-value]


class SiameseMcpClientContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_register_face_image_parses_structured_content(self) -> None:
        client = SiameseMcpClient()
        fake_session = FakeSession(
            call_tool_result=mcp_types.CallToolResult(
                content=[],
                structuredContent={
                    "request": {
                        "original_filename": "alice.jpg",
                        "person_name": "alice",
                        "probe_image_encoding": "base64_jpeg",
                        "enforce_detection": True,
                        "align": True,
                        "expand_percentage": 0,
                    },
                    "summary": {
                        "metadata_id": 7,
                        "total_registry_entries": 3,
                        "deepface_registered_embeddings": 1,
                        "ann_index_rebuilt": True,
                        "elapsed_seconds": 0.42,
                    },
                    "config": {
                        "model_name": "ArcFace",
                        "detector_backend": "mtcnn",
                        "distance_metric": "cosine",
                        "normalization": "ArcFace",
                        "registry_backend": "postgres",
                        "search_method": "ann",
                    },
                    "warnings": [],
                },
                isError=False,
            )
        )
        client._session = fake_session  # type: ignore[assignment]

        response = await client.register_face_image(filename="alice.jpg", image_jpeg_base64="Zm9v")

        self.assertEqual(fake_session.last_tool_name, "register_face_image")
        self.assertEqual(fake_session.last_arguments, {"filename": "alice.jpg", "image_jpeg_base64": "Zm9v", "enforce_detection": True, "align": True, "expand_percentage": 0})
        self.assertEqual(response.request.person_name, "alice")
        self.assertEqual(response.summary.metadata_id, 7)
        self.assertTrue(response.summary.ann_index_rebuilt)

    async def test_search_face_image_parses_structured_content(self) -> None:
        client = SiameseMcpClient()
        fake_session = FakeSession(
            call_tool_result=mcp_types.CallToolResult(
                content=[],
                structuredContent={
                    "request": {
                        "original_filename": "alice.jpg",
                        "expected_person_name": "alice",
                        "probe_image_encoding": "base64_jpeg",
                        "registry_backend": "postgres",
                        "search_method": "ann",
                        "enforce_detection": True,
                        "align": True,
                        "expand_percentage": 0,
                    },
                    "config": {
                        "model_name": "ArcFace",
                        "detector_backend": "mtcnn",
                        "distance_metric": "cosine",
                        "normalization": "ArcFace",
                        "search_method": "ann",
                    },
                    "summary": {
                        "registry_entries": 3,
                        "raw_result_count": 1,
                        "match_found": True,
                        "top_hit_face_match": True,
                        "top_hit_name_matches_expected": True,
                        "elapsed_seconds": 0.42,
                    },
                    "best_match": {
                        "registry_name": "alice",
                        "person_name": "alice",
                        "original_filename": "alice.jpg",
                        "status": "ok",
                        "name_match": True,
                        "face_match": True,
                        "match": True,
                        "accepted_name_match": True,
                        "accepted_as_match": True,
                        "verified": True,
                        "distance": 0.19,
                        "threshold": 0.68,
                    },
                    "matches": [
                        {
                            "registry_name": "alice",
                            "person_name": "alice",
                            "original_filename": "alice.jpg",
                            "status": "ok",
                            "name_match": True,
                            "face_match": True,
                            "match": True,
                            "accepted_name_match": True,
                            "accepted_as_match": True,
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
        client._session = fake_session  # type: ignore[assignment]

        response = await client.search_face_image(filename="alice.jpg", image_jpeg_base64="Zm9v")

        self.assertEqual(fake_session.last_tool_name, "search_face_image")
        self.assertEqual(response.request.expected_person_name, "alice")
        self.assertTrue(response.summary.match_found)
        self.assertTrue(response.summary.top_hit_face_match)
        self.assertEqual(response.best_match.registry_name, "alice")
        self.assertTrue(response.best_match.face_match)
        self.assertTrue(response.best_match.name_match)
        self.assertTrue(response.best_match.match)
        self.assertAlmostEqual(response.best_match.distance, 0.19)

    async def test_search_face_image_raises_on_tool_error(self) -> None:
        client = SiameseMcpClient()
        client._session = FakeSession(  # type: ignore[assignment]
            call_tool_result=mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text="bad request")],
                isError=True,
            )
        )

        with self.assertRaises(SiameseMcpToolError):
            await client.search_face_image(filename="alice.jpg", image_jpeg_base64="Zm9v")

    async def test_get_service_info_parses_resource_json(self) -> None:
        client = SiameseMcpClient()
        client._session = FakeSession(  # type: ignore[assignment]
            read_resource_result=mcp_types.ReadResourceResult(
                contents=[
                    mcp_types.TextResourceContents(
                        uri="service://face-recognition",
                        mimeType="application/json",
                        text=(
                            '{"streamable_http_path":"/mcp","transport":"streamable-http",'
                            '"registry_backend":"postgres","ann_enabled":true,'
                            '"accepted_extensions":[".jpg",".jpeg"],"image_transport":"base64_jpeg"}'
                        ),
                    )
                ]
            )
        )

        service_info = await client.get_service_info()

        self.assertEqual(service_info.streamable_http_path, "/mcp")
        self.assertEqual(service_info.transport, "streamable-http")
        self.assertEqual(service_info.registry_backend, "postgres")
        self.assertTrue(service_info.ann_enabled)


if __name__ == "__main__":
    unittest.main()
