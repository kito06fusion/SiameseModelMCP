from __future__ import annotations

import argparse
import asyncio

from .client import DEFAULT_SERVER_URL, SiameseMcpClient


def _print_search_summary(result) -> None:
    print()
    print("Most important info:")
    print(f"- Expected name: {result.request.expected_person_name}")
    print(f"- Match found: {result.summary.match_found}")
    print(f"- Raw results: {result.summary.raw_result_count}")
    print(f"- Time: {result.summary.elapsed_seconds}s")

    if result.best_match is None:
        print("- Best match: none")
    else:
        print(f"- Best match: {result.best_match.person_name}")
        print(f"- Face match: {result.best_match.face_match}")
        print(f"- Name match: {result.best_match.name_match}")
        print(f"- Overall match: {result.best_match.match}")
        print(f"- Distance: {result.best_match.distance}")
        print(f"- Threshold: {result.best_match.threshold}")
        print(f"- Distance margin: {result.best_match.distance_margin}")
        print(f"- Confidence: {result.best_match.confidence}")
        print(f"- Registered filename: {result.best_match.original_filename}")

    if result.warnings:
        print("- Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")


def _print_register_summary(result) -> None:
    print()
    print("Registration summary:")
    print(f"- Person name: {result.request.person_name}")
    print(f"- Original filename: {result.request.original_filename}")
    print(f"- Metadata id: {result.summary.metadata_id}")
    print(f"- Registered embeddings: {result.summary.deepface_registered_embeddings}")
    print(f"- ANN index rebuilt: {result.summary.ann_index_rebuilt}")
    print(f"- Registry entries: {result.summary.total_registry_entries}")
    print(f"- Time: {result.summary.elapsed_seconds}s")
    if result.warnings:
        print("- Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")


async def _run_register(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        result = await client.register_face_file(
            image_path=args.image,
            enforce_detection=not args.disable_detection,
            align=not args.disable_align,
            expand_percentage=args.expand_percentage,
        )
    print(result.model_dump_json(indent=2))
    _print_register_summary(result)
    return 0


async def _run_search(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        result = await client.search_face_file(
            image_path=args.image,
            enforce_detection=not args.disable_detection,
            align=not args.disable_align,
            expand_percentage=args.expand_percentage,
        )

    print(f"Expected person: {result.request.expected_person_name}")
    print(f"Probe filename: {result.request.original_filename}")
    print(f"Raw result count: {result.summary.raw_result_count}")
    if result.best_match is not None:
        print(f"Best match name: {result.best_match.person_name}")
        print(f"Best match distance: {result.best_match.distance}")
        print(f"Best face match: {result.best_match.face_match}")
        print(f"Best name match: {result.best_match.name_match}")
        print(f"Overall match: {result.best_match.match}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    print()
    print(result.model_dump_json(indent=2))
    _print_search_summary(result)
    return 0


async def _run_service(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        service_info = await client.get_service_info()
    print(service_info.model_dump_json(indent=2))
    return 0


async def _run_tools(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        tools = await client.list_tools()
    for tool in tools:
        print(f"{tool.name}: {tool.description or ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test client for the Siamese Face MCP server.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="MCP streamable HTTP endpoint URL.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Enroll a JPEG into the face registry.")
    register_parser.add_argument("--image", required=True, help="Local path to the JPEG image.")
    register_parser.add_argument("--expand-percentage", type=int, default=0, help="DeepFace expand percentage.")
    register_parser.add_argument(
        "--disable-detection",
        action="store_true",
        help="Disable face detection enforcement for low-quality inputs.",
    )
    register_parser.add_argument(
        "--disable-align",
        action="store_true",
        help="Disable alignment before verification.",
    )
    register_parser.set_defaults(handler=_run_register)

    search_parser = subparsers.add_parser("search", help="Search the ANN index using a JPEG image.")
    search_parser.add_argument("--image", required=True, help="Local path to the JPEG image.")
    search_parser.add_argument("--expand-percentage", type=int, default=0, help="DeepFace expand percentage.")
    search_parser.add_argument(
        "--disable-detection",
        action="store_true",
        help="Disable face detection enforcement for low-quality inputs.",
    )
    search_parser.add_argument(
        "--disable-align",
        action="store_true",
        help="Disable alignment before ANN search.",
    )
    search_parser.set_defaults(handler=_run_search)

    service_parser = subparsers.add_parser("service", help="Read the service info resource.")
    service_parser.set_defaults(handler=_run_service)

    tools_parser = subparsers.add_parser("tools", help="List tools exposed by the MCP server.")
    tools_parser.set_defaults(handler=_run_tools)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(args.handler(args)))


if __name__ == "__main__":
    main()
