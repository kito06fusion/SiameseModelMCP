from __future__ import annotations

import argparse
import asyncio

from .client import DEFAULT_SERVER_URL, SiameseMcpClient


def _print_key_summary(result) -> None:
    print()
    print("Most important info:")
    print(f"- Name: {result.request.requested_name}")
    print(f"- Match found: {result.summary.match_found}")
    print(f"- Candidate entries: {result.summary.candidate_entries}")
    print(f"- Successful comparisons: {result.summary.successful_comparisons}")
    print(f"- Time: {result.summary.elapsed_seconds}s")

    if result.best_match is None:
        print("- Best match: none")
    else:
        confidence = None
        if result.best_match.raw_deepface_response is not None:
            confidence = result.best_match.raw_deepface_response.get("confidence")

        print(f"- Best match: {result.best_match.registry_name}")
        print(f"- Verified: {result.best_match.verified}")
        print(f"- Distance: {result.best_match.distance}")
        print(f"- Threshold: {result.best_match.threshold}")
        print(f"- Distance margin: {result.best_match.distance_margin}")
        print(f"- Confidence: {confidence}")
        print(f"- Reference image: {result.best_match.reference_image_path}")

    if result.warnings:
        print("- Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")


async def _run_compare(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        result = await client.compare_face(
            image_path=args.image,
            name=args.name,
            registry_path=args.registry,
            enforce_detection=not args.disable_detection,
            align=not args.disable_align,
            expand_percentage=args.expand_percentage,
        )

    print(f"Requested name: {result.request.requested_name}")
    print(f"Probe image: {result.request.probe_image_path}")
    print(f"Candidate entries: {result.summary.candidate_entries}")
    print(f"Successful comparisons: {result.summary.successful_comparisons}")
    if result.best_match is not None:
        print(f"Best match name: {result.best_match.registry_name}")
        print(f"Best match distance: {result.best_match.distance}")
        print(f"Verified: {result.best_match.verified}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    print()
    print(result.model_dump_json(indent=2))
    _print_key_summary(result)
    return 0


async def _run_registry(args: argparse.Namespace) -> int:
    async with SiameseMcpClient(server_url=args.server_url) as client:
        registry = await client.get_registry_info()
    print(registry.model_dump_json(indent=2))
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

    compare_parser = subparsers.add_parser("compare", help="Call the face-comparison tool.")
    compare_parser.add_argument("--image", required=True, help="Local path to the probe image.")
    compare_parser.add_argument("--name", required=True, help="Name to look up in the face registry.")
    compare_parser.add_argument("--registry", help="Optional path to a registry JSON file.")
    compare_parser.add_argument("--expand-percentage", type=int, default=0, help="DeepFace expand percentage.")
    compare_parser.add_argument(
        "--disable-detection",
        action="store_true",
        help="Disable face detection enforcement for low-quality inputs.",
    )
    compare_parser.add_argument(
        "--disable-align",
        action="store_true",
        help="Disable alignment before verification.",
    )
    compare_parser.set_defaults(handler=_run_compare)

    registry_parser = subparsers.add_parser("registry", help="Read the registry config resource.")
    registry_parser.set_defaults(handler=_run_registry)

    tools_parser = subparsers.add_parser("tools", help="List tools exposed by the MCP server.")
    tools_parser.set_defaults(handler=_run_tools)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(args.handler(args)))


if __name__ == "__main__":
    main()
