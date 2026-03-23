# Siamese MCP Client

Python client for the local Siamese Face MCP server. It is designed for two use cases:

- smoke-testing the running MCP server during development
- reusing the same MCP client wrapper later inside a larger custom agent application

## What It Connects To

This client targets the Siamese Face MCP server over streamable HTTP:

```text
http://127.0.0.1:8000/mcp
```

The server should already be running before you use the client.

## Setup

From `siamese_mcp_client/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## CLI Smoke Tests

List tools exposed by the server:

```bash
python -m siamese_mcp_client.cli tools
```

Read the registry resource:

```bash
python -m siamese_mcp_client.cli registry
```

Call the face comparison tool:

```bash
python -m siamese_mcp_client.cli compare \
  --image /absolute/path/to/probe.jpg \
  --name Alice \
  --registry /absolute/path/to/faces.json
```

The compare command prints a short summary first and then the full JSON response.

## Reusable Python API

```python
import asyncio

from siamese_mcp_client import SiameseMcpClient


async def main() -> None:
    async with SiameseMcpClient("http://127.0.0.1:8000/mcp") as client:
        result = await client.compare_face(
            image_path="/absolute/path/to/probe.jpg",
            name="Alice",
            registry_path="/absolute/path/to/faces.json",
        )
        print(result.best_match)
        print(result.warnings)


asyncio.run(main())
```

## Future Agent Integration

For your larger application, keep the agent layer separate from this MCP transport layer:

- the agent decides when face verification is needed
- the agent calls `SiameseMcpClient.compare_face(...)`
- the client talks to the running MCP server
- the agent consumes typed fields like `best_match`, `matches`, `distance`, `threshold`, and `warnings`

This keeps your future Azure-backed LLM code independent from low-level MCP session management.

## Tests

Run the contract-style tests from `siamese_mcp_client/`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_client_contract.py"
```

These tests mock MCP responses and validate client-side parsing without needing the server to be live.


python -m siamese_mcp_client.cli compare --image "/Users/kitoruijter/Certificaten/myphoto.jpeg" --name Kito --registry /Users/kitoruijter/SiameseModelMCP/siamese_mcp/data/faces.json