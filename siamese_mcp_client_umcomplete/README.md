# Siamese MCP Client

Python client for the Dockerized Siamese Face MCP server.

## What It Connects To

```text
http://127.0.0.1:8000/mcp
```

The server is expected to be running already, usually via `docker compose up`.

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

Read the service resource:

```bash
python -m siamese_mcp_client.cli service
```

Register a face:

```bash
python -m siamese_mcp_client.cli register \
  --image /absolute/path/to/kitoruijter.jpg
```

Search the ANN index:

```bash
python -m siamese_mcp_client.cli search \
  --image /absolute/path/to/kitoruijter.jpg
```

The CLI reads the local JPEG, base64-encodes it, and sends only `filename` plus `image_jpeg_base64` to the MCP server.

## Reusable Python API

```python
import asyncio

from siamese_mcp_client import SiameseMcpClient


async def main() -> None:
    async with SiameseMcpClient("http://127.0.0.1:8000/mcp") as client:
        await client.register_face_file(image_path="/absolute/path/to/kitoruijter.jpg")
        result = await client.search_face_file(image_path="/absolute/path/to/kitoruijter.jpg")
        print(result.best_match)
        print(result.warnings)


asyncio.run(main())
```

## Agent Contract

This client matches the new MCP rules:

- only `.jpg` and `.jpeg` files are supported
- the person identity comes from the filename stem
- ANN search is global across the database
- a result is accepted only when the best ANN hit matches the filename-derived identity

## Tests

Run the contract-style tests from `siamese_mcp_client/`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_client_contract.py"
```

These tests mock MCP responses and validate client-side parsing without needing the server to be live.