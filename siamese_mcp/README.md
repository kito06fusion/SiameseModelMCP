# Siamese Face MCP

Local FastMCP server for comparing a probe face image against JSON-backed reference faces selected by name with DeepFace.

## What It Does

- Accepts a local `image_path` and a `name`
- Loads a JSON registry from `data/faces.json` by default
- Filters the registry by the provided `name`
- Compares the probe image only against the matching face entries with:
  - `model_name="ArcFace"`
  - `detector_backend="mtcnn"`
  - `distance_metric="cosine"`
- Returns detailed comparison data, including per-entry distances, thresholds, verification flags, facial areas, and warnings

## Registry Format

Default registry file: `data/faces.json`

```json
{
  "faces": [
    {
      "name": "Alice",
      "image_path": "reference/alice_01.jpg"
    }
  ]
}
```

Relative `image_path` values are resolved from the directory that contains the registry file.

## Local Setup

From `siamese_mcp/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Run With Uvicorn

From `siamese_mcp/` after installing dependencies:

```bash
uvicorn siamese_mcp.server:app --app-dir src --host 127.0.0.1 --port 8000
```

The MCP streamable HTTP endpoint is exposed at:

```text
http://127.0.0.1:8000/mcp
```

You can also run the packaged entrypoint:

```bash
python -m siamese_mcp.server
```

## Available MCP Interfaces

- Tool: `compare_face_against_registry`
- Resource: `registry://faces`

## Tool Arguments

- `image_path`: local path to the probe image
- `name`: requested person name used to look up candidate reference faces in the registry
- `registry_path`: optional path to a registry JSON file
- `enforce_detection`: default `true`
- `align`: default `true`
- `expand_percentage`: default `0`

## Response Shape

The tool returns:

- `request`: normalized input values and runtime options
- `config`: the DeepFace configuration used
- `summary`: counts and elapsed time, including how many records matched the requested name
- `best_match`: the closest successful match among the entries for the requested `name`
- `matches`: all per-entry results for that name, including failures
- `warnings`: non-fatal issues such as missing reference images or a name that does not exist in the registry

Each successful match includes the raw DeepFace verification payload so downstream code can inspect the score, threshold, detector output, and facial areas directly.

## Notes

- The first call may download model assets required by DeepFace.
- Add your real reference images under `data/reference/` or point `image_path` entries to absolute paths elsewhere on disk.
- The starter `faces.json` includes a placeholder record that should be replaced with real data before production use.
