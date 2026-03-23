# Siamese Face MCP

FastMCP server for JPEG-only face enrollment and ANN search with DeepFace, PostgreSQL, and Docker.

## What Changed

The server no longer reads a JSON registry or host file paths.

- Images are sent to the MCP tool as base64 JPEG bytes.
- The identity comes from the filename stem.
  - `kitoruijter.jpeg` becomes `kitoruijter`
- Enrollment uses `DeepFace.register(...)`.
- ANN search uses `DeepFace.build_index()` plus `DeepFace.search(..., search_method="ann")`.
- A search is only accepted when the top ANN hit matches the filename-derived identity.

## MCP Interfaces

- Tool: `register_face_image`
- Tool: `search_face_image`
- Resource: `service://face-recognition`

## Tool Inputs

Both tools accept:

- `filename`: original `.jpg` or `.jpeg` filename
- `image_jpeg_base64`: base64-encoded JPEG bytes
- `enforce_detection`: default `true`
- `align`: default `true`
- `expand_percentage`: default `0`

## Identity Rules

- Only `.jpg` and `.jpeg` filenames are accepted.
- The filename stem is normalized with `casefold()`.
- Registered metadata is stored in the app table `face_reference`.
- DeepFace stores embeddings and the ANN index in PostgreSQL.
- Search results are treated as a match only when `best_match.person_name == Path(filename).stem.casefold()`.

## Docker Run

From the repo root:

```bash
cp .env.example .env
docker compose up --build
```

The MCP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

## Environment

The Docker setup configures these variables for the server:

- `DEEPFACE_POSTGRES_URI`
- `SIAMESE_APP_POSTGRES_URI`
- `SIAMESE_MCP_HOST`
- `SIAMESE_MCP_PORT`

Outside Docker, you must set the two Postgres URI variables yourself before running `python -m siamese_mcp.server`.

## Local Development

From `siamese_mcp/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export DEEPFACE_POSTGRES_URI="postgresql://deepface:deepface@127.0.0.1:5432/deepface"
export SIAMESE_APP_POSTGRES_URI="$DEEPFACE_POSTGRES_URI"
python -m siamese_mcp.server
```

## Response Shape

`register_face_image` returns:

- `request`: normalized filename-derived identity and runtime options
- `summary`: metadata row id, embedding insert count, index rebuild flag, elapsed time
- `config`: DeepFace model and backend settings
- `warnings`

`search_face_image` returns:

- `request`: normalized filename-derived identity and search options
- `config`: DeepFace model and ANN settings
- `summary`: registry size, raw ANN result count, match decision, elapsed time
- `best_match`
- `matches`
- `warnings`

## Notes

- The first DeepFace call may download model assets and take longer.
- `register_face_image` rebuilds the ANN index immediately after each enrollment.
- Re-registering the same filename updates app metadata and stores another embedding with the same `img_name` in DeepFace.
