from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


def get_postgres_uri() -> str:
    uri = os.getenv("SIAMESE_APP_POSTGRES_URI") or os.getenv("DEEPFACE_POSTGRES_URI")
    if not uri:
        raise RuntimeError(
            "Set SIAMESE_APP_POSTGRES_URI or DEEPFACE_POSTGRES_URI before starting the MCP server."
        )
    return uri


def get_connection() -> psycopg.Connection[Any]:
    return psycopg.connect(get_postgres_uri(), autocommit=True, row_factory=dict_row)


def init_schema() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS face_reference (
                id BIGSERIAL PRIMARY KEY,
                person_name TEXT NOT NULL,
                original_filename TEXT NOT NULL UNIQUE,
                file_extension TEXT NOT NULL,
                image_jpeg BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_face_reference_person_name
            ON face_reference (person_name)
            """
        )


def upsert_face_reference(
    *, person_name: str, original_filename: str, file_extension: str, image_jpeg: bytes
) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_reference (person_name, original_filename, file_extension, image_jpeg)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (original_filename) DO UPDATE
            SET
                person_name = EXCLUDED.person_name,
                file_extension = EXCLUDED.file_extension,
                image_jpeg = EXCLUDED.image_jpeg,
                updated_at = NOW()
            RETURNING id, person_name, original_filename, file_extension, created_at, updated_at
            """,
            (person_name, original_filename, file_extension, image_jpeg),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert face reference metadata.")
        return dict(row)


def fetch_face_reference_by_filename(original_filename: str) -> dict[str, Any] | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, person_name, original_filename, file_extension, created_at, updated_at
            FROM face_reference
            WHERE original_filename = %s
            """,
            (original_filename,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None


def count_all_faces() -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS count FROM face_reference")
        row = cur.fetchone()
        return int(row["count"]) if row is not None else 0
