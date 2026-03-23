from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deepface import DeepFace

from . import db

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"
DISTANCE_METRIC = "cosine"
NORMALIZATION = "ArcFace"
SEARCH_METHOD = "ann"
REGISTRY_BACKEND = "postgres"
ACCEPTED_EXTENSIONS = {".jpg", ".jpeg"}


def jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    if not jpeg_bytes:
        raise ValueError("Empty JPEG bytes.")
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode JPEG image bytes.")
    return img


def decode_base64_jpeg(data: str) -> bytes:
    raw = data.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw, validate=False)


def normalize_filename(filename: str) -> str:
    normalized = Path(filename).name.strip()
    if not normalized:
        raise ValueError("A non-empty JPEG filename is required.")
    extension = Path(normalized).suffix.lower()
    if extension not in ACCEPTED_EXTENSIONS:
        raise ValueError("Only .jpg and .jpeg filenames are accepted.")
    return normalized


def derive_person_name(filename: str) -> str:
    person_name = Path(filename).stem.strip().casefold()
    if not person_name:
        raise ValueError("Filename stem must contain the person name.")
    return person_name


def build_ann_index() -> None:
    DeepFace.build_index(
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        align=True,
        database_type=REGISTRY_BACKEND,
        connection_details=db.get_postgres_uri(),
    )


def register_face_image(
    *,
    filename: str,
    image_jpeg_base64: str,
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    db.init_schema()
    normalized_filename = normalize_filename(filename)
    person_name = derive_person_name(normalized_filename)
    jpeg = decode_base64_jpeg(image_jpeg_base64)
    jpeg_bytes_to_bgr(jpeg)

    registration = DeepFace.register(
        img=io.BytesIO(jpeg),
        img_name=normalized_filename,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=NORMALIZATION,
        database_type=REGISTRY_BACKEND,
        connection_details=db.get_postgres_uri(),
    )
    metadata = db.upsert_face_reference(
        person_name=person_name,
        original_filename=normalized_filename,
        file_extension=Path(normalized_filename).suffix.lower(),
        image_jpeg=jpeg,
    )
    build_ann_index()
    finished_at = time.perf_counter()

    return {
        "request": {
            "original_filename": normalized_filename,
            "person_name": person_name,
            "probe_image_encoding": "base64_jpeg",
            "enforce_detection": enforce_detection,
            "align": align,
            "expand_percentage": expand_percentage,
        },
        "summary": {
            "metadata_id": int(metadata["id"]),
            "total_registry_entries": db.count_all_faces(),
            "deepface_registered_embeddings": int(registration.get("inserted", 0)),
            "ann_index_rebuilt": True,
            "elapsed_seconds": round(finished_at - started_at, 4),
        },
        "config": {
            "model_name": MODEL_NAME,
            "detector_backend": DETECTOR_BACKEND,
            "distance_metric": DISTANCE_METRIC,
            "normalization": NORMALIZATION,
            "registry_backend": REGISTRY_BACKEND,
            "search_method": SEARCH_METHOD,
        },
        "warnings": [],
    }


def search_face_image(
    *,
    filename: str,
    image_jpeg_base64: str,
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    db.init_schema()
    normalized_filename = normalize_filename(filename)
    expected_person_name = derive_person_name(normalized_filename)
    warnings: list[str] = []
    total_in_db = db.count_all_faces()

    if total_in_db == 0:
        warnings.append("The face registry (PostgreSQL) is empty.")
        finished_at = time.perf_counter()
        return {
            "request": {
                "original_filename": normalized_filename,
                "expected_person_name": expected_person_name,
                "probe_image_encoding": "base64_jpeg",
                "registry_backend": REGISTRY_BACKEND,
                "search_method": SEARCH_METHOD,
                "enforce_detection": enforce_detection,
                "align": align,
                "expand_percentage": expand_percentage,
            },
            "config": {
                "model_name": MODEL_NAME,
                "detector_backend": DETECTOR_BACKEND,
                "distance_metric": DISTANCE_METRIC,
                "normalization": NORMALIZATION,
                "search_method": SEARCH_METHOD,
            },
            "summary": {
                "registry_entries": 0,
                "raw_result_count": 0,
                "match_found": False,
                "top_hit_name_matches_expected": False,
                "elapsed_seconds": round(finished_at - started_at, 4),
            },
            "best_match": None,
            "matches": [],
            "warnings": warnings,
        }

    try:
        probe_jpeg = decode_base64_jpeg(image_jpeg_base64)
        jpeg_bytes_to_bgr(probe_jpeg)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid probe image: {exc}") from exc

    search_results = _search_with_ann(
        probe_jpeg=probe_jpeg,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )
    matches = _normalize_search_matches(
        search_results=search_results,
        expected_person_name=expected_person_name,
    )

    if len(search_results) > 1:
        warnings.append("Multiple faces were detected in the probe image; only the first face result is used.")

    finished_at = time.perf_counter()
    best_match = matches[0] if matches else None
    top_hit_name_matches_expected = bool(best_match and best_match.get("name_match"))
    top_hit_face_matches = bool(best_match and best_match.get("face_match"))
    top_hit_overall_match = bool(best_match and best_match.get("match"))

    if matches and not top_hit_name_matches_expected:
        warnings.append(
            f"ANN returned '{best_match.get('person_name')}' as the top match, but the expected filename person is '{expected_person_name}'."
        )
    if matches and not top_hit_face_matches:
        warnings.append(
            f"ANN returned a top hit for '{best_match.get('person_name')}', but the face similarity score did not pass threshold."
        )
    if not matches and total_in_db > 0:
        warnings.append("ANN search returned no verified matches for the supplied image.")

    return {
        "request": {
            "original_filename": normalized_filename,
            "expected_person_name": expected_person_name,
            "probe_image_encoding": "base64_jpeg",
            "registry_backend": REGISTRY_BACKEND,
            "search_method": SEARCH_METHOD,
            "enforce_detection": enforce_detection,
            "align": align,
            "expand_percentage": expand_percentage,
        },
        "config": {
            "model_name": MODEL_NAME,
            "detector_backend": DETECTOR_BACKEND,
            "distance_metric": DISTANCE_METRIC,
            "normalization": NORMALIZATION,
            "search_method": SEARCH_METHOD,
        },
        "summary": {
            "registry_entries": total_in_db,
            "raw_result_count": len(matches),
            "match_found": top_hit_overall_match,
            "top_hit_face_match": top_hit_face_matches,
            "top_hit_name_matches_expected": top_hit_name_matches_expected,
            "elapsed_seconds": round(finished_at - started_at, 4),
        },
        "best_match": best_match,
        "matches": matches,
        "warnings": warnings,
    }


def _search_with_ann(
    *, probe_jpeg: bytes, enforce_detection: bool, align: bool, expand_percentage: int
) -> list[Any]:
    try:
        return DeepFace.search(
            img=io.BytesIO(probe_jpeg),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=NORMALIZATION,
            database_type=REGISTRY_BACKEND,
            connection_details=db.get_postgres_uri(),
            search_method=SEARCH_METHOD,
        )
    except Exception:
        build_ann_index()
        return DeepFace.search(
            img=io.BytesIO(probe_jpeg),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=NORMALIZATION,
            database_type=REGISTRY_BACKEND,
            connection_details=db.get_postgres_uri(),
            search_method=SEARCH_METHOD,
        )


def _normalize_search_matches(
    *, search_results: list[Any], expected_person_name: str
) -> list[dict[str, Any]]:
    if not search_results:
        return []

    first_result = search_results[0]
    if hasattr(first_result, "to_dict"):
        rows = first_result.to_dict(orient="records")
    elif isinstance(first_result, dict):
        rows = [first_result]
    elif isinstance(first_result, list):
        rows = first_result
    else:
        raise TypeError(f"Unsupported DeepFace search result type: {type(first_result)!r}")

    matches: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        original_filename = str(row.get("img_name") or "").strip()
        metadata = db.fetch_face_reference_by_filename(original_filename) if original_filename else None
        person_name = (
            str(metadata["person_name"])
            if metadata is not None
            else (Path(original_filename).stem.casefold() if original_filename else None)
        )
        distance = row.get("distance")
        threshold = row.get("threshold")
        name_match = person_name == expected_person_name
        face_match = threshold is not None and distance is not None and distance <= threshold
        combined_match = bool(face_match and name_match)
        matches.append(
            {
                "rank": rank,
                "registry_name": person_name,
                "person_name": person_name,
                "expected_person_name": expected_person_name,
                "original_filename": original_filename or None,
                "reference_id": int(metadata["id"]) if metadata is not None else None,
                "name_match": name_match,
                "face_match": face_match,
                "match": combined_match,
                "accepted_name_match": name_match,
                "accepted_as_match": combined_match,
                "status": (
                    "ok"
                    if combined_match
                    else "face_mismatch"
                    if not face_match
                    else "name_mismatch"
                ),
                "verified": face_match,
                "confidence": row.get("confidence"),
                "distance": distance,
                "threshold": threshold,
                "distance_margin": (threshold - distance) if threshold is not None and distance is not None else None,
                "model": row.get("model_name", MODEL_NAME),
                "detector_backend": DETECTOR_BACKEND,
                "distance_metric": row.get("distance_metric", DISTANCE_METRIC),
                "search_method": row.get("search_method", SEARCH_METHOD),
                "raw_search_result": row,
            }
        )
    return matches
