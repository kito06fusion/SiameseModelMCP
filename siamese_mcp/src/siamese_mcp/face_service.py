from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"
DISTANCE_METRIC = "cosine"
NORMALIZATION = "ArcFace"


class RegistryError(ValueError):
    """Raised when the face registry is malformed."""


def compare_face_to_registry(
    *,
    probe_image_path: str,
    requested_name: str,
    registry_path: str | Path,
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    registry_file = Path(registry_path).expanduser().resolve()
    probe_image = Path(probe_image_path).expanduser().resolve()

    if not probe_image.is_file():
        raise FileNotFoundError(f"Probe image does not exist: {probe_image}")
    if not registry_file.is_file():
        raise FileNotFoundError(f"Face registry does not exist: {registry_file}")

    registry_records = load_face_registry(registry_file)
    requested_name_key = requested_name.strip().casefold()
    candidate_records = [record for record in registry_records if record["name"].casefold() == requested_name_key]

    matches: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not registry_records:
        warnings.append("The face registry is empty.")
    if requested_name and not candidate_records:
        warnings.append(f"No registry entries were found for requested name '{requested_name}'.")

    for record in candidate_records:
        match_result: dict[str, Any] = {
            "registry_index": record["registry_index"],
            "registry_name": record["name"],
            "reference_image_path": str(record["resolved_image_path"]),
            "reference_image_path_raw": record["image_path"],
        }

        reference_image_path = record["resolved_image_path"]
        if not reference_image_path.is_file():
            error_message = f"Reference image does not exist: {reference_image_path}"
            match_result.update({"status": "error", "error": error_message})
            warnings.append(error_message)
            matches.append(match_result)
            continue

        try:
            verification = DeepFace.verify(
                img1_path=str(probe_image),
                img2_path=str(reference_image_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                distance_metric=DISTANCE_METRIC,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                normalization=NORMALIZATION,
                silent=True,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            match_result.update({"status": "error", "error": error_message})
            warnings.append(
                f"Comparison failed for registry entry '{record['name']}' at {reference_image_path}: {error_message}"
            )
            matches.append(match_result)
            continue

        threshold = verification.get("threshold")
        distance = verification.get("distance")
        match_result.update(
            {
                "status": "ok",
                "verified": verification.get("verified"),
                "distance": distance,
                "threshold": threshold,
                "distance_margin": (threshold - distance) if threshold is not None and distance is not None else None,
                "model": verification.get("model"),
                "detector_backend": verification.get("detector_backend"),
                "distance_metric": verification.get("similarity_metric", DISTANCE_METRIC),
                "facial_areas": verification.get("facial_areas"),
                "deepface_time_seconds": verification.get("time"),
                "raw_deepface_response": verification,
            }
        )
        matches.append(match_result)

    successful_matches = sorted(
        (match for match in matches if match.get("status") == "ok"),
        key=lambda match: match.get("distance", float("inf")),
    )

    finished_at = time.perf_counter()

    return {
        "request": {
            "probe_image_path": str(probe_image),
            "requested_name": requested_name,
            "registry_path": str(registry_file),
            "enforce_detection": enforce_detection,
            "align": align,
            "expand_percentage": expand_percentage,
        },
        "config": {
            "model_name": MODEL_NAME,
            "detector_backend": DETECTOR_BACKEND,
            "distance_metric": DISTANCE_METRIC,
            "normalization": NORMALIZATION,
        },
        "summary": {
            "registry_entries": len(registry_records),
            "candidate_entries": len(candidate_records),
            "successful_comparisons": len(successful_matches),
            "failed_comparisons": len(matches) - len(successful_matches),
            "requested_name_entries": len(candidate_records),
            "match_found": bool(successful_matches),
            "elapsed_seconds": round(finished_at - started_at, 4),
        },
        "best_match": successful_matches[0] if successful_matches else None,
        "matches": matches,
        "warnings": warnings,
    }


def load_face_registry(registry_path: Path) -> list[dict[str, Any]]:
    try:
        raw_data = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid JSON in registry file {registry_path}: {exc}") from exc

    if isinstance(raw_data, dict):
        faces = raw_data.get("faces")
    else:
        faces = raw_data

    if not isinstance(faces, list):
        raise RegistryError("Face registry must be a list or an object with a 'faces' list.")

    registry_records: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        if not isinstance(face, dict):
            raise RegistryError(f"Registry entry at index {index} must be an object.")

        name = face.get("name")
        image_path = face.get("image_path")
        if not isinstance(name, str) or not name.strip():
            raise RegistryError(f"Registry entry at index {index} is missing a valid 'name'.")
        if not isinstance(image_path, str) or not image_path.strip():
            raise RegistryError(f"Registry entry at index {index} is missing a valid 'image_path'.")

        resolved_image_path = resolve_registry_path(registry_path.parent, image_path)
        registry_records.append(
            {
                "registry_index": index,
                "name": name.strip(),
                "image_path": image_path,
                "resolved_image_path": resolved_image_path,
            }
        )

    return registry_records


def resolve_registry_path(base_directory: Path, path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_directory / candidate).resolve()
