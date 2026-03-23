from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RegistryInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    default_registry_path: str
    streamable_http_path: str
    transport: str


class FaceMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    registry_index: int | None = None
    registry_name: str | None = None
    reference_image_path: str | None = None
    reference_image_path_raw: str | None = None
    status: str | None = None
    error: str | None = None
    verified: bool | None = None
    distance: float | None = None
    threshold: float | None = None
    distance_margin: float | None = None
    model: str | None = None
    detector_backend: str | None = None
    distance_metric: str | None = None
    facial_areas: dict[str, Any] | None = None
    deepface_time_seconds: float | None = None
    raw_deepface_response: dict[str, Any] | None = None


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    probe_image_path: str
    requested_name: str
    registry_path: str
    enforce_detection: bool
    align: bool
    expand_percentage: int


class CompareConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    detector_backend: str
    distance_metric: str
    normalization: str


class CompareSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    registry_entries: int
    candidate_entries: int
    successful_comparisons: int
    failed_comparisons: int
    requested_name_entries: int
    match_found: bool
    elapsed_seconds: float


class CompareFaceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request: CompareRequest
    config: CompareConfig
    summary: CompareSummary
    best_match: FaceMatch | None = None
    matches: list[FaceMatch]
    warnings: list[str] = []
