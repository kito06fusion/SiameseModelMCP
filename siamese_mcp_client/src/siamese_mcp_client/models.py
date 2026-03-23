from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    streamable_http_path: str
    transport: str
    registry_backend: str
    ann_enabled: bool
    accepted_extensions: list[str]
    image_transport: str


class FaceSearchMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    registry_name: str | None = None
    person_name: str | None = None
    original_filename: str | None = None
    expected_person_name: str | None = None
    reference_id: int | None = None
    name_match: bool | None = None
    face_match: bool | None = None
    match: bool | None = None
    accepted_name_match: bool | None = None
    accepted_as_match: bool | None = None
    status: str | None = None
    error: str | None = None
    verified: bool | None = None
    confidence: float | None = None
    distance: float | None = None
    threshold: float | None = None
    distance_margin: float | None = None
    model: str | None = None
    detector_backend: str | None = None
    distance_metric: str | None = None
    search_method: str | None = None
    raw_search_result: dict[str, Any] | None = None


class RegisterFaceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_filename: str
    person_name: str
    probe_image_encoding: str
    enforce_detection: bool
    align: bool
    expand_percentage: int


class RegisterFaceSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    metadata_id: int
    total_registry_entries: int
    deepface_registered_embeddings: int
    ann_index_rebuilt: bool
    elapsed_seconds: float


class RegisterFaceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request: RegisterFaceRequest
    summary: RegisterFaceSummary
    config: dict[str, Any]
    warnings: list[str] = []


class SearchFaceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_filename: str
    expected_person_name: str
    probe_image_encoding: str
    registry_backend: str
    search_method: str
    enforce_detection: bool
    align: bool
    expand_percentage: int


class SearchFaceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    detector_backend: str
    distance_metric: str
    normalization: str
    search_method: str


class SearchFaceSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    registry_entries: int
    raw_result_count: int
    match_found: bool
    top_hit_face_match: bool | None = None
    top_hit_name_matches_expected: bool
    elapsed_seconds: float


class SearchFaceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request: SearchFaceRequest
    config: SearchFaceConfig
    summary: SearchFaceSummary
    best_match: FaceSearchMatch | None = None
    matches: list[FaceSearchMatch]
    warnings: list[str] = []
