"""
Pydantic schemas za FastAPI endpoints.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Detection(BaseModel):
    """Jedna detekcija osobe."""

    bbox: list[float]  # [cx, cy, w, h] normalizirano 0-1
    confidence: float  # YOLO confidence
    name: Optional[str]  # ime ako je poznata osoba
    face_score: float  # cosine similarity (-1.0 = nema lica)
    status: str  # "known" | "unknown" | "no_face"


class Metrics(BaseModel):
    """Real-time metrike sustava."""

    # Performanse modela
    fps: float
    inference_ms: float
    detection_ms: float
    recognition_ms: float

    # Sistem
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    temperature_c: float

    # Status
    pir_active: bool
    active_model: str
    num_persons: int
    timestamp: str


class ModelInfo(BaseModel):
    """Info o jednom modelu."""

    name: str  # display ime npr. "YOLOv8n TFLite INT8"
    filename: str  # ime fajla npr. "yolov8n_int8.tflite"
    format: str  # "tflite" | "onnx"
    quantization: str  # "int8" | "fp32"
    size_mb: float
    map_score: float  # mAP@0.5 iz naših benchmarkova
    benchmark_fps: float  # FPS iz benchmarkova


class ModelSwitchRequest(BaseModel):
    """Request za promjenu modela."""

    filename: str


class ModelSwitchResponse(BaseModel):
    """Response na promjenu modela."""

    success: bool
    message: str
    active_model: str


class Person(BaseModel):
    """Poznata osoba iz baze."""

    name: str
    num_images: int
    registered: str


class PersonsResponse(BaseModel):
    """Lista poznatih osoba."""

    persons: list[Person]
    total: int


class UnknownCluster(BaseModel):
    """Cluster nepoznatih osoba (DBSCAN)."""

    cluster_id: int
    seen_count: int
    first_seen: str
    last_seen: str
    thumbnail: Optional[str]  # base64 slika


class SystemInfo(BaseModel):
    """Opće info o sustavu."""

    hostname: str
    ip_address: str
    python_version: str
    uptime_seconds: float
    models_available: int
    persons_in_db: int
