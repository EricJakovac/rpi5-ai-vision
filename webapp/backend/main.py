"""
FastAPI backend za RPi5 AI Vision sustav.
"""

import asyncio
import io
import json
import socket
import time
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import RPi.GPIO as GPIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image as PILImage, ImageDraw, ImageFont

from camera import CameraManager
from models import ModelManager, AVAILABLE_MODELS
from pipeline import InferencePipeline
from schemas import (
    ModelSwitchRequest,
    ModelSwitchResponse,
    PersonsResponse,
    Person,
    SystemInfo,
)

# ─── Inicijalizacija ──────────────────────────────────────────────────────────

PIR_PIN = 17
CAM_WIDTH = 1280
CAM_HEIGHT = 720

app = FastAPI(
    title="RPi5 AI Vision API",
    description="Live detekcija i prepoznavanje osoba na Raspberry Pi 5",
    version="1.0.0",
)

# CORS – dopusti React frontend (localhost + Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globalni objekti
camera = CameraManager(width=CAM_WIDTH, height=CAM_HEIGHT)
model_manager = ModelManager()
pipeline = InferencePipeline(camera, model_manager)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    print("\n🚀 RPi5 AI Vision API startanje...")

    # GPIO za PIR
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN)
    print("✅ GPIO inicijaliziran")

    # Pokreni kameru
    camera.start()

    # Učitaj defaultni model
    model_manager.load_model("yolov8n_int8.tflite")

    # Pokreni inference pipeline
    pipeline.start()

    # PIR monitoring u pozadini
    asyncio.create_task(pir_monitor())

    print("✅ Sustav spreman!\n")


@app.on_event("shutdown")
async def shutdown():
    pipeline.stop()
    camera.stop()
    GPIO.cleanup()
    print("🛑 Sustav ugašen")


# ─── PIR Monitor ─────────────────────────────────────────────────────────────


async def pir_monitor():
    """Async PIR monitoring loop."""
    POST_MOTION_DELAY = 5.0
    last_motion = 0.0

    while True:
        pir_state = GPIO.input(PIR_PIN)
        now = time.time()

        if pir_state == 1:
            last_motion = now
            if not pipeline.pir_active:
                print("🟢 PIR: Pokret detektiran")
                pipeline.pir_active = True
        else:
            if pipeline.pir_active and (now - last_motion > POST_MOTION_DELAY):
                print("💤 PIR: Nema pokreta")
                pipeline.pir_active = False

        await asyncio.sleep(0.1)


# ─── Video stream ─────────────────────────────────────────────────────────────


def draw_frame(frame: np.ndarray, detections: list) -> bytes:
    """Nacrtaj bounding boxove na frameu i vrati JPEG bytes."""
    img = PILImage.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        cx, cy, w, h = det.bbox
        x1 = int((cx - w / 2) * CAM_WIDTH)
        y1 = int((cy - h / 2) * CAM_HEIGHT)
        x2 = int((cx + w / 2) * CAM_WIDTH)
        y2 = int((cy + h / 2) * CAM_HEIGHT)

        if det.status == "known":
            color = (0, 255, 0)
            label = f"{det.name} ({det.face_score:.2f})"
        elif det.status == "unknown":
            color = (255, 0, 0)
            label = f"Nepoznata osoba ({det.face_score:.2f})"
        else:
            color = (0, 150, 255)
            label = f"Osoba ({det.confidence:.2f})"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox_text = draw.textbbox((x1, y1 - 28), label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0))
        draw.text((x1, y1 - 28), label, fill=color, font=font)

    # Konvertiraj u JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


async def generate_stream():
    """Generator za MJPEG stream."""
    while True:
        frame = pipeline.get_frame()
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        detections = pipeline.get_detections()
        jpeg = draw_frame(frame, detections)

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

        await asyncio.sleep(0.1)  # ~10 FPS stream


@app.get("/stream")
async def video_stream():
    """MJPEG live stream s bounding boxovima."""
    return StreamingResponse(
        generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─── WebSocket metrike ────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket za real-time metrike i detekcije."""
    await websocket.accept()
    print(f"WebSocket spojen: {websocket.client}")

    try:
        while True:
            metrics = pipeline.get_metrics()
            detections = pipeline.get_detections()

            data = {
                "metrics": metrics.model_dump(),
                "detections": [d.model_dump() for d in detections],
                "timestamp": datetime.now().isoformat(),
            }

            await websocket.send_json(data)
            await asyncio.sleep(0.5)  # update svake 0.5s

    except WebSocketDisconnect:
        print(f"WebSocket odspojoen: {websocket.client}")
    except Exception as e:
        print(f"WebSocket greška: {e}")


# ─── REST Endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def get_metrics():
    """Trenutne metrike sustava."""
    return pipeline.get_metrics()


@app.get("/detections")
async def get_detections():
    """Zadnje detekcije."""
    return pipeline.get_detections()


@app.get("/models")
async def get_models():
    """Lista dostupnih modela."""
    models = ModelManager.list_available()
    current = model_manager.get_current_info()
    return {"available": models, "current": current}


@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Promijeni aktivni model."""
    success = model_manager.load_model(request.filename)
    current = model_manager.get_current_info()
    return ModelSwitchResponse(
        success=success,
        message="Model promijenjen" if success else "Greška pri promjeni modela",
        active_model=current.get("name", "N/A"),
    )


@app.get("/persons")
async def get_persons():
    """Lista poznatih osoba iz baze."""
    db_path = (
        Path(__file__).parent.parent.parent
        / "ai"
        / "recognition"
        / "face_database.json"
    )
    if not db_path.exists():
        return PersonsResponse(persons=[], total=0)

    with open(db_path) as f:
        db = json.load(f)

    persons = [
        Person(
            name=name,
            num_images=data.get("num_images", 0),
            registered=data.get("registered", ""),
        )
        for name, data in db.get("persons", {}).items()
    ]
    return PersonsResponse(persons=persons, total=len(persons))


@app.get("/system")
async def get_system_info():
    """Opće info o sustavu."""
    import psutil

    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "N/A"

    db_path = (
        Path(__file__).parent.parent.parent
        / "ai"
        / "recognition"
        / "face_database.json"
    )
    persons_count = 0
    if db_path.exists():
        with open(db_path) as f:
            db = json.load(f)
        persons_count = len(db.get("persons", {}))

    return SystemInfo(
        hostname=hostname,
        ip_address=ip,
        python_version=sys.version.split()[0],
        uptime_seconds=time.time() - pipeline._start_time,
        models_available=len(ModelManager.list_available()),
        persons_in_db=persons_count,
    )
