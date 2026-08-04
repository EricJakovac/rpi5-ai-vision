"""
Model management – učitavanje i switching modela.
"""

import threading
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter
import onnxruntime as ort

# Putanja do modela
MODELS_DIR = Path(__file__).parent.parent.parent / "ai" / "detection" / "models"

# Dostupni modeli s metapodacima iz naših benchmarkova
AVAILABLE_MODELS = [
    {
        "name": "YOLOv8n v2 TFLite INT8 ⚡",
        "filename": "yolov8n_v2_ft_int8.tflite",
        "format": "tflite",
        "quantization": "int8",
        "map_score": 0.835,
        "benchmark_fps": 25.6,
    },
    {
        "name": "YOLOv11n v2 TFLite INT8",
        "filename": "yolo11n_v2_ft_int8.tflite",
        "format": "tflite",
        "quantization": "int8",
        "map_score": 0.839,
        "benchmark_fps": 20.5,
    },
    {
        "name": "YOLOv8n v2 TFLite FP32",
        "filename": "yolov8n_v2_ft_fp32.tflite",
        "format": "tflite",
        "quantization": "fp32",
        "map_score": 0.847,
        "benchmark_fps": 7.1,
    },
    {
        "name": "YOLOv8s v2 TFLite FP32 🎯",
        "filename": "yolov8s_v2_ft_fp32.tflite",
        "format": "tflite",
        "quantization": "fp32",
        "map_score": 0.895,
        "benchmark_fps": 2.8,
    },
    {
        "name": "YOLOv10n ONNX FP32 (baseline)",
        "filename": "yolov10n_fp32.onnx",
        "format": "onnx",
        "quantization": "fp32",
        "map_score": 0.602,
        "benchmark_fps": 6.7,
    },
]


class ModelManager:
    """
    Upravljanje detekcijskim modelima.
    Podržava TFLite i ONNX formate.
    Thread-safe switching modela.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._interpreter = None
        self._session = None
        self._current_model = None
        self._input_details = None
        self._output_details = None
        self._format = None

    def load_model(self, filename: str) -> bool:
        """Učitaj model po nazivu fajla."""
        model_info = self._find_model_info(filename)
        if not model_info:
            print(f"❌ Model nije u listi dostupnih: {filename}")
            return False

        fmt = model_info["format"]

        if fmt == "tflite":
            path = MODELS_DIR / "tflite" / model_info["quantization"] / filename
        else:
            path = MODELS_DIR / "onnx" / model_info["quantization"] / filename

        if not path.exists():
            print(f"❌ Model fajl ne postoji: {path}")
            return False

        print(f"Učitavam model: {filename}...")

        try:
            with self._lock:
                if fmt == "tflite":
                    interp = Interpreter(model_path=str(path), num_threads=4)
                    interp.allocate_tensors()
                    self._interpreter = interp
                    self._input_details = interp.get_input_details()
                    self._output_details = interp.get_output_details()
                    self._session = None
                else:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    sess = ort.InferenceSession(
                        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
                    self._session = sess
                    self._interpreter = None
                    self._input_details = None
                    self._output_details = None

                self._current_model = model_info
                self._format = fmt

            print(f"✅ Model učitan: {filename}")
            return True

        except Exception as e:
            print(f"❌ Greška pri učitavanju modela: {e}")
            return False

    def get_current_info(self) -> dict:
        """Vrati info o trenutno aktivnom modelu."""
        with self._lock:
            if not self._current_model:
                return {}
            info = self._current_model.copy()
            if self._format == "tflite":
                fmt = "tflite"
                quant_dir = self._current_model["quantization"]
                path = (
                    MODELS_DIR / "tflite" / quant_dir / self._current_model["filename"]
                )
            else:
                fmt = "onnx"
                quant_dir = self._current_model["quantization"]
                path = MODELS_DIR / "onnx" / quant_dir / self._current_model["filename"]
            info["size_mb"] = (
                round(path.stat().st_size / 1024 / 1024, 1) if path.exists() else 0
            )
            return info

    def get_interpreter(self):
        """Dohvati TFLite interpreter (thread-safe)."""
        with self._lock:
            return self._interpreter, self._input_details, self._output_details

    def get_session(self):
        """Dohvati ONNX session (thread-safe)."""
        with self._lock:
            return self._session

    def get_format(self) -> str:
        with self._lock:
            return self._format

    def _find_model_info(self, filename: str) -> dict:
        for m in AVAILABLE_MODELS:
            if m["filename"] == filename:
                return m
        return None

    @staticmethod
    def list_available() -> list:
        result = []
        for m in AVAILABLE_MODELS:
            fmt = m["format"]
            quant = m["quantization"]
            path = MODELS_DIR / fmt / quant / m["filename"]
            if path.exists():
                info = m.copy()
                info["size_mb"] = round(path.stat().st_size / 1024 / 1024, 1)
                result.append(info)
        return result
