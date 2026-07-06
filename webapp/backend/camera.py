"""
Kamera management – thread-safe pristup picamera2.
"""

import threading
import numpy as np
import time
from picamera2 import Picamera2


class CameraManager:
    """
    Singleton kamera manager.
    Osigurava thread-safe pristup kameri iz više threadova.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._picam2 = None
        self._capture_thread = None

    def start(self):
        """Pokreni kameru i capture thread."""
        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(2)  # AWB warmup

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("✅ Kamera pokrenuta")

    def _capture_loop(self):
        """Kontinuirano snima frameove u pozadini."""
        while self._running:
            frame = self._picam2.capture_array()
            with self._lock:
                self._frame = frame
            time.sleep(0.01)  # ~100 FPS capture, inference je sporiji

    def get_frame(self) -> np.ndarray:
        """Dohvati najnoviji frame (thread-safe)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        """Zaustavi kameru."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._picam2:
            self._picam2.stop()
        print("🛑 Kamera zaustavljena")
