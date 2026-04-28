"""
Test kamere s picamera2.
Snima nekoliko frameova i sprema ih kao slike.
"""

from picamera2 import Picamera2
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent.parent / "detection" / "benchmark" / "results" / "camera_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_camera():
    print("=== Test kamere ===")
    
    picam2 = Picamera2()
    
    # Konfiguracija za inference (640x640)
    config = picam2.create_preview_configuration(
        main={"size": (640, 640), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    # Warmup
    print("Warmup kamere (2 sekunde)...")
    time.sleep(2)
    
    # Snimi 5 frameova i izmjeri FPS
    print("Snimam frameove...")
    times = []
    
    for i in range(5):
        start = time.perf_counter()
        frame = picam2.capture_array()
        end = time.perf_counter()
        times.append(end - start)
        
        # Spremi sliku
        output_path = OUTPUT_DIR / f"frame_{i:02d}.jpg"
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(output_path)
        print(f"  Frame {i+1}: {frame.shape} – {(end-start)*1000:.1f}ms – spremljeno: {output_path.name}")
    
    picam2.stop()
    
    avg_capture_ms = sum(times) / len(times) * 1000
    print(f"\n✅ Kamera radi!")
    print(f"  Rezolucija: {frame.shape}")
    print(f"  Avg capture time: {avg_capture_ms:.1f}ms")
    print(f"  Slike spremljene u: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_camera()