"""
Ekstraktira frame-ove iz jednog videa i sprema ih kao slike.
"""

import cv2
from pathlib import Path
import argparse


def extract_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int = 250,
    fps_extract: float = 25.0,
):
    folder_name = video_path.stem
    save_dir = output_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Video:   {video_path.name}")
    print(f"Output:  {save_dir}")
    print(f"{'='*55}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Ne mogu otvoriti video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"FPS videa:       {video_fps:.1f}")
    print(f"Trajanje:        {duration:.1f}s")
    print(f"Ukupno frameova: {total_frames}")
    print(f"Cilj:            {max_frames} slika @ {fps_extract} fps")
    print(f"Očekivano:       ~{int(duration * fps_extract)} slika")

    frame_interval = max(1, int(video_fps / fps_extract))

    saved = 0
    frame_idx = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_idx += 1

        # Spremi bez resizea – originalna rezolucija
        filename = save_dir / f"frame_{saved + 1:04d}.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 25 == 0:
            print(f"  → {saved}/{max_frames} slika...")

    cap.release()

    print(f"\n📊 Rezultat:")
    print(f"  Spremljeno: {saved} slika")
    print(f"  Output:     {save_dir}")

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Putanja do videa npr. videos/eric_svjetlo.MOV",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Maksimalni broj slika (default: 250)",
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="Frameovi po sekundi (default: 25.0)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    video_path = base_dir / args.video
    output_dir = base_dir / "pictures"

    if not video_path.exists():
        print(f"❌ Video ne postoji: {video_path}")
        return

    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        max_frames=args.max_frames,
        fps_extract=args.fps,
    )


if __name__ == "__main__":
    main()
