import cv2
import os
import numpy as np

def extract_frames(
    video_path,
    output_dir="assets/frames",
    fps=1,          # ← frames-per-second you want to sample
    max_frames=None # ← optional cap on total frames
):
    """
    Extracts exactly `fps` frames per second from the video,
    up to `max_frames` total.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration_sec = total_frames / orig_fps

    # how many samples in total:
    num_target = int(duration_sec * fps)
    if max_frames:
        num_target = min(num_target, max_frames)

    # pick evenly‐spaced frame indices from 0 to total_frames-1
    indices = np.linspace(0, total_frames - 1, num=num_target, dtype=int)

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue
        path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1

    cap.release()
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])