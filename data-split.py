import os
import csv
import cv2

CSV_FILE = "data/info.csv"
SOURCE_VIDEO_DIR = "data/source"
OUTPUT_DIR = "data/transform"
MIN_FRAMES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_video(row):
    source_video = row["source_video"]
    target_video = row["target_video"]
    is_flip = row.get("IsFlip", "F").upper() == "T"

    source_path = os.path.join(SOURCE_VIDEO_DIR, source_video)
    target_path = os.path.join(OUTPUT_DIR, target_video)

    start_frame = int(row["start_frame"]) - 1  # 0-indexed
    end_frame = int(row["end_frame"]) - 1

    duration = end_frame - start_frame + 1
    if duration < MIN_FRAMES:
        print(f"[SKIP] Too short ({duration} frame): {target_video}")
        return
    if not os.path.exists(source_path):
        print(f"[WARN] Source not found: {source_path}")
        return
    if os.path.exists(target_path):
        print(f"[SKIP] Exists: {target_path}")
        return

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MJPG = intra-frame, effectively uncompressed for ML
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply horizontal flip if needed
        if is_flip:
            frame = cv2.flip(frame, 1)
        
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()

    flip_status = "[FLIP]" if is_flip else "[ORIG]"
    print(f"[OK] {flip_status} Generated: {target_path}")

def main():
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_video(row)

if __name__ == "__main__":
    main()