import os
import csv
import cv2

# Configuration for file paths
CSV_FILE = "data/info.csv"
# 1. Update this to the actual path of your source videos
SOURCE_VIDEO_DIR = "archive"

# Directory where processed video clips will be saved
OUTPUT_DIR = "data/transform" 
MIN_FRAMES = 3

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_video(row):
    source_video = row["source_video"]
    target_video = row["target_video"]
    is_flip = row.get("IsFlip", "F").upper() == "T"

    # 2. Extract action label from filename to determine sub-folder
    # Example: "person01_boxing_d1_uncomp.avi" -> split('_')[1] -> "boxing"
    action_name = source_video.split('_')[1]

    # 3. Construct full path: archive root + action folder + filename
    # Example: C:\Users\aabab\kth\archive\boxing\person01_boxing_d1_uncomp.avi
    source_path = os.path.join(SOURCE_VIDEO_DIR, action_name, source_video)
    target_path = os.path.join(OUTPUT_DIR, target_video)

    # Convert to 0-indexed frame numbers
    start_frame = int(row["start_frame"]) - 1  
    end_frame = int(row["end_frame"]) - 1

    duration = end_frame - start_frame + 1
    
    # Validation checks
    if duration < MIN_FRAMES:
        print(f"[SKIP] Too short ({duration} frames): {target_video}")
        return
    if not os.path.exists(source_path):
        print(f"[WARN] Source not found: {source_path}")
        return
    if os.path.exists(target_path):
        print(f"[SKIP] Exists: {target_path}")
        return

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source_path}")
        return

    # VideoWriter Configuration
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # Handle NaN or zero FPS
        fps = 25.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Using XVID codec for AVI output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height), isColor=False)

    # Seek to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    # Extraction loop
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Data Augmentation: Horizontal Flip
        if is_flip:
            frame = cv2.flip(frame, 1) # 1: horizontal flip
            
        # Ensure frame is grayscale for consistent processing
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    print(f"[SAVED] {target_video} ({duration} frames)")

def main():
    # Verify metadata file existence
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] CSV file not found: {CSV_FILE}")
        return

    # Load processing list from CSV
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total segments to process: {len(rows)}")
    
    # Process segments sequentially
    for i, row in enumerate(rows, 1):
        if i % 100 == 0:
            print(f"Processing {i}/{len(rows)}...")
        split_video(row)
        
    print("All processing finished.")

if __name__ == "__main__":
    main()