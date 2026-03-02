import csv
import re
from collections import defaultdict

INPUT_FILE = "data/info.txt"
OUTPUT_FILE = "data/info.csv"

FRAME_RANGE_PATTERN = re.compile(r"^\d+\s*-\s*\d+$")
rows = []

def extract_label(video_id):
    # Assumes format: personXX_<label>_dY
    parts = video_id.split("_")
    return parts[1] if len(parts) >= 3 else "unknown"

def main():
    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip missing annotations
            if "*missing*" in line.lower():
                continue

            # Must contain 'frames'
            if "frames" not in line.lower():
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue

            video_id = parts[0]
            action_label = extract_label(video_id)
            source_video = f"{video_id}_uncomp.avi"

            try:
                frames_idx = parts.index("frames")
            except ValueError:
                continue

            ranges_str = " ".join(parts[frames_idx + 1:])
            raw_ranges = ranges_str.split(",")

            # Parse valid ranges
            ranges = []
            for fr in raw_ranges:
                fr = fr.strip()
                if not FRAME_RANGE_PATTERN.match(fr):
                    continue
                start_str, end_str = fr.split("-")
                start, end = int(start_str), int(end_str)
                if start < end:
                    ranges.append((start, end))

            if not ranges:
                continue

            ranges.sort()

            # Sequence counters PER LABEL
            seq_counter = defaultdict(int)

            prev_end = None
            segment_id = 0

            for start, end in ranges:
                # Insert empty gap if exists
                if prev_end is not None and start > prev_end + 1:
                    gap_start = prev_end + 1
                    gap_end = start - 1

                    label = "empty"
                    s_idx = seq_counter[label]
                    seq_counter[label] += 1

                    # Generate 2 outputs: original and flipped
                    for is_flip, flip_suffix in [(False, "original"), (True, "flip")]:
                        target_video = (
                            f"{video_id.replace(action_label, label)}_s{s_idx}_{action_label}_{flip_suffix}.avi"
                        )

                        rows.append([
                            video_id,
                            source_video,
                            target_video,
                            segment_id,
                            gap_start,
                            gap_end,
                            label,
                            "T" if is_flip else "F"
                        ])
                    segment_id += 1

                # Insert labeled action segment
                label = action_label
                s_idx = seq_counter[label]
                seq_counter[label] += 1

                # Generate 2 outputs: original and flipped
                for is_flip, flip_suffix in [(False, "original"), (True, "flip")]:
                    target_video = (
                        f"{video_id}_s{s_idx}_{action_label}_{flip_suffix}.avi"
                    )

                    rows.append([
                        video_id,
                        source_video,
                        target_video,
                        segment_id,
                        start,
                        end,
                        label,
                        "T" if is_flip else "F"
                    ])

                segment_id += 1
                prev_end = end

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "video_id",
            "source_video",
            "target_video",
            "segment_id",
            "start_frame",
            "end_frame",
            "label",
            "IsFlip"
        ])
        writer.writerows(rows)

    print(f"Saved {len(rows)} segments to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()