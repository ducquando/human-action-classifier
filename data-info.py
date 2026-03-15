import csv
import re
import random
from collections import defaultdict

# Fixed seed for reproducible random sampling
random.seed(42)

# Input path for KTH sequence annotations
INPUT_FILE = "archive/00sequences.txt"
OUTPUT_FILE = "data/info.csv"

# --- [User Settings] ---
MIN_EMPTY_LEN = 10         # Discard empty gaps shorter than 10 frames
QUOTA_PER_SCENARIO = 100   # Target number of empty sequences to collect per scenario (d1-d4)
# ---------------------

FRAME_RANGE_PATTERN = re.compile(r"^\d+\s*-\s*\d+$")

# Classification: Upper body (static person) vs Lower body (empty background)
UPPER_BODY = {"boxing", "handclapping", "handwaving"}

def extract_label_and_scenario(video_id):
    # Example format: person01_walking_d1
    parts = video_id.split("_")
    action = parts[1] if len(parts) >= 3 else "unknown"
    scenario = parts[2] if len(parts) >= 3 else "unknown"
    return action, scenario

def main():
    final_rows = []
    
    # Storage for empty sequence candidates (categorized by scenario and body type)
    # empty_candidates['d1']['upper'] = [candidate_pair1, candidate_pair2...]
    empty_candidates = defaultdict(lambda: {"upper": [], "lower": []})

    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or "*missing*" in line.lower() or "frames" not in line.lower():
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue

            video_id = parts[0]
            action_label, scenario = extract_label_and_scenario(video_id)
            source_video = f"{video_id}_uncomp.avi"

            try:
                frames_idx = parts.index("frames")
            except ValueError:
                continue

            ranges_str = " ".join(parts[frames_idx + 1:])
            raw_ranges = ranges_str.split(",")

            # Parse frame ranges
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
            seq_counter = defaultdict(int)
            prev_end = None
            segment_id = 0

            for start, end in ranges:
                # 1. Processing 'Empty' (background/static) segments
                if prev_end is not None and start > prev_end + 1:
                    gap_start = prev_end + 1
                    gap_end = start - 1
                    gap_len = gap_end - gap_start + 1

                    # Only register as candidate if length meets MIN_EMPTY_LEN
                    if gap_len >= MIN_EMPTY_LEN:
                        label = "empty"
                        s_idx = seq_counter[label]
                        seq_counter[label] += 1
                        
                        # Treat (Original, Flipped) as a single candidate pair
                        candidate_pair = []
                        for is_flip, flip_suffix in [(False, "original"), (True, "flip")]:
                            target_video = f"{video_id.replace(action_label, label)}_s{s_idx}_{action_label}_{flip_suffix}.avi"
                            candidate_pair.append([
                                video_id, source_video, target_video, segment_id,
                                gap_start, gap_end, label, "T" if is_flip else "F"
                            ])
                        
                        # Distribute to 'upper' (rare) or 'lower' (common) based on action type
                        if action_label in UPPER_BODY:
                            empty_candidates[scenario]["upper"].append(candidate_pair)
                        else:
                            empty_candidates[scenario]["lower"].append(candidate_pair)
                        
                    segment_id += 1

                # 2. Processing 'Action' segments (All are included in CSV)
                label = action_label
                s_idx = seq_counter[label]
                seq_counter[label] += 1

                # Generate both original and flipped variants
                for is_flip, flip_suffix in [(False, "original"), (True, "flip")]:
                    target_video = f"{video_id}_s{s_idx}_{action_label}_{flip_suffix}.avi"
                    final_rows.append([
                        video_id, source_video, target_video, segment_id,
                        start, end, label, "T" if is_flip else "F"
                    ])

                segment_id += 1
                prev_end = end

    # ---------------------------------------------------------
    # 3. Quota-Based Sampling Logic for 'Empty' Class
    # ---------------------------------------------------------
    sampled_empty_count = 0
    print("\n[Empty Sampling Summary]")
    
    for sc, groups in empty_candidates.items():
        upper_list = groups["upper"]
        lower_list = groups["lower"]
        
        # Step 1: Priority - include all rare upper-body empty sequences
        selected = upper_list[:]
        
        # Step 2: Filling - randomly sample lower-body sequences to reach the quota
        needed = QUOTA_PER_SCENARIO - len(selected)
        if needed > 0:
            random.shuffle(lower_list) 
            selected.extend(lower_list[:needed]) 
            
        print(f"Scenario {sc.upper()}: Upper(all used)={len(upper_list)}, Lower(sampled)={min(needed, len(lower_list))} -> Total {len(selected)} sequences adopted")
        
        # Append selected empty pairs to final rows
        for candidate_pair in selected:
            final_rows.extend(candidate_pair)
            sampled_empty_count += 1

    # Write final metadata to CSV
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "video_id", "source_video", "target_video", "segment_id", 
            "start_frame", "end_frame", "label", "IsFlip"
        ])
        writer.writerows(final_rows)

    print(f"\n✅ Completed! Total {len(final_rows)} rows saved to {OUTPUT_FILE}. (Includes {sampled_empty_count} unique empty sequences)")

if __name__ == "__main__":
    main()