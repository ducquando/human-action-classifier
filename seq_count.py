from collections import defaultdict
import os, re, glob

# Action labels including the 'empty' class
ACTIONS = ["walking", "jogging", "running", "boxing", "handclapping", "handwaving", "empty"]
LABEL2ID = {a: i for i, a in enumerate(ACTIONS)}

# Training: Subjects 1 to 17
TRAIN_SUBJECTS = list(range(1, 18))

# Testing: Subjects 18 to 25
TEST_SUBJECTS = list(range(18, 26))

def get_segments_from_txt(txt_path):
    segments_map = {}
    if not os.path.exists(txt_path):
        return segments_map

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'frames' not in line: 
                continue
            
            # 1. Clearly separate the filename and the frame information section.
            # Usually separated by a tab (\t) or whitespace.
            parts = re.split(r'\s+frames\s+', line)
            if len(parts) < 2:
                continue
            
            filename = parts[0].strip() # e.g., "person01_walking_d1"
            frame_info = parts[1].strip() # e.g., "1-100, 105-200..."
            
            # 2. Extract numbers only from the frame_info part.
            frame_nums = re.findall(r'\d+', frame_info)
            frame_nums = [int(n) for n in frame_nums]
            
            valid_ranges = []
            for i in range(0, len(frame_nums), 2):
                if i + 1 < len(frame_nums):
                    start_f = frame_nums[i]
                    end_f = frame_nums[i+1]
                    # Only add if start frame is less than end frame
                    if start_f < end_f:
                        valid_ranges.append((start_f, end_f))
            
            if valid_ranges:
                # Calculate empty_ranges (intervals between actions)
                empty_ranges = []
                for i in range(len(valid_ranges) - 1):
                    e_start = valid_ranges[i][1] + 1
                    e_end = valid_ranges[i+1][0] - 1
                    if e_start < e_end:
                        empty_ranges.append((e_start, e_end))
                
                segments_map[filename] = {
                    'action': valid_ranges,
                    'empty': empty_ranges
                }
    return segments_map

def parse_kth(fn: str):
    """Parse KTH filename and return (person_id, class_id)."""
    # Example filename: person01_boxing_d1_uncomp.avi
    m = re.search(r"person(\d+)_([a-z]+)_", os.path.basename(fn))
    if not m:
        return None
    pid = int(m.group(1))
    act = m.group(2)
    if act not in LABEL2ID:
        return None
    return pid, LABEL2ID[act]

def list_videos(kth_dir: str):
    """Collect all .avi files and parse metadata."""
    vids = sorted(glob.glob(os.path.join(kth_dir, "**", "*.avi"), recursive=True))
    print("KTH_DIR =", kth_dir)
    print("AVI files found =", len(vids))
    
    if len(vids) > 0:
        print("Sample avi path =", vids[0])
        print("Sample basename =", os.path.basename(vids[0]))

    items = []
    for v in vids:
        p = parse_kth(v)
        if p is None:
            print("parse_kth failed for:", os.path.basename(v))
            continue
        pid, y = p
        items.append((v, pid, y))

    print("Parsed items =", len(items))
    return items

def analyze_dataset_distribution(items, segments_map, actions_list):
    # Dictionary to store statistics
    # { class_id: { 'seq_count': 0, 'total_frames': 0, 'frame_lengths': [] } }
    stats = defaultdict(lambda: {'seq_count': 0, 'total_frames': 0, 'frame_lengths': []})
    
    for v_path, pid, y_class in items:
        v_name = os.path.basename(v_path).replace('.avi', '').replace('_uncomp', '')
        if v_name not in segments_map:
            continue
        
        # 1) Analyze Action segments
        for start_f, end_f in segments_map[v_name]['action']:
            length = end_f - start_f + 1
            stats[y_class]['seq_count'] += 1
            stats[y_class]['total_frames'] += length
            stats[y_class]['frame_lengths'].append(length)
            
        # 2) Analyze Empty segments (Class ID 6)
        for start_f, end_f in segments_map[v_name]['empty']:
            length = end_f - start_f + 1
            stats[6]['seq_count'] += 1
            stats[6]['total_frames'] += length
            stats[6]['frame_lengths'].append(length)

    # Print results
    print("\n" + "="*85)
    print(f"{'Class Name (ID)':<20} | {'Seq Count':<12} | {'Total Frames':<15} | {'Avg Frames/Seq':<15}")
    print("-" * 85)
    
    for i, action in enumerate(actions_list):
        s = stats[i]
        avg_len = sum(s['frame_lengths']) / len(s['frame_lengths']) if s['seq_count'] > 0 else 0
        print(f"{f'{action} ({i})':<20} | {s['seq_count']:<12} | {s['total_frames']:<15} | {avg_len:<15.2f}")
    
    print("="*85)
    return stats

def analyze_detailed_distribution(items, segments_map, actions_list):
    # stats structure: { action_id: { 'act_seq': 0, 'act_frames': 0, 'emp_seq': 0, 'emp_frames': 0 } }
    # Stores action segments and corresponding empty segments for each of the 6 action classes.
    stats = defaultdict(lambda: {'act_seq': 0, 'act_frames': 0, 'emp_seq': 0, 'emp_frames': 0})
    
    total_empty_seq = 0
    total_empty_frames = 0

    for v_path, pid, y_class in items:
        # File name processing to match keys in the segments_map
        v_name = os.path.basename(v_path).replace('.avi', '').replace('_uncomp', '')
        if v_name not in segments_map:
            continue
        
        # 1) Action segment statistics for this specific video
        for start_f, end_f in segments_map[v_name]['action']:
            length = end_f - start_f + 1
            stats[y_class]['act_seq'] += 1
            stats[y_class]['act_frames'] += length
            
        # 2) Empty (interval) segment statistics within this action's video
        # Uses y_class (0-5) as the key to record that this "empty" data originated from that specific action's video.
        for start_f, end_f in segments_map[v_name]['empty']:
            length = end_f - start_f + 1
            stats[y_class]['emp_seq'] += 1
            stats[y_class]['emp_frames'] += length
            
            # Totals for summary
            total_empty_seq += 1
            total_empty_frames += length

    # Print results
    print("\n" + "="*95)
    header = f"{'Action Name (ID)':<18} | {'Act Seq':<8} | {'Act Frames':<10} | {'Emp Seq':<8} | {'Emp Frames':<10} | {'Avg Emp Len':<10}"
    print(header)
    print("-" * 95)
    
    for i in range(len(actions_list) - 1): # Iterate only through action classes 0-5
        s = stats[i]
        action_name = actions_list[i]
        avg_emp = s['emp_frames'] / s['emp_seq'] if s['emp_seq'] > 0 else 0
        
        print(f"{f'{action_name} ({i})':<18} | "
              f"{s['act_seq']:<8} | "
              f"{s['act_frames']:<10} | "
              f"{s['emp_seq']:<8} | "
              f"{s['emp_frames']:<10} | "
              f"{avg_emp:<10.2f}")
    
    print("-" * 95)
    avg_total_emp = total_empty_frames / total_empty_seq if total_empty_seq > 0 else 0
    print(f"{'TOTAL EMPTY (6)':<18} | {'-':<8} | {'-':<10} | "
          f"{total_empty_seq:<8} | "
          f"{total_empty_frames:<10} | "
          f"{avg_total_emp:.2f}")
    print("="*95)

    return stats

def main(KTH_DIR, txt_path):
    items = list_videos(KTH_DIR)
    segments_map = get_segments_from_txt(txt_path)

    # Analyze detailed distribution of the entire dataset (including empty segments per action)
    print("\n[Detailed Dataset Analysis (Action vs Empty per Action)]")
    analyze_detailed_distribution(items, segments_map, ACTIONS)

    # Split into training and testing pools
    train_pool = [it for it in items if it[1] in TRAIN_SUBJECTS]
    test_pool  = [it for it in items if it[1] in TEST_SUBJECTS]

    # Analyze detailed distribution for the training set
    print("\n[Training Dataset Detailed Analysis]")
    analyze_detailed_distribution(train_pool, segments_map, ACTIONS)
    
    # ... (rest of the logic)

if __name__ == "__main__":
    KTH_DIR = r"C:\Users\aabab\kth\archive"
    TXT_PATH = r"C:\Users\aabab\kth\00sequences.txt"
    main(KTH_DIR, TXT_PATH)