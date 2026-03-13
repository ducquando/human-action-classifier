import os, re, glob
import numpy as np
import cv2
import hashlib
from pathlib import Path
import random

# Reproducibility
random.seed(0)
np.random.seed(0)

from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb  # Requires: pip install xgboost

# -----------------------
# 1) Dataset indexing
# -----------------------
# Action labels including the 'empty' class
ACTIONS = ["walking", "jogging", "running", "boxing", "handclapping", "handwaving", "empty"]
LABEL2ID = {a: i for i, a in enumerate(ACTIONS)}

# Training: Subjects 1 to 17
TRAIN_SUBJECTS = list(range(1, 18))

# Testing: Subjects 18 to 25
TEST_SUBJECTS = list(range(18, 26))

print(f"TRAIN_SUBJECTS: {TRAIN_SUBJECTS}")
print(f"TEST_SUBJECTS: {TEST_SUBJECTS}")

def get_segments_from_txt(txt_path):
    segments_map = {}
    if not os.path.exists(txt_path):
        return segments_map

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'frames' not in line: 
                continue
            
            # 1. Clearly separate the filename and frame information.
            # They are usually separated by tabs (\t) or spaces.
            parts = re.split(r'\s+frames\s+', line)
            if len(parts) < 2:
                continue
            
            filename = parts[0].strip() # "person01_walking_d1"
            frame_info = parts[1].strip() # "1-100, 105-200..."
            
            # 2. Extract numbers strictly from the frame_info section.
            frame_nums = re.findall(r'\d+', frame_info)
            frame_nums = [int(n) for n in frame_nums]
            
            valid_ranges = []
            for i in range(0, len(frame_nums), 2):
                if i + 1 < len(frame_nums):
                    start_f = frame_nums[i]
                    end_f = frame_nums[i+1]
                    # Append only if the start frame is strictly less than the end frame
                    if start_f < end_f:
                        valid_ranges.append((start_f, end_f))
            
            if valid_ranges:
                # Calculate empty_ranges (same as previous logic)
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

# -----------------------
# 2) Local descriptors (HOG + HOF)
# -----------------------
def hof_patch(flow_u, flow_v, x, y, patch=16, bins=8):
    """Histogram of Optical Flow (HOF) on a local patch."""
    h, w = flow_u.shape
    r = patch // 2
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)

    u = flow_u[y0:y1, x0:x1]
    v = flow_v[y0:y1, x0:x1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)
    ang = (ang + np.pi) * (bins / (2*np.pi))

    hist = np.zeros((bins,), dtype=np.float32)
    idx = np.floor(ang).astype(np.int32)
    idx = np.clip(idx, 0, bins-1)

    for b in range(bins):
        hist[b] = mag[idx == b].sum()

    s = hist.sum()
    if s > 1e-8:
        hist /= s
    return hist

def hog_patch(img_g, x, y, patch=16, bins=9):
    """Histogram of Oriented Gradients (HOG) on a local patch."""
    h, w = img_g.shape
    r = patch // 2
    x0, x1 = max(1, x - r), min(w - 1, x + r)
    y0, y1 = max(1, y - r), min(h - 1, y + r)

    patch_img = img_g[y0:y1, x0:x1].astype(np.float32)

    gx = patch_img[:, 2:] - patch_img[:, :-2]
    gy = patch_img[2:, :] - patch_img[:-2, :]

    ph = min(gx.shape[0], gy.shape[0])
    pw = min(gx.shape[1], gy.shape[1])
    gx = gx[:ph, :pw]
    gy = gy[:ph, :pw]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)
    ang = ang * (bins / np.pi)

    hist = np.zeros((bins,), dtype=np.float32)
    idx = np.floor(ang).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)

    for b in range(bins):
        hist[b] = mag[idx == b].sum()

    s = hist.sum()
    if s > 1e-8:
        hist /= s
    return hist

# -----------------------
# 2.5) Descriptor caching utilities
# -----------------------
def _cache_key(video_path: str, params: dict) -> str:
    """Create a unique cache key based on video path and extraction parameters."""
    raw = video_path + "|" + "|".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _cache_path(cache_dir: str, video_path: str, params: dict) -> str:
    """Return a cache filename for a given video and parameter set."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path, params)
    base = os.path.splitext(os.path.basename(video_path))[0]
    return str(Path(cache_dir) / f"{base}_{key}_v3.npy")

def extract_descriptors_segmented(video_path, start_f, end_f, 
                                 resize=(160,120), grid=8, patch=16, 
                                 hog_bins=18, hof_bins=16, max_desc=1000):
    """Extract local descriptors (HOG + HOF) within a specific frame range [start_f, end_f]."""
    cap = cv2.VideoCapture(video_path)
    
    # Move to start frame (OpenCV is 0-indexed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f - 1)
    
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.array([])

    prev_g = cv2.cvtColor(cv2.resize(prev, resize), cv2.COLOR_BGR2GRAY)
    descs = []
    
    curr_f = start_f
    while curr_f < end_f:
        ok, frame = cap.read()
        if not ok: break
        
        curr_f += 1
        frame_res = cv2.resize(frame, resize)
        g = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY)

        # Farneback Optical Flow for HOF
        flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        h, w = g.shape
        for yy in range(patch//2, h - patch//2, grid):
            for xx in range(patch//2, w - patch//2, grid):
                hog = hog_patch(g, xx, yy, patch=patch, bins=hog_bins)
                hof = hof_patch(flow[...,0], flow[...,1], xx, yy, patch=patch, bins=hof_bins)
                descs.append(np.concatenate([hog, hof]))
        
        prev_g = g
        if len(descs) > max_desc * 2: break

    cap.release()
    
    if len(descs) == 0: return np.array([])
    X = np.vstack(descs).astype(np.float32)
    if X.shape[0] > max_desc:
        idx = np.random.choice(X.shape[0], max_desc, replace=False)
        X = X[idx]
    return X

# -----------------------
# 3) BoVW + classifier
# -----------------------
def build_codebook(train_desc_list, k=400, seed=0):
    """Fit k-means codebook on training descriptors."""
    X = np.vstack([d for d in train_desc_list if d.shape[0] > 0])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10000)
    km.fit(X)
    return km

def bovw_hist(desc, codebook):
    """Convert descriptors into a BoVW histogram."""
    k = codebook.n_clusters
    if desc.shape[0] == 0:
        return np.zeros((k,), dtype=np.float32)
    words = codebook.predict(desc)
    hist = np.bincount(words, minlength=k).astype(np.float32)

    s = hist.sum()
    if s > 1e-8:
        hist /= s
    return hist

def main(KTH_DIR, txt_path):
    items = list_videos(KTH_DIR)
    segments_map = get_segments_from_txt(txt_path)

    # Split data based on Subject IDs
    train_pool = [it for it in items if it[1] in TRAIN_SUBJECTS]
    test_pool  = [it for it in items if it[1] in TEST_SUBJECTS]

    print(f"Total videos found: {len(items)}")
    print(f"Train pool size: {len(train_pool)}")
    print(f"Test pool size: {len(test_pool)}")
    
    if len(train_pool) == 0:
       print("Check: Ensure TRAIN_SUBJECTS list matches personIDs in filenames.")

    # 1) Extract training descriptors (Segmented)
    print("Extracting training descriptors by segments...")
    action_data = []
    empty_data = []

    for v_path, pid, y_class in train_pool:
        v_name = os.path.basename(v_path).replace('.avi', '').replace('_uncomp', '')
        if v_name not in segments_map: continue
        
        # Extract Action segments
        for start_f, end_f in segments_map[v_name]['action']:
            d = extract_descriptors_segmented(v_path, start_f, end_f, hog_bins=18, hof_bins=16)
            if d.size > 0:
                action_data.append((d, y_class))
                
        # Extract Empty segments
        for start_f, end_f in segments_map[v_name]['empty']:
            d = extract_descriptors_segmented(v_path, start_f, end_f, hog_bins=18, hof_bins=16)
            if d.size > 0:
                empty_data.append((d, 6))

    # [Strategy 1] Undersampling the 'empty' class to balance the dataset
    random.shuffle(empty_data)
    sampled_empty = empty_data[:150] 

    # Construct final training set
    train_desc = []
    y_train = []
    for d, y in action_data + sampled_empty:
        train_desc.append(d)
        y_train.append(y)

    print(f"Final training set: {len(action_data)} action segments + {len(sampled_empty)} empty segments.")

    # 2) Build visual vocabulary
    print(f"Building codebook with {len(train_desc)} segments...")
    codebook = build_codebook(train_desc, k=2400, seed=0)

    # 3) Build histograms
    print("Building training histograms...")
    X_train = np.vstack([bovw_hist(d, codebook) for d in train_desc])
    y_train = np.array(y_train)

    # 4) Extract testing data
    print("Extracting testing descriptors by segments...")
    X_test = []
    y_test = []
    for v_path, pid, y_class in test_pool:
        v_name = os.path.basename(v_path).replace('.avi', '').replace('_uncomp', '')
        if v_name not in segments_map: continue

        for start_f, end_f in segments_map[v_name]['action']:
            d = extract_descriptors_segmented(v_path, start_f, end_f, hog_bins=18, hof_bins=16)
            X_test.append(bovw_hist(d, codebook))
            y_test.append(y_class)
            
        for start_f, end_f in segments_map[v_name]['empty']:
            d = extract_descriptors_segmented(v_path, start_f, end_f, hog_bins=18, hof_bins=16)
            X_test.append(bovw_hist(d, codebook))
            y_test.append(6)

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # -----------------------
    # 5) Classifier Experiments
    # -----------------------
    print("\n--- Starting Classifier Experiments ---")
    
    models = {
        "Baseline (Chi2 + SVM)": make_pipeline(
            AdditiveChi2Sampler(sample_steps=2),
            LinearSVC(C=4.0, max_iter=20000, random_state=0)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, class_weight='balanced'),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, random_state=0),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    }

    results = {}
    
    for name, clf in models.items():
        print(f"\n[Training {name}]...")
        clf.fit(X_train, y_train)
        
        print(f"[Predicting {name}]...")
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        results[name] = acc

        print(f"[{name}] Test Accuracy: {acc:.4f}")
        print(f"[{name}] Confusion Matrix:\n{cm}")

    # Summary
    print("\n" + "="*30)
    print("      EXPERIMENT SUMMARY")
    print("="*30)
    for name, acc in results.items():
        print(f"{name:25s}: {acc:.4f}")
    print("="*30)
    
    print(f"Total Train Segments: {len(train_desc)}")
    print(f"Total Test Segments: {len(X_test)}")

import matplotlib.pyplot as plt

def plot_histograms(video_path):
    """Utility to visualize HOG/HOF for a sample frame."""
    cap = cv2.VideoCapture(video_path)
    ok, frame1 = cap.read()
    ok, frame2 = cap.read()
    cap.release()

    if not ok:
        print("Could not read video.")
        return

    img1 = cv2.cvtColor(cv2.resize(frame1, (160, 120)), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.resize(frame2, (160, 120)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    cx, cy = 80, 60 
    hog_h = hog_patch(img2, cx, cy, patch=16, bins=18)
    hof_h = hof_patch(flow[..., 0], flow[..., 1], cx, cy, patch=16, bins=16)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(len(hog_h)), hog_h, color='skyblue', edgecolor='black')
    ax[0].set_title("HOG Histogram (Spatial Gradients)")
    ax[1].bar(range(len(hof_h)), hof_h, color='salmon', edgecolor='black')
    ax[1].set_title("HOF Histogram (Temporal Motion)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    KTH_DIR = r"C:\Users\aabab\kth\archive"
    TXT_PATH = r"C:\Users\aabab\kth\00sequences.txt"
    main(KTH_DIR, TXT_PATH)

    sample_video = r"C:\Users\aabab\kth\archive\running\person22_running_d4_uncomp.avi"
    print(f"Visualizing histograms for: {sample_video}")
    plot_histograms(sample_video)