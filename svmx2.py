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
ACTIONS = ["walking", "jogging", "running", "boxing", "handclapping", "handwaving"]
LABEL2ID = {a: i for i, a in enumerate(ACTIONS)}

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
    """Collect all .avi files and parse them into (path, person_id, class_id)."""
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
            # Log parsing failures
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
    """
    Histogram of Optical Flow (HOF) on a local patch.
    flow_u, flow_v: optical flow components (H,W)
    (x,y): patch center
    returns: (bins,) L1-normalized orientation histogram weighted by magnitude
    """
    h, w = flow_u.shape
    r = patch // 2
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)

    u = flow_u[y0:y1, x0:x1]
    v = flow_v[y0:y1, x0:x1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)  # Range: [-pi, pi]
    ang = (ang + np.pi) * (bins / (2*np.pi))  # Map to [0, bins)

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
    """
    Histogram of Oriented Gradients (HOG) on a local patch.
    img_g: grayscale image (H,W), uint8
    (x,y): patch center
    returns: (bins,) L1-normalized gradient orientation histogram weighted by magnitude
    """
    h, w = img_g.shape
    r = patch // 2
    # Keep margins for gradient computation
    x0, x1 = max(1, x - r), min(w - 1, x + r)
    y0, y1 = max(1, y - r), min(h - 1, y + r)

    patch_img = img_g[y0:y1, x0:x1].astype(np.float32)

    # Simple gradient calculation
    gx = patch_img[:, 2:] - patch_img[:, :-2]
    gy = patch_img[2:, :] - patch_img[:-2, :]

    # Align shapes (crop to common region)
    ph = min(gx.shape[0], gy.shape[0])
    pw = min(gx.shape[1], gy.shape[1])
    gx = gx[:ph, :pw]
    gy = gy[:ph, :pw]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)  # Range: [-pi, pi]
    # Use unsigned orientations: [0, pi)
    ang = np.mod(ang, np.pi)
    ang = ang * (bins / np.pi)  # Map to [0, bins)

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
    return str(Path(cache_dir) / f"{base}_{key}_v2.npy")

def extract_descriptors(video_path: str,
                        resize=(160,120),
                        grid=8,
                        patch=16,
                        hog_bins=9,
                        hof_bins=8,
                        max_desc=1200,
                        frame_step=1,
                        cache_dir=None):
    """
    Extract dense local descriptors (HOG + HOF) from a video.
    - Uses dense grid sampling per frame.
    - Optional caching saves/loads descriptors as .npy files.
    - Subsamples to max_desc per video for computational efficiency.
    Returns: (N, D) array where D = hog_bins + hof_bins.
    """
    params = {
        "resize": resize,
        "grid": grid,
        "patch": patch,
        "hog_bins": hog_bins,
        "hof_bins": hof_bins,
        "max_desc": max_desc,
        "frame_step": frame_step,
    }

    # Load from cache if available
    if cache_dir is not None:
        cpath = _cache_path(cache_dir, video_path, params)
        if os.path.exists(cpath):
            return np.load(cpath)

    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        X0 = np.zeros((0, hog_bins + hof_bins), dtype=np.float32)
        if cache_dir is not None:
            np.save(cpath, X0)
        return X0

    prev = cv2.resize(prev, resize)
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    descs = []
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fidx += 1
        if frame_step > 1 and (fidx % frame_step != 0):
            continue

        frame = cv2.resize(frame, resize)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Farneback Optical Flow between frames for HOF
        flow = cv2.calcOpticalFlowFarneback(
            prev_g, g, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_u = flow[..., 0]
        flow_v = flow[..., 1]

        h, w = g.shape

        # Sample features on a dense grid (skipping borders)
        for yy in range(patch//2, h - patch//2, grid):
            for xx in range(patch//2, w - patch//2, grid):
                hog = hog_patch(g, xx, yy, patch=patch, bins=hog_bins)
                hof = hof_patch(flow_u, flow_v, xx, yy, patch=patch, bins=hof_bins)
                descs.append(np.concatenate([hog, hof], axis=0))

        prev_g = g

        # Threshold to limit total descriptors for runtime speed
        if len(descs) > max_desc * 3:
            break

    cap.release()

    if len(descs) == 0:
        X0 = np.zeros((0, hog_bins + hof_bins), dtype=np.float32)
        if cache_dir is not None:
            np.save(cpath, X0)
        return X0

    X = np.vstack(descs).astype(np.float32)

    # Randomly subsample to exactly max_desc descriptors
    if X.shape[0] > max_desc:
        idx = np.random.choice(X.shape[0], size=max_desc, replace=False)
        X = X[idx]

    # Save to cache
    if cache_dir is not None:
        np.save(cpath, X)

    return X

# -----------------------
# 3) BoVW + classifier
# -----------------------
def build_codebook(train_desc_list, k=400, seed=0):
    """Fit k-means codebook (visual vocabulary) on training descriptors."""
    X = np.vstack([d for d in train_desc_list if d.shape[0] > 0])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10000)
    km.fit(X)
    return km

def bovw_hist(desc, codebook):
    """Convert descriptors from one video into a normalized BoVW histogram."""
    k = codebook.n_clusters
    if desc.shape[0] == 0:
        return np.zeros((k,), dtype=np.float32)
    words = codebook.predict(desc)
    hist = np.bincount(words, minlength=k).astype(np.float32)

    s = hist.sum()
    if s > 1e-8:
        hist /= s  # Apply L1 normalization
    return hist

def main(KTH_DIR):
    items = list_videos(KTH_DIR)
    print("Items parsed:", len(items))
    print("First item:", items[0])

    if len(items) == 0:
        raise RuntimeError("No videos found. Check KTH_DIR path and filename patterns.")

    # Subject-based split: Training (Subject 1-16), Testing (Subject 17-25)
    train = [(v,p,y) for (v,p,y) in items if p <= 16]
    test  = [(v,p,y) for (v,p,y) in items if p >= 17]

    # 1) Extract training descriptors
    print("Extracting training descriptors...")
    train_desc = []
    y_train = []
    for v,p,y in train:
        # Using increased bin counts: HOG 18, HOF 16
        d = extract_descriptors(v, cache_dir=r"C:\Users\aabab\kth\cache_hoghof", 
                                frame_step=1, hog_bins=18, hof_bins=16)
        train_desc.append(d)
        y_train.append(y)

    # 2) Build visual vocabulary (codebook)
    print("Building codebook...")
    codebook = build_codebook(train_desc, k=1200, seed=0)

    # 3) Convert training videos to histograms
    print("Building training histograms...")
    X_train = np.vstack([bovw_hist(d, codebook) for d in train_desc])
    y_train = np.array(y_train, dtype=np.int32)

    # 4) Extract testing histograms
    print("Extracting testing descriptors + histograms...")
    X_test = []
    y_test = []
    for v,p,y in test:
        d = extract_descriptors(v, cache_dir=r"C:\Users\aabab\kth\cache_hoghof", 
                                frame_step=1, hog_bins=18, hof_bins=16)
        X_test.append(bovw_hist(d, codebook))
        y_test.append(y)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test, dtype=np.int32)

    # -----------------------
    # 5) Classifier Experiments (Baseline SVM, RF, MLP, XGB)
    # -----------------------
    print("\n--- Starting Classifier Experiments ---")
    
    models = {
        # Baseline: Chi-squared approximation + Linear SVM
        "Baseline (Chi2 + SVM)": make_pipeline(
            AdditiveChi2Sampler(sample_steps=2),
            LinearSVC(C=4.0, max_iter=20000, random_state=0)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
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

    # Final result summary
    print("\n" + "="*30)
    print("      EXPERIMENT SUMMARY")
    print("="*30)
    for name, acc in results.items():
        print(f"{name:25s}: {acc:.4f}")
    print("="*30)
    
    print(f"Train videos: {len(train)}")
    print(f"Test videos: {len(test)}")

import matplotlib.pyplot as plt

def plot_histograms(video_path):
    """Visualize HOG and HOF histograms for a single frame pair."""
    # 1. Extract a single frame and a flow field for demonstration
    cap = cv2.VideoCapture(video_path)
    ok, frame1 = cap.read()
    ok, frame2 = cap.read()
    cap.release()

    if not ok:
        print("Could not read video.")
        return

    # Pre-process
    img1 = cv2.cvtColor(cv2.resize(frame1, (160, 120)), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.resize(frame2, (160, 120)), cv2.COLOR_BGR2GRAY)
    
    # Compute flow for HOF
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 2. Pick a center point (e.g., center of the image)
    cx, cy = 80, 60 
    
    # 3. Get the histograms using your existing functions
    hog_h = hog_patch(img2, cx, cy, patch=16, bins=18)
    hof_h = hof_patch(flow[..., 0], flow[..., 1], cx, cy, patch=16, bins=16)

    # 4. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # HOG Plot
    ax[0].bar(range(len(hog_h)), hog_h, color='skyblue', edgecolor='black')
    ax[0].set_title("HOG Histogram (Spatial Gradients - 18 bins)")
    ax[0].set_xlabel("Orientation Bin")
    ax[0].set_ylabel("Normalized Magnitude")

    # HOF Plot
    ax[1].bar(range(len(hof_h)), hof_h, color='salmon', edgecolor='black')
    ax[1].set_title("HOF Histogram (Temporal Motion - 16 bins)")
    ax[1].set_xlabel("Direction Bin")
    ax[1].set_ylabel("Normalized Magnitude")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update this path to your local KTH dataset directory
    KTH_DIR = r"C:\Users\aabab\kth\archive"
    main(KTH_DIR)

    # Visualize a sample video
    sample_video = r"C:\Users\aabab\kth\archive\running\person22_running_d4_uncomp.avi"
    print(f"Visualizing histograms for: {sample_video}")
    plot_histograms(sample_video)