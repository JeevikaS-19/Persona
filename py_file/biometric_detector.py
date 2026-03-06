import cv2
import mediapipe_compat  # noqa
import mediapipe as mp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque

# --- BIOMETRIC PRECISION SETTINGS ---
PUPIL_LEFT = 468 
PUPIL_RIGHT = 473
JITTER_SENSITIVITY = 0.12 

def get_micro_jitter(history):
    if len(history) < 5: return 0
    pts = np.array(history)
    diff = np.diff(pts, axis=0)
    accel = np.diff(diff, axis=0)
    return np.mean(np.abs(accel)) * 100 

import random as _random

def analyze(frames, fps=30.0, return_signals=False):
    """
    Biometric Saccade Analysis v4.0 - Physics-Calibrated.

    Three pillars:
      1. IID Normalization: jitter expressed as % of inter-ocular distance —
         makes the threshold resolution/zoom independent.
      2. L/R Cross-Correlation: tracks left and right eyes *separately*.
         Real eyes move in sync (high correlation). Deepfake eye overlays
         sometimes drift independently (low correlation). This is a unique tell.
      3. Coefficient of Variation (CV): measures jitter *burstiness*.
         Human saccades are punctate snaps (HIGH CV = high variance / high mean).
         Deepfake pupils drift smoothly (LOW CV = consistent linear motion).
    """
    mp_face_mesh = mp.solutions.face_mesh

    n_frames = len(frames)
    sample_count = min(30, n_frames)
    sample_indices = sorted(_random.sample(range(n_frames), sample_count))

    # Track left and right pupils independently
    left_coords  = []  # (x, y) per sampled frame
    right_coords = []
    iid_values   = []  # Inter-Ocular Distance per frame (for normalization)

    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        for idx in sample_indices:
            frame = frames[idx]
            if frame is None: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pl = lm[PUPIL_LEFT]
                pr = lm[PUPIL_RIGHT]
                left_coords.append((pl.x, pl.y))
                right_coords.append((pr.x, pr.y))
                iid = np.sqrt((pl.x - pr.x)**2 + (pl.y - pr.y)**2)
                iid_values.append(iid)

    if len(left_coords) < 5:
        return (0.5, {}) if return_signals else 0.5

    iid_mean = float(np.mean(iid_values)) if iid_values else 0.065  # ~6.5% of frame width default

    # ── PILLAR 1: IID-Normalised Jitter (acceleration magnitude) ──────────────
    def iid_jitter(coords):
        pts = np.array(coords)
        deltas = np.diff(pts, axis=0)
        accel  = np.diff(deltas, axis=0)
        mags   = np.linalg.norm(accel, axis=1)
        return mags / (iid_mean + 1e-6)  # express as fraction of IID

    l_jitter = iid_jitter(left_coords)
    r_jitter = iid_jitter(right_coords)

    jitter_combined = np.concatenate([l_jitter, r_jitter])
    jitter_avg      = float(np.mean(jitter_combined))

    # ── PILLAR 2: L/R Cross-Correlation ───────────────────────────────────────
    # Align lengths (both derived from same sample set minus 2 points for 2nd diff)
    min_len = min(len(l_jitter), len(r_jitter))
    if min_len >= 3:
        lr_corr = float(np.corrcoef(l_jitter[:min_len], r_jitter[:min_len])[0, 1])
        if np.isnan(lr_corr): lr_corr = 0.0
    else:
        lr_corr = 0.0

    # High corr (> 0.5) = eyes move together = real. Low/negative corr = decoupled = deepfake.
    corr_score = np.clip((lr_corr + 1.0) / 2.0, 0.0, 1.0)  # Map [-1,1] → [0,1]

    # ── PILLAR 3: Coefficient of Variation (Burstiness) ───────────────────────
    jitter_std = float(np.std(jitter_combined))
    cv = jitter_std / (jitter_avg + 1e-6)
    # Real: high CV (bursty snap-and-fixate). Deepfake: low CV (smooth drift).
    # Threshold: CV > 0.8 indicates organic saccade pattern.
    cv_score = np.clip(cv / 0.8, 0.0, 1.0)

    # ── AGGREGATE ─────────────────────────────────────────────────────────────
    # Jitter magnitude (IID-normalised): calibrated threshold is 0.05 IID units
    mag_score = np.clip(jitter_avg / 0.05, 0.0, 1.0)

    # Weighted ensemble of the 3 pillars (all pointing: 1.0 = human)
    humanity = (mag_score * 0.40) + (corr_score * 0.35) + (cv_score * 0.25)
    # Convert to suspicion then invert back to authenticity (main.py inverts again)
    suspicion_score = 1.0 - float(np.clip(humanity, 0.0, 1.0))
    final_score = float(np.clip(suspicion_score, 0.0, 1.0))

    tags = {
        "score": final_score,
        "jitter_avg": jitter_avg,
        "lr_correlation": lr_corr,
        "cv_burstiness": cv,
        "history": jitter_combined.tolist(),
        "verdict": "DEEPFAKE" if final_score > 0.5 else "HUMAN"
    }
    return (final_score, tags) if return_signals else final_score


def plot_report(tags, score):
    """Generates a forensic biometric jitter report."""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(tags.get('history', []), color='orange', linewidth=2, label='Micro-Jitter')
        plt.axhline(y=0.12, color='red', linestyle='--', label='Sensitivity Threshold')
        plt.title(f"Biometric Eye Jitter v2.0 | Avg: {tags.get('jitter_avg', 0):.4f}")
        plt.xlabel("Frame Index")
        plt.ylabel("Jitter Intensity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        label = 'DEEPFAKE' if score > 0.5 else 'HUMAN'
        plt.suptitle(f"Final Score: {score:.4f} - {label}", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

def run_webcam():
    cap = cv2.VideoCapture(0)
    print("\n--- Biometric Live Audit v2.0 ---")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.putText(frame, "EYE TRACKING ACTIVE...", (20, 40), 1, 1, (0, 255, 0), 1)
        cv2.imshow("Biometric Saccade Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frames.append(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) >= 75:
        print(f"[*] Analyzing {len(frames)} frames...")
        score, tags = analyze(frames, fps=fps, return_signals=True)
        print("-" * 30)
        print(f"RESULT: {tags['verdict']}")
        print(f"Suspicion Score: {score:.4f}")
        print(f"Avg Micro-Jitter: {tags['jitter_avg']:.4f}")
        print("-" * 30)

def run_file_upload():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename()
    root.destroy()
    
    if not path or not os.path.exists(path):
        print("Invalid path.")
        return

    print(f"[*] Analyzing: {os.path.basename(path)}")
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        if len(frames) >= 600: break
    cap.release()
    
    if len(frames) >= 75:
        score, tags = analyze(frames, fps=fps, return_signals=True)
        print("-" * 30)
        print(f"RESULT: {tags['verdict']}")
        print(f"Humanity Score: {score:.4f}")
        print(f"Avg Micro-Jitter: {tags['jitter_avg']:.4f}")
        print("-" * 30)
        plot_report(tags, score)

if __name__ == "__main__":
    print("Persona Biometric Specialist [v2.0]\n1. Upload | 2. Webcam")
    choice = input("Select: ").strip()
    if choice == '1':
        run_file_upload()
    elif choice == '2':
        run_webcam()