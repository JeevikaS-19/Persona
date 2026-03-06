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
    Biometric Saccade Analysis v3.0 - Random Frame Sampling.
    Instead of a consecutive rolling window, we randomly sample frames
    scattered across the video to compare pupil displacements.
    Human eyes exhibit organic micro-jitter even between widely spaced frames,
    while deepfake pupils tend to sit locked or move too linearly.
    """
    mp_face_mesh = mp.solutions.face_mesh
    jitter_values = []
    all_frame_scores = []

    # Step 1: Sample random frames across the full video (not consecutive)
    n_frames = len(frames)
    sample_count = min(30, n_frames)  # Up to 30 randomly scattered frames
    sample_indices = sorted(_random.sample(range(n_frames), sample_count))

    pupil_coords = []  # List of (x, y) for each sampled frame that had a face

    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        for idx in sample_indices:
            frame = frames[idx]
            if frame is None: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pl = landmarks[PUPIL_LEFT]
                pr = landmarks[PUPIL_RIGHT]
                avg_x = (pl.x + pr.x) / 2
                avg_y = (pl.y + pr.y) / 2
                pupil_coords.append((avg_x, avg_y))

    if len(pupil_coords) < 5:
        return (0.5, {}) if return_signals else 0.5

    # Step 2: Compute pairwise deltas between random pairs of sampled positions
    # This measures whether the eye moved at all between distant points in time.
    # Human eyes have chaotic, non-linear trajectories. Deepfakes have linear or frozen paths.
    pts = np.array(pupil_coords)
    deltas = np.diff(pts, axis=0)  # Step vectors between consecutive sampled positions
    accel = np.diff(deltas, axis=0)  # Second derivative — acceleration (jitter)
    
    # Compute frame-level jitter as the magnitude of directional changes
    jitter_magnitudes = np.linalg.norm(accel, axis=1) * 100  # Scale to readable px units
    jitter_values = jitter_magnitudes.tolist()
    jitter_avg = float(np.mean(jitter_magnitudes)) if len(jitter_magnitudes) > 0 else 0.0
    
    # Step 3: Score — Human (high non-linear jitter) = 0.0 suspicion (low score = more fake)
    # Threshold: 0.25 is calibrated to typical human micro-saccade amplitudes
    suspicion_score = 1.0 - np.clip(jitter_avg / 0.25, 0.0, 1.0)
    final_score = float(np.clip(suspicion_score, 0.0, 1.0))

    tags = {
        "score": final_score,
        "jitter_avg": jitter_avg,
        "history": jitter_values,
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