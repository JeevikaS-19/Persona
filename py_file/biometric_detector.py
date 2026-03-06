import cv2
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

def analyze(frames, fps=30.0, return_signals=False):
    """
    Biometric Saccade Analysis v2.0 - Headless Forensic Specialist.
    Detects eye micro-dynamics (saccades) to distinguish between human 
    movement and robotic/frozen AI pupil overlays.
    """
    mp_face_mesh = mp.solutions.face_mesh
    raw_pupil_history = deque(maxlen=20)
    all_frame_scores = []
    jitter_values = []
    
    # Use refined landmarks for pupil tracking
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        for idx, frame in enumerate(frames):
            if frame is None: continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            jitter_score = 0
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                pl = landmarks[PUPIL_LEFT]
                pr = landmarks[PUPIL_RIGHT]
                avg_x = (pl.x + pr.x) / 2
                avg_y = (pl.y + pr.y) / 2
                
                raw_pupil_history.append((avg_x, avg_y))
                jitter_score = get_micro_jitter(list(raw_pupil_history))
                jitter_values.append(jitter_score)

                # Map jitter score to 0-1 suspicion scale
                # Human (high jitter) = 0.0 suspicion | Deepfake (low jitter) = 1.0 suspicion
                current_suspicion = 1 - (jitter_score / 0.25)
                current_suspicion = np.clip(current_suspicion, 0, 1)
                all_frame_scores.append(current_suspicion)
            else:
                # If no face is detected, we use neutral or previous state
                if all_frame_scores:
                    all_frame_scores.append(all_frame_scores[-1])

    if not all_frame_scores:
        return (0.5, {}) if return_signals else 0.5
        
    final_score = float(np.mean(all_frame_scores))
    tags = {
        "score": final_score,
        "jitter_avg": float(np.mean(jitter_values)) if jitter_values else 0.0,
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
        print(f"Suspicion Score: {score:.4f}")
        print(f"Avg Micro-Jitter: {tags['jitter_avg']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    print("Persona Biometric Specialist [v2.0]\n1. Upload | 2. Webcam")
    choice = input("Select: ").strip()
    if choice == '1':
        run_file_upload()
    elif choice == '2':
        run_webcam()