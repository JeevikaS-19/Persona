import cv2
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque

# --- REFLECTION PRECISION SETTINGS ---
PUPIL_LEFT = 468 
PUPIL_RIGHT = 473

def get_eye_glint(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return None, False
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    min_val, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    mean_val = np.mean(blurred)
    std_val = np.std(blurred)
    # Adaptive highlight detection
    if max_val > mean_val + (1.5 * std_val):
        return max_loc, True
    return None, False

def get_pupil_metric(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return 0
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    c = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(c)
    if radius < 2 or radius > 40: return 0
    return float(radius)

def analyze(frames, fps=30.0, return_signals=False):
    """
    Reflection Forensic Specialist v2.0 - Headless.
    Analyzes corneal reflections (glints) and pupil size variance.
    """
    mp_face_mesh = mp.solutions.face_mesh
    pupil_history = deque(maxlen=30)
    scores = []
    glint_consistency = []
    
    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        for frame in frames:
            if frame is None: continue
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark
                
                def crop_eye(landmark_idx):
                    cx, cy = int(mesh[landmark_idx].x * w), int(mesh[landmark_idx].y * h)
                    if 40 < cx < w-40 and 40 < cy < h-40:
                        return frame[cy-40:cy+40, cx-40:cx+40]
                    return None

                l_eye = crop_eye(PUPIL_LEFT)
                r_eye = crop_eye(PUPIL_RIGHT)
                
                # Detect glints
                _, l_light = get_eye_glint(l_eye)
                _, r_light = get_eye_glint(r_eye)
                glint_consistency.append(1 if (l_light == r_light) else 0)
                
                # Pupil metrics
                l_p = get_pupil_metric(l_eye)
                r_p = get_pupil_metric(r_eye)
                avg_pupil = (l_p + r_p) / 2
                pupil_history.append(avg_pupil)
                
                # Scoring: Real eyes have variable pupil sizes and consistent glints
                # Deepfakes often have perfectly static pupils or mismatched glints
                score = 0.3 # Base (Human)
                if len(pupil_history) > 10:
                    var = np.var(pupil_history)
                    if var < 0.2: score += 0.4 # Static pupils = Deepfake threat
                if l_light != r_light: score += 0.2 # Mismatched light source reflection
                
                scores.append(np.clip(score, 0, 1))
            else:
                if scores: scores.append(scores[-1])

    if not scores:
        return (0.5, {}) if return_signals else 0.5
        
    final_score = float(np.mean(scores))
    tags = {
        "score": final_score,
        "glint_match": float(np.mean(glint_consistency)) if glint_consistency else 0.0,
        "pupil_var": float(np.var(list(pupil_history))) if len(pupil_history) > 1 else 0.0,
        "verdict": "DEEPFAKE" if final_score > 0.5 else "HUMAN"
    }
    
    return (final_score, tags) if return_signals else final_score

def run_webcam():
    cap = cv2.VideoCapture(0)
    print("\n--- Reflection Forensic Audit v2.0 ---")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, "REFLECTION ANALYSIS ACTIVE...", (20, 40), 1, 1, (255, 255, 0), 1)
        cv2.imshow("Reflection Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frames.append(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) >= 75:
        score, tags = analyze(frames, fps=fps, return_signals=True)
        print("-" * 30)
        print(f"RESULT: {tags['verdict']}")
        print(f"Suspicion Score: {score:.4f}")
        print(f"Glint Consistency: {tags['glint_match']:.2%}")
        print("-" * 30)

def run_file_upload():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(); root.destroy()
    
    if not path or not os.path.exists(path): return

    print(f"[*] Analyzing Reflections: {os.path.basename(path)}")
    cap = cv2.VideoCapture(path)
    frames = []; fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        print(f"Pupil Variance: {tags['pupil_var']:.6f}")
        print("-" * 30)

if __name__ == "__main__":
    print("Persona Reflection Specialist [v2.0]\n1. Upload | 2. Webcam")
    choice = input("Select: ").strip()
    if choice == '1': run_file_upload()
    elif choice == '2': run_webcam()
