import cv2
import numpy as np
import mediapipe_compat  # noqa
import mediapipe as mp
import os
import time
from collections import deque
from scipy.optimize import curve_fit
import math

# --- OPTICAL PHYSICS SETTINGS ---
# Eyelid Landmarks for Eye Aspect Ratio (EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Iris Landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio to detect blinks."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    ear = (v1 + v2) / (2.0 * h_dist + 1e-6)
    return ear

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for glint morphology testing."""
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def extract_glint_physics(eye_crop, is_blinking):
    """
    Analyzes the crop for Morphology (Gaussian Fall-off) and Intensity.
    Returns: (Centroid X, Centroid Y, Intensity, Morphology_Score)
    """
    if eye_crop is None or eye_crop.size == 0:
        return None, None, 0, 0.0

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    
    # Simple Blink Occlusion Check (Intensity Test)
    max_intensity = np.max(gray)
    if is_blinking and max_intensity > 150:
        # Critical Fail: Reflection persists during a full blink (Baked Texture)
        return None, None, max_intensity, -1.0 # -1 signals hard fail
        
    if is_blinking:
        return None, None, 0, 1.0 # Normal behavior

    # Locate peak intensity (glint)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(gray, (5,5), 0))
    mean_val, std_val = np.mean(gray), np.std(gray)
    
    # If no distinct light source, return inconclusive
    if max_val < mean_val + (1.5 * std_val):
        return None, None, max_val, 0.5

    gx, gy = max_loc
    
    # Morphology Test (Gaussian Fall-off)
    # Extract small patch around glint to fit 2D Gaussian
    patch_size = 10
    half = patch_size // 2
    px1, py1 = max(0, gx - half), max(0, gy - half)
    px2, py2 = min(gray.shape[1], gx + half), min(gray.shape[0], gy + half)
    patch = gray[py1:py2, px1:px2]
    
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        return gx, gy, max_val, 0.5
        
    # Fit Gaussian
    x = np.linspace(0, patch.shape[1]-1, patch.shape[1])
    y = np.linspace(0, patch.shape[0]-1, patch.shape[0])
    x, y = np.meshgrid(x, y)
    
    initial_guess = (max_val, half, half, 2.0, 2.0, 0.0, np.mean(patch))
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), patch.ravel(), p0=initial_guess, maxfev=400)
        # Calculate residual error (R^2)
        fit = gaussian_2d((x, y), *popt).reshape(patch.shape)
        ss_res = np.sum((patch - fit) ** 2)
        ss_tot = np.sum((patch - np.mean(patch)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-6))
        
        morphology_score = np.clip(r2, 0.0, 1.0)
        
        # Super-sharp, flat white blobs (synthetic) will have terrible Gaussian fits (Low Morphology)
        # Soft, realistic physics-based light will fit well (High Morphology)
        
    except Exception:
        morphology_score = 0.0 # Synthetic blob failed curve fit

    return gx, gy, max_val, morphology_score

def analyze(frames, fps=30.0, return_signals=False):
    """
    Reflection Forensic Specialist v3.0 - The Humanity Checklist.
    1. Geometric Consistency (Parallax)
    2. Temporal Dynamics (Blink Occlusion, Micro-Jitter)
    3. Morphology (Gaussian Fall-off)
    Returns: Humanity Score (1.0 = Organic Physical Perfection, 0.0 = Synthetic)
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    history_l_glint = deque(maxlen=10)
    history_r_glint = deque(maxlen=10)
    history_head_pose = deque(maxlen=10)
    
    frame_scores = []
    
    # Physics Tracking
    parallax_failures = 0
    baked_blink_failures = 0
    morphology_scores = []
    micro_jitters = []

    with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
        for idx, frame in enumerate(frames):
            if frame is None: continue
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark
                
                # 1. Temporal Dynamics: Blink Occlusion Check
                ear_l = get_ear(mesh, LEFT_EYE, w, h)
                ear_r = get_ear(mesh, RIGHT_EYE, w, h)
                is_blinking = (ear_l < 0.20) or (ear_r < 0.20)
                
                # Head Pose (Proxy for rotation/texture baking check)
                nose = mesh[1]
                history_head_pose.append((nose.x, nose.y))

                def crop_iris(eye_indices):
                    # Get bounding box of Iris
                    pts = np.array([(int(mesh[i].x * w), int(mesh[i].y * h)) for i in eye_indices])
                    x_min, y_min = np.min(pts, axis=0) - 10
                    x_max, y_max = np.max(pts, axis=0) + 10
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    if x_max <= x_min or y_max <= y_min: return None, 0, 0
                    return frame[y_min:y_max, x_min:x_max], x_min, y_min

                l_crop, lx, ly = crop_iris(LEFT_IRIS)
                r_crop, rx, ry = crop_iris(RIGHT_IRIS)
                
                # Extract Physics
                lgx, lgy, l_int, l_morph = extract_glint_physics(l_crop, is_blinking)
                rgx, rgy, r_int, r_morph = extract_glint_physics(r_crop, is_blinking)
                
                # Process Rules & Failures
                frame_humanity = 0.5 # Default ambiguous
                
                if l_morph == -1.0 or r_morph == -1.0:
                    baked_blink_failures += 1
                    frame_humanity = 0.0 # CRITICAL FAIL: Persistence during blink
                elif (lgx is not None) and (rgx is not None):
                    # Accumulate valid morphology for average
                    morphology_scores.append((l_morph + r_morph) / 2.0)
                    
                    # Store absolute centroid coordinates
                    history_l_glint.append((lx + lgx, ly + lgy))
                    history_r_glint.append((rx + rgx, ry + rgy))
                    
                    # 2. Geometric Consistency: The Parallax Test
                    # Glints in left and right eye shouldn't have the exact same relative offset
                    # due to 3D spherical cornea vs flat planar projection (deepfake)
                    l_rel_x, l_rel_y = lgx, lgy # Relative to iris bounding box
                    r_rel_x, r_rel_y = rgx, rgy
                    
                    # If coordinates are identical (Zero Parallax), it's a baked 2D texture mask.
                    if abs(l_rel_x - r_rel_x) <= 1 and abs(l_rel_y - r_rel_y) <= 1:
                        parallax_failures += 1
                        frame_humanity = 0.0 # CRITICAL FAIL: Zero Parallax
                    else:
                        frame_humanity = min(1.0, frame_humanity + 0.3) # Positive Parallax Proof
                        
                frame_scores.append(frame_humanity)
            else:
                if frame_scores: frame_scores.append(frame_scores[-1])

    # Temporal Dynamics: Micro-Jitter (Floating Check)
    # Check if reflection stays glued to iris despite head movement (Texture Baking)
    jitter_score = 0.5
    if len(history_l_glint) > 5 and len(history_head_pose) > 5:
        # Calculate glint variance vs head variance
        glint_var = np.var(history_l_glint, axis=0).sum()
        head_var = np.var(history_head_pose, axis=0).sum() * (h*w) # Scale head to pixel space roughly
        
        # Real glints slide on the curvature independently of head pan
        if glint_var < 0.5 and head_var > 10.0:
            # Glint is frozen while head moves = Baked Texture (Score 0)
            jitter_score = 0.0 
        elif glint_var > 1.0:
            # Evidence of organic sub-pixel tremor
            jitter_score = 1.0 
            
    # Aggregate Final "Humanity" Score
    if not frame_scores:
        return (0.5, {}) if return_signals else 0.5
        
    # Start with base frame average
    humanity = float(np.mean(frame_scores))
    
    # Apply Pillar Multipliers
    morph_avg = float(np.mean(morphology_scores)) if morphology_scores else 0.5
    
    # 1. Intensity/Morphology
    humanity = (humanity * 0.5) + (morph_avg * 0.5)
    
    # 2. Temporal Dynamics (Jitter)
    humanity = (humanity * 0.7) + (jitter_score * 0.3)
    
    # Critical System Failures (Hard zeros)
    if parallax_failures > (len(frames) * 0.15): # 15% of frames have zero parallax
        humanity = 0.0
    if baked_blink_failures > 2: # Even 2 frames of light escaping a closed eyelid is physically impossible
        humanity = 0.0

    final_score = float(np.clip(1.0 - humanity, 0.0, 1.0))  # 0=human, 1=deepfake
    tags = {
        "score": final_score,
        "morphology_r2": morph_avg,
        "parallax_fails": parallax_failures,
        "blink_persistence": baked_blink_failures,
        "verdict": "DEEPFAKE" if final_score >= 0.5 else "HUMAN"
    }
    
    return (final_score, tags) if return_signals else final_score

import matplotlib.pyplot as plt

def plot_report(tags, score):
    """Display a graphical forensic report for the Optical Physics Engine."""
    try:
        label = 'DEEPFAKE' if score >= 0.5 else 'HUMAN'  # 0=human, 1=deepfake
        color = 'red' if label == 'DEEPFAKE' else 'green'
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#1c1c1c')
        for ax in axes:
            ax.set_facecolor('#2a2a2a')
        
        # Left plot: Pillar Scores (show authenticity for readability, invert for display)
        pillars = ['Morphology\n(Gaussian R²)', 'Parallax\nProof', 'Blink\nIntegrity']
        parallax_score = 1.0 if tags.get('parallax_fails', 0) == 0 else max(0, 1 - tags['parallax_fails'] / 10)
        blink_score = 1.0 if tags.get('blink_persistence', 0) == 0 else 0.0
        values = [tags.get('morphology_r2', 0), parallax_score, blink_score]
        bar_colors = ['green' if v > 0.5 else 'red' for v in values]
        bars = axes[0].bar(pillars, values, color=bar_colors, edgecolor='white', linewidth=0.5)
        axes[0].set_ylim(0, 1.2)
        axes[0].set_ylabel('Physics Score (1=pass)', color='white')
        axes[0].set_title('Physics Pillar Breakdown', color='white', fontsize=12, fontweight='bold')
        axes[0].tick_params(colors='white')
        axes[0].axhline(y=0.5, color='yellow', linestyle='--', linewidth=1, label='Pass Threshold')
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=10)
        axes[0].legend(facecolor='#333')
        
        # Right plot: Suspicion gauge (score=0 human, score=1 deepfake)
        gauge_data = [1 - score, score]  # human portion first
        wedge_colors = ['#2ecc71', color]
        axes[1].pie(gauge_data, colors=wedge_colors, startangle=90,
                    wedgeprops={'width': 0.4, 'edgecolor': '#1c1c1c', 'linewidth': 2})
        axes[1].text(0, 0, f'{score:.2%}', ha='center', va='center',
                     fontsize=22, fontweight='bold', color=color)
        axes[1].set_title('Suspicion Score (0=Human, 1=Fake)', color='white', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Optical Physics Forensic [v3.0]  —  Verdict: {label}',
                     fontsize=14, fontweight='bold', color=color, y=1.01)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'Plotting failed: {e}')

def run_webcam():
    cap = cv2.VideoCapture(0)
    print("\n--- Optical Physics Audit v3.0 ---")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, "PHYSICS ENGINE ACTIVE...", (20, 40), 1, 1, (255, 255, 0), 1)
        cv2.imshow("Optical Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frames.append(frame)
        
    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) >= 75:
        score, tags = analyze(frames, fps=fps, return_signals=True)
        print("-" * 30)
        print(f"RESULT: {tags['verdict']} (Score: {score:.4f})")
        print(f"Morphology (Gaussian R2): {tags['morphology_r2']:.2%}")
        print(f"Zero Parallax Frames: {tags['parallax_fails']}")
        print(f"Blink Physics Fails: {tags['blink_persistence']}")
        print("-" * 30)
        plot_report(tags, score)

def run_file_upload():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(); root.destroy()
    
    if not path or not os.path.exists(path): return

    print(f"[*] Analyzing Physics: {os.path.basename(path)}")
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
        print(f"RESULT: {tags['verdict']} (Score: {score:.4f})")
        print(f"Morphology (Gaussian R2): {tags['morphology_r2']:.2%}")
        print(f"Zero Parallax Frames: {tags['parallax_fails']}")
        print(f"Blink Physics Fails: {tags['blink_persistence']}")
        print("-" * 30)
        plot_report(tags, score)

if __name__ == "__main__":
    print("Persona Optical Physics Engine [v3.0]\n1. Upload | 2. Webcam")
    choice = input("Select: ").strip()
    if choice == '1': run_file_upload()
    elif choice == '2': run_webcam()
