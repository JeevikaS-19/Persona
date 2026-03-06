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

def analyze(frames, fps=30.0, return_signals=False, precomputed_landmarks=None):
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

    def get_meshes():
        if precomputed_landmarks is not None:
            class _DL: __slots__=['x','y','z']
            class _DM:
                def __init__(self, a): self.a = a
                def __getitem__(self, j):
                    d = _DL(); d.x, d.y, d.z = self.a[j,0], self.a[j,1], self.a[j,2]; return d
            for i, f in enumerate(frames):
                if i % 2 != 0 or f is None: continue
                yield i, f, _DM(precomputed_landmarks[i]) if (i < len(precomputed_landmarks) and precomputed_landmarks[i] is not None) else None
        else:
            with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
                for i, f in enumerate(frames):
                    if i % 2 != 0 or f is None: continue
                    res = face_mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    yield i, f, res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None

    if True:
        for idx, frame, mesh in get_meshes():
            h, w, _ = frame.shape
            
            if mesh is not None:
                
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
    """Premium 3-panel forensic report for the Optical Physics Engine."""
    try:
        DARK_BG   = '#0f0f14'
        PANEL_BG  = '#16161f'
        BORDER    = '#2a2a3a'
        TEXT_DIM  = '#8888aa'
        TEXT_MAIN = '#e0e0f0'

        label   = 'DEEPFAKE' if score >= 0.5 else 'HUMAN'
        v_color = '#e74c3c' if label == 'DEEPFAKE' else '#2ecc71'

        # Compute pillar PASS scores (higher = passed the physics check)
        parallax_pass = 1.0 if tags.get('parallax_fails', 0) == 0 \
                        else max(0.0, 1.0 - tags['parallax_fails'] / 10.0)
        blink_pass    = 1.0 if tags.get('blink_persistence', 0) == 0 else 0.0
        morph_pass    = float(tags.get('morphology_r2', 0.0))

        pillar_names   = ['Morphology\n(Gaussian R²)', 'Parallax\nConsistency', 'Blink\nPhysics']
        pillar_vals    = [morph_pass, parallax_pass, blink_pass]
        pillar_colors  = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in pillar_vals]

        fig = plt.figure(figsize=(14, 5), facecolor=DARK_BG)
        fig.patch.set_facecolor(DARK_BG)

        # ── Grid: Left (bar chart) | Centre (ring) | Right (stats table)
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.4, 1.2],
                              left=0.06, right=0.97, top=0.83, bottom=0.13, wspace=0.35)
        ax_bars  = fig.add_subplot(gs[0])
        ax_ring  = fig.add_subplot(gs[1])
        ax_stats = fig.add_subplot(gs[2])

        # ── Header banner ──────────────────────────────────────────────────
        fig.text(0.5, 0.95, 'PERSONA · OPTICAL PHYSICS AUDIT',
                 ha='center', va='top', fontsize=11, color=TEXT_DIM,
                 fontfamily='monospace', fontweight='bold', letterSpacing=4)  # type: ignore
        fig.text(0.5, 0.90, f'Verdict  ·  {label}',
                 ha='center', va='top', fontsize=18, color=v_color, fontweight='bold')

        # ── Panel 1: Horizontal bar chart of pillars ───────────────────────
        ax_bars.set_facecolor(PANEL_BG)
        ax_bars.spines[:].set_color(BORDER)
        ax_bars.tick_params(colors=TEXT_DIM, labelsize=9)
        ax_bars.xaxis.label.set_color(TEXT_DIM)

        y_pos = range(len(pillar_names))
        bars = ax_bars.barh(list(y_pos), pillar_vals, color=pillar_colors,
                            height=0.5, edgecolor=DARK_BG, linewidth=1.5)
        ax_bars.set_xlim(0, 1.15)
        ax_bars.set_yticks(list(y_pos))
        ax_bars.set_yticklabels(pillar_names, color=TEXT_MAIN, fontsize=9.5,
                                fontfamily='monospace')
        ax_bars.axvline(x=0.5, color='#f39c12', linewidth=1, linestyle='--', alpha=0.7)
        ax_bars.set_xlabel('Pass Score  (>0.5 = physics intact)', color=TEXT_DIM, fontsize=8.5)
        ax_bars.set_title('Physics Pillars', color=TEXT_MAIN, fontsize=10,
                           fontweight='bold', pad=8)
        ax_bars.set_facecolor(PANEL_BG)
        ax_bars.grid(axis='x', color=BORDER, linewidth=0.5, alpha=0.5)
        ax_bars.set_axisbelow(True)

        for bar, val, pc in zip(bars, pillar_vals, pillar_colors):
            ax_bars.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f'{val:.2f}', va='center', color=pc,
                         fontsize=10, fontweight='bold', fontfamily='monospace')

        # ── Panel 2: Ring gauge ────────────────────────────────────────────
        ax_ring.set_facecolor(PANEL_BG)
        ax_ring.set_aspect('equal')
        ax_ring.axis('off')
        ax_ring.set_title('Suspicion', color=TEXT_MAIN, fontsize=10,
                            fontweight='bold', pad=8)

        ring_data   = [score, 1.0 - score]
        ring_colors = [v_color, '#22222f']
        wedges, _   = ax_ring.pie(ring_data, colors=ring_colors, startangle=90,
                                   wedgeprops={'width': 0.38, 'edgecolor': DARK_BG,
                                               'linewidth': 2.5})
        ax_ring.text(0, 0.07, f'{score:.0%}', ha='center', va='center',
                     fontsize=24, fontweight='bold', color=v_color, fontfamily='monospace')
        ax_ring.text(0, -0.22, '0 = human   1 = fake',
                     ha='center', va='center', fontsize=7, color=TEXT_DIM,
                     fontfamily='monospace')

        # ── Panel 3: Stats panel ───────────────────────────────────────────
        ax_stats.set_facecolor(PANEL_BG)
        ax_stats.axis('off')
        ax_stats.set_title('Signal Details', color=TEXT_MAIN, fontsize=10,
                             fontweight='bold', pad=8)

        stats = [
            ('Morphology R²', f"{morph_pass:.1%}"),
            ('Parallax Fails', str(tags.get('parallax_fails', 0))),
            ('Blink Fails',    str(tags.get('blink_persistence', 0))),
        ]
        for i, (k, v) in enumerate(stats):
            y = 0.82 - i * 0.28
            ax_stats.text(0.05, y,       k, transform=ax_stats.transAxes,
                          color=TEXT_DIM, fontsize=9, fontfamily='monospace')
            ax_stats.text(0.05, y - 0.10, v, transform=ax_stats.transAxes,
                          color=TEXT_MAIN, fontsize=14, fontweight='bold',
                          fontfamily='monospace')

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
