import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def analyze(frames, return_signals=False):
    """
    Forensic rPPG Analysis refined for High-Motion and Cam Capture.
    - POS Method for robust pulse extraction.
    - Broad Drift Threshold (up to 15.0) for natural movement.
    - G/R Ratio (>1.15) as the primary biometric fingerprint.
    """
    if not frames or len(frames) < 150:
        return (0.5, None) if return_signals else 0.5

    mp_face_mesh = mp.solutions.face_mesh
    CHEEK_R = [117, 118, 119, 120, 121, 101]
    CHEEK_L = [346, 347, 348, 349, 350, 330]
    FOREHEAD = [10, 109, 67, 103, 285, 297, 338]
    ROI_INDICES = CHEEK_R + CHEEK_L + FOREHEAD
    
    rgb_means = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for frame in frames:
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                rgb_means.append([np.nan, np.nan, np.nan])
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in ROI_INDICES])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, points, 255)
            mean_vals = cv2.mean(frame, mask=mask)[:3] 
            rgb_means.append([mean_vals[2], mean_vals[1], mean_vals[0]]) # R, G, B

    rgb_array = np.array(rgb_means)
    for i in range(3):
        series = rgb_array[:, i]
        mask_nan = np.isnan(series)
        if mask_nan.any():
            if not mask_nan.all():
                series[mask_nan] = np.interp(np.flatnonzero(mask_nan), np.flatnonzero(~mask_nan), series[~mask_nan])
            else: return (0.5, None) if return_signals else 0.5
        rgb_array[:, i] = series

    R, G, B = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]
    fs = 30.0
    
    # --- POS Pulse Extraction ---
    R_norm = R / np.mean(R)
    G_norm = G / np.mean(G)
    B_norm = B / np.mean(B)
    pos_signal = (1.0 - (R_norm / 2.0)) + G_norm - (B_norm / 2.0)
    
    b, a = butter(4, [0.75 / (0.5 * fs), 3.0 / (0.5 * fs)], btype='band')
    bvp_signal = filtfilt(b, a, pos_signal)

    # --- Windowed Drift Calculation ---
    win_len = 150 
    stride = 30 
    bpms = []
    for i in range(0, len(bvp_signal) - win_len, stride):
        win = bvp_signal[i:i+win_len]
        yf = np.abs(rfft(win))
        xf = rfftfreq(win_len, 1/fs)
        band_mask = (xf >= 0.75) & (xf <= 2.5)
        peak_bpm = xf[band_mask][np.argmax(yf[band_mask])] * 60
        bpms.append(peak_bpm)
    bpm_drift = np.std(bpms) if len(bpms) > 1 else 0.0

    # --- Multi-Spectral Consistency ---
    Rf = filtfilt(b, a, R_norm)
    Gf = filtfilt(b, a, G_norm)
    r_peak = np.max(np.abs(rfft(Rf)))
    g_peak = np.max(np.abs(rfft(Gf)))
    gr_ratio = g_peak / r_peak if r_peak > 0 else 1.0

    # --- Global FFT & SNR ---
    n = len(bvp_signal)
    yf = rfft(bvp_signal)
    xf = rfftfreq(n, 1/fs)
    mags = np.abs(yf)
    band_mask = (xf >= 0.75) & (xf <= 2.5)
    peak_bpm = xf[band_mask][np.argmax(mags[band_mask])] * 60
    snr = np.max(mags[band_mask]) / np.mean(mags[band_mask])

    # --- REFINED VERDICT LOGIC (v4.1) ---
    if snr < 1.8: 
        score = 0.5
    else:
        # HUMAN criteria:
        # 1. Reliable G/R ratio (Primary marker)
        # 2. Reasonable drift (Broadened to 15.0 for motion noise)
        is_human = (gr_ratio > 1.15) and (0.3 < bpm_drift < 18.0) # Even more lenient for noisy cam
        
        # Confident Biometric Marker
        if gr_ratio > 1.30: is_human = True 

        if is_human:
            score = max(0.0, 0.25 - (gr_ratio - 1.15) * 0.5)
            if bpm_drift < 0.2: score += 0.4
        else:
            score = 0.85
            if gr_ratio < 1.05: score += 0.1
            if bpm_drift < 0.2: score += 0.1
            if snr > 25.0: score = 0.98

    tags = {
        "raw": G, "filtered": bvp_signal, "fft_xf": xf * 60, "fft_yf": mags,
        "bpm": peak_bpm, "snr": snr, "drift": bpm_drift, "gr_ratio": gr_ratio
    }
    return float(np.clip(score, 0.0, 1.0)), tags if return_signals else float(np.clip(score, 0.0, 1.0))

def plot_report(signals, score):
    """Generates a visual forensic report."""
    if not signals: return
    try:
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(signals["filtered"], color='red')
        plt.title(f"Forensic BVP Projection\nG/R Ratio: {signals['gr_ratio']:.2f} (Human > 1.15)")
        
        plt.subplot(3, 1, 2)
        plt.plot(signals["fft_xf"], signals["fft_yf"], color='blue')
        plt.axvline(x=signals["bpm"], color='orange', linestyle='--')
        plt.title(f"Frequency Analysis\nBPM: {signals['bpm']:.1f} | SNR: {signals['snr']:.2f}")
        plt.xlim(40, 160)
        
        plt.subplot(3, 1, 3)
        plt.bar(["BPM Drift", "G/R Ratio"], [signals["drift"], signals["gr_ratio"]], color=['orange', 'green'])
        plt.axhline(y=1.15, color='black', linestyle='--')
        plt.title(f"Bio-Forensic Indicators (Drift human range: 0.5 - 18.0)")
        plt.ylim(0, max(2.5, signals["gr_ratio"] + 0.5))
        
        if score == 0.5:
            label = "INCONCLUSIVE"
        else:
            label = "DEEPFAKE" if score > 0.5 else "HUMAN"
            
        plt.suptitle(f"Persona rPPG Forensic Report [v4.1]\nResult: {label} (Score: {score:.4f})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display report graph ({e})")

def run_webcam():
    """Captures 10 seconds of live video for analysis."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
        
    print("\n--- Live Webcam Capture ---")
    print("Keep your face steady and well-lit.")
    print("Press 'q' to stop early. Collecting 300 frames (~10s)...")
    
    frames = []
    mp_face_mesh = mp.solutions.face_mesh
    window_working = True
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while len(frames) < 300:
            ret, frame = cap.read()
            if not ret: break
            
            # Fallback for systems with broken cv2.imshow
            if window_working:
                try:
                    display_frame = frame.copy()
                    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        h, w, _ = frame.shape
                        for idx in [117, 346]:
                            lm = results.multi_face_landmarks[0].landmark[idx]
                            cv2.circle(display_frame, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)
                        cv2.putText(display_frame, f"Capturing: {len(frames)}/300", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "FACE NOT DETECTED", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("Persona Live Capture", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    window_working = False
                    print("\nGUI window failed to open. Switching to Blind Capture Mode...")
            
            if not window_working:
                if len(frames) % 30 == 0:
                    print(f"Progress: {len(frames)}/300 frames collected...")
            
            frames.append(frame)
                
    cap.release()
    try: cv2.destroyAllWindows()
    except: pass
    
    if len(frames) >= 150:
        print("\nAnalyzing live pulse...")
        score, signals = analyze(frames, return_signals=True)
        v_text = "INCONCLUSIVE" if score == 0.5 else ("DEEPFAKE" if score > 0.5 else "HUMAN")
        print(f"--- Result: {v_text} (Score: {score:.4f}) ---")
        plot_report(signals, score)
    else:
        print("Error: Not enough frames captured.")

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    import sys

    print("Persona rPPG Forensic Detector [v4.1]")
    print("-" * 35)
    print("1. Upload Video File")
    print("2. Launch Live Webcam")
    print("-" * 35)
    
    try:
        # Handle choice without input() if possible for non-interactive shells
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '1':
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            video_path = filedialog.askopenfilename(title="Select Video to Verify")
            root.destroy()
            
            if video_path:
                print(f"Processing: {video_path}")
                cap = cv2.VideoCapture(video_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                    if len(frames) >= 600: break
                cap.release()
                
                if len(frames) > 150:
                    score, signals = analyze(frames, return_signals=True)
                    v_text = "INCONCLUSIVE" if score == 0.5 else ("DEEPFAKE" if score > 0.5 else "HUMAN")
                    print(f"\n--- Result: {v_text} (Score: {score:.4f}) ---")
                    plot_report(signals, score)
                else: 
                    print("Error: Video too short (Need at least 150 frames).")
            else:
                print("No file selected.")
                
        elif choice == '2':
            run_webcam()
        else:
            print("Invalid choice.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
