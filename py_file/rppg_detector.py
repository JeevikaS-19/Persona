import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def analyze(frames, return_signals=False, source="auto"):
    """
    Forensic rPPG Analysis v8.5 - Source Aware.
    - Webcam Mode: Strict drift and spectral anchors.
    - Upload Mode: Gaussian Smoothing for codec noise + Relaxed drift.
    - Buffer Gate: Minimum 150 (Webcam) or 300 (Upload) frames.
    """
    frame_count = len(frames)
    if not frames or frame_count < 150:
        return (0.5, None) if return_signals else 0.5
    
    # Auto-detect source if not specified
    if source == "auto":
        source = "upload" if frame_count >= 300 else "webcam"

    # Strict gate for uploads (Forensic depth)
    if source == "upload" and frame_count < 300:
        print("Warning: Uploaded file should be at least 10s (300 frames) for reliable analysis.")

    mp_face_mesh = mp.solutions.face_mesh
    CHEEK_R = [117, 118, 119, 120, 121, 101]
    CHEEK_L = [346, 347, 348, 349, 350, 330]
    FOREHEAD = [10, 109, 67, 103, 285, 297, 338]
    ROI_INDICES = CHEEK_R + CHEEK_L + FOREHEAD
    
    raw_rgb_means = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for frame in frames:
            if frame is None: continue
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raw_rgb_means.append([np.nan, np.nan, np.nan])
                continue
            
            landmarks = results.multi_face_landmarks[0].landmark
            points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in ROI_INDICES])
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(poly_mask, points, 255)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            valid_mask = cv2.bitwise_and(poly_mask, cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1])
            if cv2.countNonZero(valid_mask) < 100: valid_mask = poly_mask
            mean_vals = cv2.mean(frame, mask=valid_mask)[:3] 
            raw_rgb_means.append([mean_vals[2], mean_vals[1], mean_vals[0]]) # R, G, B

    rgb_array = np.array(raw_rgb_means)
    for i in range(3):
        series = rgb_array[:, i]
        mask_nan = np.isnan(series)
        if mask_nan.any():
            if not mask_nan.all():
                series[mask_nan] = np.interp(np.flatnonzero(mask_nan), np.flatnonzero(~mask_nan), series[~mask_nan])
            else: return (0.5, None) if return_signals else 0.5
        
        # v8.5 - Gaussian Temporal Smoothing for Compression Noise
        if source == "upload":
            # 7-frame Gaussian blur to mop up inter-frame codec artifacts
            rgb_array[:, i] = cv2.GaussianBlur(series.reshape(-1, 1), (1, 7), 0).flatten()
        else:
            # 5-frame SMA for live sensor noise
            rgb_array[:, i] = np.convolve(series, np.ones(5)/5, mode='same')

    R, G, B = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]
    fs = 30.0
    R_p, G_p, B_p = R/np.mean(R), G/np.mean(G), B/np.mean(B)
    pos_signal = (1.0 - (R_p / 2.0)) + G_p - (B_p / 2.0)
    
    b, a = butter(4, [0.75 / (0.5 * fs), 2.5 / (0.5 * fs)], btype='band')
    bvp_signal = filtfilt(b, a, pos_signal)

    # Drift Calculation
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

    # Biological Anchors
    Rf, Gf = filtfilt(b, a, R_p), filtfilt(b, a, G_p)
    gr_ratio = np.max(np.abs(rfft(Gf))) / (np.max(np.abs(rfft(Rf))) + 1e-6)
    yf_all = np.abs(rfft(bvp_signal))
    xf_all = rfftfreq(len(bvp_signal), 1/fs)
    band_mask = (xf_all >= 0.75) & (xf_all <= 2.5)
    peak_bpm = xf_all[band_mask][np.argmax(yf_all[band_mask])] * 60
    snr = np.max(yf_all[band_mask]) / np.mean(yf_all[band_mask])

    # --- SCORING v8.5 (Source Aware) ---
    score = 0.5
    
    if source == "upload":
        # RELAXED MODE: Focus on SNR Peak (Codec Awareness)
        if snr > 1.8: # Even squashed by MP4, human pulse is rhythmic
            score -= 0.25
        if gr_ratio > 0.98:
            score -= 0.15
        if bpm_drift > 50.0: # Only penalize astronomical glitched values
            score += 0.3
        if gr_ratio < 0.90: # Severe non-biological tint
            score += 0.3
    else:
        # STRICT MODE: Live Webcam Stability
        if gr_ratio > 1.05:
            score -= 0.3 # Strong biological favor
        if bpm_drift < 3.0: # Predictable biological rhythm
            score -= 0.1
        if bpm_drift > 15.0 and gr_ratio < 1.0:
            score += 0.4 # Chaotic non-bio signal
        if bpm_drift < 0.5 and snr > 5.0:
            score += 0.4 # Synthetic perfection marker

    # Global Physio Match
    if 48 <= peak_bpm <= 110:
        score -= 0.1

    final_score = float(np.clip(score, 0.0, 1.0))
    tags = {"filtered": bvp_signal, "fft_xf": xf_all*60, "fft_yf": yf_all, "bpm": peak_bpm, "snr": snr, "drift": bpm_drift, "gr_ratio": gr_ratio, "source": source}
    return final_score, tags if return_signals else final_score

def plot_report(signals, score):
    if not signals: return
    try:
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1); plt.plot(signals["filtered"], color='red'); plt.title(f"Forensic Pulse (Source: {signals['source']})")
        plt.subplot(3, 1, 2); plt.plot(signals["fft_xf"], signals["fft_yf"], color='blue'); plt.axvline(x=signals["bpm"], color='orange', linestyle='--')
        plt.title(f"Analysis (BPM: {signals['bpm']:.1f} | SNR: {signals['snr']:.2f})"); plt.xlim(40, 160)
        plt.subplot(3, 1, 3); plt.bar(["Drift", "G/R Ratio"], [signals["drift"], signals["gr_ratio"]], color=['orange', 'green'])
        plt.axhline(y=0.98, color='black', linestyle='--'); plt.axhline(y=15.0, color='gray', linestyle=':')
        label = "INCONCLUSIVE" if score == 0.5 else ("DEEPFAKE" if score > 0.5 else "HUMAN")
        plt.suptitle(f"Persona Forensic [v8.5 - {signals['source'].upper()}]\nResult: {label} (Score: {score:.4f})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    except: pass

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return
    print("\n--- Live Capture v8.5 ---"); frames = []
    while len(frames) < 300:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Persona v8.5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frames.append(frame)
    cap.release(); cv2.destroyAllWindows()
    if len(frames) >= 150:
        score, signals = analyze(frames, return_signals=True, source="webcam")
        v_text = "INCONCLUSIVE" if score == 0.5 else ("DEEPFAKE" if score > 0.5 else "HUMAN")
        print(f"--- Result: {v_text} (Score: {score:.4f}) ---")
        if signals: print(f"SNR: {signals['snr']:.2f} | G/R: {signals['gr_ratio']:.2f} | Drift: {signals['drift']:.2f}")
        plot_report(signals, score)

if __name__ == "__main__":
    import tkinter as tk; from tkinter import filedialog; import sys
    print("Persona rPPG Forensic [v8.5]"); print("1. Upload | 2. Webcam")
    try:
        choice = input("Enter choice (1/2): ").strip()
        if choice == '1':
            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
            video_path = filedialog.askopenfilename(); root.destroy()
            if video_path:
                cap = cv2.VideoCapture(video_path); frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                    if len(frames) >= 600: break
                cap.release()
                if len(frames) >= 150:
                    score, signals = analyze(frames, return_signals=True, source="upload")
                    v_text = "INCONCLUSIVE" if score == 0.5 else ("DEEPFAKE" if score > 0.5 else "HUMAN")
                    print(f"\n--- Result: {v_text} (Score: {score:.4f}) ---")
                    plot_report(signals, score)
        elif choice == '2': run_webcam()
    except KeyboardInterrupt: sys.exit(0)
    except Exception as e: print(f"Error: {e}")
