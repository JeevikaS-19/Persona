import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def analyze(frames, return_signals=False):
    """
    Forensic rPPG Analysis refined with User Feedback:
    - POS Method for robust pulse extraction.
    - HRV Drift (StDev) threshold increased to 2.0 to account for compression noise.
    - G/R Spectral Ratio prioritized as a strong human indicator.
    - SNR < 2.0 returns "Inconclusive" (0.5).
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

    # --- REFINED VERDICT LOGIC ---
    
    # 1. Quality Check (Inconclusive if SNR too low)
    if snr < 2.0:
        score = 0.5
    else:
        # 2. Bio-Consistency Check
        # Human criteria:
        # - Stronger Green Pulse (Ratio > 1.15)
        # - Moderate Drift (0.5 < Drift < 6.0)
        # - SNR > 2.0
        
        is_human = (gr_ratio > 1.15) and (0.4 < bpm_drift < 8.0)
        
        if is_human:
            # Score scales: Better ratio and reasonable drift = lower deepfake score
            score = max(0.0, 0.3 - (gr_ratio - 1.15) - (snr * 0.01))
            # Penalize the score if drift is extremely low (AI perfection)
            if bpm_drift < 0.3: score += 0.4
        else:
            # Deepfake markers:
            # - Ratio near 1.0 (BVP noise is grayscale)
            # - Drift is near 0.0 (too static)
            score = 0.85
            if gr_ratio < 1.05: score += 0.1
            if bpm_drift < 0.3: score += 0.1
            if snr > 20.0: score = 0.98 # Synthetic perfection

    tags = {
        "raw": G, "filtered": bvp_signal, "fft_xf": xf * 60, "fft_yf": mags,
        "bpm": peak_bpm, "snr": snr, "drift": bpm_drift, "gr_ratio": gr_ratio
    }
    return float(np.clip(score, 0.0, 1.0)), tags if return_signals else float(np.clip(score, 0.0, 1.0))

def plot_report(signals, score):
    if not signals: return
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(signals["filtered"], color='red')
    plt.title(f"Step C: POS Pulse Extraction\nBVP Strength Ratio: {signals['gr_ratio']:.2f} (Human > 1.15)")
    
    plt.subplot(3, 1, 2)
    plt.plot(signals["fft_xf"], signals["fft_yf"], color='blue')
    plt.axvline(x=signals["bpm"], color='orange', linestyle='--')
    plt.title(f"Step D: FFT Spectrum Analysis\nPulse BPM: {signals['bpm']:.1f} | SNR: {signals['snr']:.2f}")
    plt.xlim(40, 160)
    
    plt.subplot(3, 1, 3)
    plt.bar(["BPM Drift", "G/R Ratio"], [signals["drift"], signals["gr_ratio"]], color=['orange', 'green'])
    plt.axhline(y=1.15, color='black', linestyle='--', label='Ratio Threshold')
    plt.title(f"Forensic Indicators\nVerdict is Inconclusive if SNR < 2.0")
    plt.ylim(0, max(2.5, signals["gr_ratio"] + 0.5))
    
    if score == 0.5:
        label = "INCONCLUSIVE (Low Quality)"
    else:
        label = "DEEPFAKE" if score > 0.5 else "HUMAN"
        
    plt.suptitle(f"Refined Forensic rPPG Report\nResult: {label} (Score: {score:.4f})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    print("Persona rPPG Forensic Detector [v3.0 - Refined]")
    video_path = filedialog.askopenfilename(title="Select Video to Verify")
    if video_path:
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
            if score == 0.5:
                v_text = "INCONCLUSIVE"
            else:
                v_text = "DEEPFAKE" if score > 0.5 else "HUMAN"
            print(f"\n--- Result: {v_text} ---")
            print(f"Score: {score:.4f} | Drift: {signals['drift']:.2f} | G/R Ratio: {signals['gr_ratio']:.2f} | SNR: {signals['snr']:.2f}")
            plot_report(signals, score)
        else: print("Error: Video too short for reliable analysis.")
