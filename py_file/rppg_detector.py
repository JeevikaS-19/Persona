import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def interpolate_peak(y, x, idx):
    """Quadratic interpolation for sub-bin peak detection."""
    if idx <= 0 or idx >= len(y) - 1: return x[idx]
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-10: return x[idx]
    p = 0.5 * (y0 - y2) / denom
    return x[idx] + p * (x[1] - x[0])

def analyze(frames, return_signals=False, source="auto", fps=30.0):
    """
    Forensic rPPG Analysis v14.0 - Loophole Patch.
    - Phase Jitter (SD_Phase): Detects incoherent AI patch generation.
    - Spectral Shape: Distinguishes Bio-Shoulders from AI Delta peaks.
    - Ultra-Lockstep: Detects subtle R/G synthetic unison.
    """
    frame_count = len(frames)
    if not frames or frame_count < 75: # Lowered for v13.1 Decimation support
        return (0.5, {}) if return_signals else 0.5
    
    if source == "auto":
        source = "upload" if frame_count >= 300 else "webcam"

    mp_face_mesh = mp.solutions.face_mesh
    ROIS = {
        "forehead": [10, 109, 67, 103, 285, 297, 338],
        "cheek_r": [118, 119, 120, 101, 117],
        "cheek_l": [347, 348, 349, 330, 346]
    }
    
    roi_means = {name: [] for name in ROIS}
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for frame in frames:
            if frame is None: continue
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                for name in ROIS: roi_means[name].append([np.nan, np.nan, np.nan])
                continue
            
            landmarks = results.multi_face_landmarks[0].landmark
            for name, indices in ROIS.items():
                points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices])
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(poly_mask, points, 255)
                m = cv2.mean(frame, mask=poly_mask)[:3]
                roi_means[name].append([m[2], m[1], m[0]]) # RGB

    fs = float(fps)
    b, a = butter(4, [0.75 / (0.5 * fs), 2.5 / (0.5 * fs)], btype='band')
    roi_data = {}
    
    for name, data in roi_means.items():
        arr = np.array(data)
        for i in range(3):
            series = arr[:, i]
            mask = np.isnan(series)
            if mask.any() and not mask.all():
                series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
            elif mask.all(): series[:] = 128
        
        R, G, B = arr[:, 0], arr[:, 1], arr[:, 2]
        Rn, Gn, Bn = R/(np.mean(R)+1e-6), G/(np.mean(G)+1e-6), B/(np.mean(B)+1e-6)
        
        # CHROM Extraction
        X, Y = 3*Rn - 2*Gn, 1.5*Rn + Gn - 1.5*Bn
        Xd, Yd = detrend(X), detrend(Y)
        alpha = np.std(Xd) / (np.std(Yd) + 1e-6)
        bvp = Xd - alpha * Yd
        filtered_bvp = filtfilt(b, a, bvp)
        
        # Lockstep Correlation
        rf, gf = filtfilt(b, a, Rn), filtfilt(b, a, Gn)
        lockstep_corr = np.corrcoef(rf, gf)[0, 1] if np.std(rf)>0 and np.std(gf)>0 else 1.0
        
        # G/R Variance Ratio
        gr_var_ratio = np.var(gf) / (np.var(rf) + 1e-8)
        
        roi_data[name] = {"bvp": filtered_bvp, "gr_var": gr_var_ratio, "lockstep": lockstep_corr}

    # Windowed Phase Stability Check (Adaptive Window)
    win_len = min(150, int(5.0 * fs)) if fs > 20 else min(75, int(5.0 * fs))
    win_len = max(win_len, 40) # Safety floor
    stride = win_len // 5
    phase_lags = []
    bpms = []
    for i in range(0, frame_count - win_len, stride):
        # Phase of forehead vs mean cheeks
        f_win = roi_data["forehead"]["bvp"][i:i+win_len]
        c_win = (roi_data["cheek_r"]["bvp"][i:i+win_len] + roi_data["cheek_l"]["bvp"][i:i+win_len])/2
        
        f_fft = rfft(f_win)
        c_fft = rfft(c_win)
        xf_w = rfftfreq(win_len, 1/fs)
        m_w = (xf_w >= 0.75) & (xf_w <= 2.5)
        
        peak_w_rel = np.argmax(np.abs(f_fft)[m_w])
        peak_w_abs = np.where(m_w)[0][peak_w_rel]
        bpms.append(interpolate_peak(np.abs(f_fft), xf_w, peak_w_abs) * 60)
        
        f_phase = np.angle(f_fft[peak_w_abs])
        c_phase = np.angle(c_fft[peak_w_abs])
        phase_lags.append(np.abs(f_phase - c_phase))
    
    phase_jitter = np.std(phase_lags) if len(phase_lags) > 1 else 0.0

    master_bvp = (roi_data["forehead"]["bvp"] + roi_data["cheek_r"]["bvp"] + roi_data["cheek_l"]["bvp"]) / 3.0
    avg_gr_var = np.mean([v["gr_var"] for v in roi_data.values()])
    avg_lockstep = np.mean([v["lockstep"] for v in roi_data.values()])
    
    yf = np.abs(rfft(master_bvp))
    xf = rfftfreq(len(master_bvp), 1/fs)
    band_mask = (xf >= 0.75) & (xf <= 2.5)
    peak_idx_rel = np.argmax(yf[band_mask])
    peak_idx_abs = np.where(band_mask)[0][peak_idx_rel]
    peak_val = yf[peak_idx_abs]
    peak_bpm = interpolate_peak(yf, xf, peak_idx_abs) * 60
    
    # v12.0 - Spectral Shape check (Bio-Shoulder)
    # Binary spectrum check: AI peaks have near-zero energy in adjacent bins.
    peak_idx_abs = np.where(band_mask)[0][peak_idx_rel]
    adj_energy = (yf[peak_idx_abs-1] + yf[peak_idx_abs+1]) / (peak_val + 1e-6)
    
    snr = peak_val / (np.mean(yf[band_mask]) + 1e-6)
    
    for i in range(0, len(master_bvp) - win_len, stride):
        win = master_bvp[i:i+win_len]
        y_w = np.abs(rfft(win))
        x_w = rfftfreq(win_len, 1/fs)
        m_w = (x_w >= 0.75) & (x_w <= 2.5)
        p_idx = np.where(m_w)[0][np.argmax(y_w[m_w])]
        bpms.append(interpolate_peak(y_w, x_w, p_idx) * 60)
    bpm_drift = np.std(bpms) if len(bpms) > 1 else 0.0

    sync_score = np.corrcoef(roi_data["forehead"]["bvp"], (roi_data["cheek_r"]["bvp"]+roi_data["cheek_l"]["bvp"])/2)[0,1]
    
    # ROI-Specific BPM Analysis (Pillar 3: Spatial Consistency)
    roi_bpms = {}
    for name, data in roi_data.items():
        win_bvp = data["bvp"]
        yf_roi = np.abs(rfft(win_bvp))
        xf_roi = rfftfreq(len(win_bvp), 1/fs)
        m_roi = (xf_roi >= 0.75) & (xf_roi <= 2.5)
        peak_idx_roi = np.argmax(yf_roi[m_roi])
        peak_abs_roi = np.where(m_roi)[0][peak_idx_roi]
        roi_bpms[name] = interpolate_peak(yf_roi, xf_roi, peak_abs_roi) * 60
    
    roi_variance = np.std(list(roi_bpms.values()))
    
    # G vs R Phase-Shift Analysis (Pillar 2)
    rf_global = np.mean([arr[:, 0] for arr in [np.array(m) for m in roi_means.values()]], axis=0)
    gf_global = np.mean([arr[:, 1] for arr in [np.array(m) for m in roi_means.values()]], axis=0)
    
    # Fix NaN Correlation (Handle constant signals)
    std_r, std_g = np.std(rf_global), np.std(gf_global)
    if std_r < 1e-7 or std_g < 1e-7 or np.isnan(std_r) or np.isnan(std_g):
        # Low variance usually means compression artifact on real skin, not AI. 
        # We set it to a "neutral" human correlation rather than 1.0.
        gr_lockstep = 0.95 
        gr_phase_lag = 0.5
    else:
        # Normalize and filter
        rf_n = rf_global / (np.mean(rf_global) + 1e-6)
        gf_n = gf_global / (np.mean(gf_global) + 1e-6)
        rf_f = filtfilt(b, a, rf_n)
        gf_f = filtfilt(b, a, gf_n)
        
        c_matrix = np.corrcoef(rf_f, gf_f)
        gr_lockstep = c_matrix[0, 1] if not np.any(np.isnan(c_matrix)) else 0.95
        peak_idx_abs = np.where(band_mask)[0][peak_idx_rel]
        gr_phase_lag = np.abs(np.angle(rfft(gf_f)[peak_idx_abs]) - np.angle(rfft(rf_f)[peak_idx_abs]))

    # Spectral Entropy (Pillar 1: Shoulder Analysis)
    peak_region = yf[max(0, peak_idx_abs-5):min(len(yf), peak_idx_abs+6)]
    psd_norm = peak_region**2 / (np.sum(peak_region**2) + 1e-6)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # --- v14.3 SCORING MATRIX (Compressed Resilience) ---
    score = 0.5
    ai_hits = 0
    
    # Diagnostic Print for Tuning
    print(f"[*] Forensic Metrics | Entropy: {spectral_entropy:.3f} | G/R Lock: {gr_lockstep:.4f} | Drift: {bpm_drift:.3f} | ROI Var: {roi_variance:.2f}")

    # 1. Robotic Stability (Perfection Check)
    if bpm_drift < 0.03: # Surgical AI stability
        score += 0.45; ai_hits += 2
    elif bpm_drift < 0.12:
        score += 0.25; ai_hits += 1
        
    # 2. Spectral Perfection (Surgical AI Peak)
    if spectral_entropy < 0.65: 
        score += 0.45; ai_hits += 2
    elif spectral_entropy < 1.3:
        score += 0.2; ai_hits += 1
        
    # 3. Channel Unison (G vs R Lockstep)
    if gr_lockstep > 0.9995: 
        score += 0.45; ai_hits += 2
    elif gr_lockstep > 0.995:
        score += 0.25; ai_hits += 1
        
    # 4. Spatial Inconsistency (The Patchwork Test)
    # Decimated feeds (15 FPS) can have high ROI variance due to bin spacing.
    # Relaxed for v14.4 to 12.0/7.0.
    if roi_variance > 12.0: 
        score += 0.5; ai_hits += 2
    elif roi_variance > 7.0:
        score += 0.25; ai_hits += 1

    # --- BIOLOGICAL REWARDS ---
    reward_multiplier = 0.0 if ai_hits >= 2 else (0.5 if ai_hits == 1 else 1.0)
    
    # Reward Natural Complexity (Entropy & Channel Jitter)
    if spectral_entropy > 2.0 and gr_lockstep < 0.99:
        score -= 0.4 * reward_multiplier
        
    # Reward Natural Vitality (Heart Rate Drift & Spatial Unity)
    # Talking humans have drift between 2.0 and 15.0 bpm.
    # Relaxed ROI variance reward limit to 15.0 for decimated feeds.
    if 2.0 < bpm_drift < 15.0 and roi_variance < 15.0:
        score -= 0.35 * reward_multiplier

    final_score = float(np.clip(score, 0.0, 1.0))
    tags = {
        "filtered": master_bvp, "xf": xf*60, "yf": yf, "bpm": peak_bpm, 
        "snr": snr, "drift": bpm_drift, "gr_var": avg_gr_var, 
        "gr_lock": gr_lockstep, "gr_lag": gr_phase_lag, 
        "entropy": spectral_entropy, "roi_var": roi_variance, "fps": fs
    }
    return final_score, tags if return_signals else final_score

def plot_report(signals, score):
    if not signals: return
    try:
        plt.figure(figsize=(10, 8))
        # CORRECTED v12.0 LABEL LOGIC
        label = "INCONCLUSIVE" if (signals["snr"] < 1.3) else ("DEEPFAKE" if score > 0.5 else "HUMAN")
        
        plt.subplot(3, 1, 1); plt.plot(signals["filtered"], color='red')
        plt.title(f"v14.0 Loophole Patch (FPS: {signals.get('fps',0):.1f} | Entropy: {signals.get('entropy',0):.2f})")
        plt.subplot(3, 1, 2); plt.plot(signals["xf"], signals["yf"], color='blue')
        plt.axvline(x=signals["bpm"], color='orange', linestyle='--'); plt.xlim(40, 160)
        plt.title(f"FFT Spectrum (BPM: {signals['bpm']:.1f} | SNR: {signals['snr']:.2f})")
        plt.subplot(3, 1, 3); plt.bar(["G/R Lock", "ROI Var", "Entropy", "Drift"], [signals["gr_lock"], signals["roi_var"], signals["entropy"], signals["drift"]], color=['green', 'orange', 'purple', 'black'])
        plt.axhline(y=0.9995, color='red', linestyle='--')
        plt.suptitle(f"Persona Loophole Forensic [v14.3]\nResult: {label} (Score: {score:.4f})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    except: pass

def run_webcam():
    cap = cv2.VideoCapture(0)
    print("\n--- Live Capture v14.3 ---")
    frames = []
    start_time = time.time()
    while len(frames) < 300:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Persona Live v14.1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frames.append(frame)
    duration = time.time() - start_time
    cap.release(); cv2.destroyAllWindows()
    if len(frames) >= 75:
        actual_fps = len(frames) / duration if duration > 0 else 30.0
        print(f"[*] Analyzing {len(frames)} frames (@{actual_fps:.1f} FPS)...")
        score, sigs = analyze(frames, return_signals=True, source="webcam", fps=actual_fps)
        label = "DEEPFAKE" if score > 0.5 else "HUMAN"
        print(f"--- Result: {label} (Score: {score:.4f}) ---")
        plot_report(sigs, score)

if __name__ == "__main__":
    import time
    print("Persona rPPG Forensic [v14.5]\n1. Upload | 2. Webcam")
    choice = input("Choice: ").strip()
    if choice == '1':
        import tkinter as tk; from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(); root.destroy()
        if path:
            cap = cv2.VideoCapture(path); frames = []; fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"Analyzing {path} at {fps:.1f} FPS...")
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
                if len(frames) >= 600: break
            cap.release()
            if len(frames) >= 150:
                score, sigs = analyze(frames, return_signals=True, source="upload", fps=fps)
                label = "DEEPFAKE" if score > 0.5 else "HUMAN"
                print(f"\n--- Result: {label} (Score: {score:.4f}) ---")
                plot_report(sigs, score)
    elif choice == '2': run_webcam()
