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

def analyze(frames, return_signals=False, source="auto", fps=30.0, env_flags=None, callback=None):
    """
    Forensic rPPG Analysis v14.7 - Shaky Cam Hardening.
    - Environmental Calibration: Aggressive relaxation for extreme handheld motion.
    - Motion Self-Detection: Detects shaky cam even if env_flags is missing.
    """
    env = env_flags or {"low_light": False, "shaky": False, "grainy": False}
    frame_count = len(frames)
    
    # Pillar 5: Motion Self-Detection (If env_flags missing or backup)
    if not env.get("shaky") and frame_count > 100:
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as mesh:
            nose_movements = []
            for i in range(0, frame_count, 10):
                res = mesh.process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    l = res.multi_face_landmarks[0].landmark[1]
                    nose_movements.append([l.x, l.y])
            if len(nose_movements) > 5:
                var = np.var([m[0] for m in nose_movements]) + np.var([m[1] for m in nose_movements])
                if var > 0.0012: env["shaky"] = True # Tightened (0.0004 -> 0.0012)
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
    
    # v18.0 - Headless Optimized Extraction (Stride=3)
    bg_means = []
    stride_extraction = 3
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for i in range(0, frame_count, stride_extraction):
            frame = frames[i]
            if frame is None: continue
            h, w, _ = frame.shape
            
            # 1. Global Background Sample (Corners)
            bg_left = frame[10:60, 10:60]
            bg_right = frame[10:60, w-60:w-10]
            bg_avg = (cv2.mean(bg_left)[:3] + cv2.mean(bg_right)[:3])
            bg_means.append([bg_avg[2]/2, bg_avg[1]/2, bg_avg[0]/2]) # RGB
            
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
        
        # v17.0 CHROM Extraction (Ambient Light Resilience)
        # Projection: S = 3R - 2G
        bvp_chrom = 3*Rn - 2*Gn
        filtered_bvp = filtfilt(b, a, bvp_chrom)
        
        # Red-Dominance Check (Light Quality)
        rg_ratio = np.mean(R) / (np.mean(G) + 1e-6)
        is_red_dominant = rg_ratio > 1.8
        
        # Lockstep Correlation
        rf, gf = filtfilt(b, a, Rn), filtfilt(b, a, Gn)
        lockstep_corr = np.corrcoef(rf, gf)[0, 1] if np.std(rf)>0 and np.std(gf)>0 else 1.0
        
        # G/R Variance Ratio
        gr_var_ratio = np.var(gf) / (np.var(rf) + 1e-8)
        
        roi_data[name] = {
            "bvp": filtered_bvp, "gr_var": gr_var_ratio, 
            "lockstep": lockstep_corr, "rg_ratio": rg_ratio,
            "red_dominant": is_red_dominant
        }

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
        # Stream BVP update removed to prioritize single-result JSON
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
    
    # v17.0 - Context & Background Analysis
    bg_arr = np.array(bg_means)
    bg_var_global = np.mean(np.var(bg_arr, axis=0))
    # Spatial Continuity (Uniform Noise Check)
    face_var_avg = np.mean([np.var(np.array(m)) for m in roi_means.values()])
    noise_uniformity = abs(face_var_avg - bg_var_global) / (bg_var_global + 1e-6)
    is_organic_grain = noise_uniformity < 0.4 and bg_var_global > 0.5
    
    # v17.0 - Background-Anchor Motion Correlation
    # Track nose-tip vs background signal patterns
    nose_y = []
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=False) as mesh:
        for fr in frames[::stride]:
            res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            nose_y.append(res.multi_face_landmarks[0].landmark[1].y if res.multi_face_landmarks else 0)
    
    anchor_correlation = 0.0
    if len(nose_y) > 2 and len(bg_means) > 2:
        # Interpolate background means to match nose tracking density if needed
        bg_intensity = np.mean(bg_arr, axis=1) # Global brightness anchor
        # Normalize for correlation
        n_y = np.array(nose_y)
        bg_i = np.interp(np.linspace(0, 1, len(n_y)), np.linspace(0, 1, len(bg_intensity)), bg_intensity)
        anchor_correlation = np.abs(np.corrcoef(n_y, bg_i)[0, 1])
    
    # Global Motion (Handheld detection)
    is_handheld = bg_var_global > 1.2 or anchor_correlation > 0.85 # Anchor-aware
    
    # Red-Dominance (Light quality signal)
    avg_rg_ratio = np.mean([v["rg_ratio"] for v in roi_data.values()])
    is_low_confidence_light = avg_rg_ratio > 2.0
    
    peak_idx_abs = np.where(band_mask)[0][peak_idx_rel]
    snr = peak_val / (np.mean(yf[band_mask]) + 1e-6)
    
    # 1. Harmonic Consistency (Sine-Wave Penalty)
    # Generate a pure oscillator at the detected peak frequency
    t = np.arange(len(master_bvp)) / fs
    oscillator = np.sin(2 * np.pi * (peak_bpm / 60.0) * t)
    # Normalize oscillator and BVP for correlation
    bvp_norm = (master_bvp - np.mean(master_bvp)) / (np.std(master_bvp) + 1e-6)
    osc_norm = (oscillator - np.mean(oscillator)) / (np.std(oscillator) + 1e-6)
    sine_correlation = np.abs(np.corrcoef(bvp_norm, osc_norm)[0, 1])
    
    # 2. Fixed Frequency / HRV Penalty (Standard Deviation of IBIs)
    # Simple peak detection for Inter-Beat Interval (IBI)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(master_bvp, distance=fs*0.4) # min 0.4s between beats
    ibis = np.diff(peaks) / fs
    ibi_std = np.std(ibis) if len(ibis) > 2 else 0.05 # default to human if too short
    
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
    
    # v17.0 Signal Phase Analysis
    blue_pulse_strength = 0.0
    motion_pulse_corr = 0.0
    grd_delta = 0.0
    
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
        
        # 3. Green-Red Differential (GRD) / Blue Ghosting
        # Multi-Channel Validation: Real pulses are green-dominant.
        bf_n = np.mean([arr[:, 2] for arr in [np.array(m) for m in roi_means.values()]], axis=0)
        bf_n = bf_n / (np.mean(bf_n) + 1e-6)
        bf_f = filtfilt(b, a, bf_n)
        yf_blue = np.abs(rfft(bf_f))
        
        # Calculate SNR per channel for Differential Analysis
        blue_snr = yf_blue[peak_idx_abs] / (np.mean(yf_blue[band_mask]) + 1e-6)
        green_snr = snr # Existing snr is BVP, centered on CHROM (weighted Green)
        grd_delta = green_snr - blue_snr
        
        blue_pulse_strength = yf_blue[peak_idx_abs] / (peak_val + 1e-6)
        
        # 4. Motion-Pulse Correlation (Overlay Check)
        # Real pulses are disrupted by motion. AI overlays often remain constant.
        # We check correlation between global motion and BVP envelope.
        from scipy.signal import hilbert
        bvp_envelope = np.abs(hilbert(master_bvp))
        # Use nose variance as motion proxy
        nose_motion = []
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=False) as mesh:
            for fr in frames[::stride]:
                res = mesh.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                nose_motion.append(res.multi_face_landmarks[0].landmark[1].y if res.multi_face_landmarks else 0)
        # Interpolate motion to match BVP length
        if len(nose_motion) > 2:
            motion_interp = np.interp(np.linspace(0, 1, len(master_bvp)), np.linspace(0, 1, len(nose_motion)), nose_motion)
            motion_pulse_corr = np.abs(np.corrcoef(motion_interp, bvp_envelope)[0, 1])
        else:
            motion_pulse_corr = 0.0

    # Spectral Entropy (Pillar 1: Shoulder Analysis)
    peak_region = yf[max(0, peak_idx_abs-5):min(len(yf), peak_idx_abs+6)]
    psd_norm = peak_region**2 / (np.sum(peak_region**2) + 1e-6)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Diagnostic Print for Tuning (v17.0)
    print(f"[*] Forensic | Sine-C: {sine_correlation:.2f} | Anchor-C: {anchor_correlation:.2f} | R/G: {avg_rg_ratio:.2f} | Entropy: {spectral_entropy:.2f} | B-Grd: {grd_delta:.2f} | B-Ghost: {blue_pulse_strength:.2f}")

    # --- v16.0 CONTEXT-AWARE SCORING MATRIX ---
    score = 0.5
    ai_hits = 0
    
    # 1. Sine-Wave Perfection (Biological Impossibility)
    if sine_correlation > 0.992: # Mathematically "Too Pure"
        score += 0.5; ai_hits += 2
    elif sine_correlation > 0.985:
        score += 0.25; ai_hits += 1
        
    # 2. Fixed Frequency / HRV Stability
    if ibi_std < 0.008: 
        score += 0.45; ai_hits += 2
    elif ibi_std < 0.015:
        score += 0.2; ai_hits += 1

    # 3. Blue Ghosting / Channel Leakage
    if blue_pulse_strength > 0.8: 
        score += 0.45; ai_hits += 2
    elif blue_pulse_strength > 0.6:
        score += 0.2; ai_hits += 1

    # 4. Motion-Pulse Correlation (Floating Overlay)
    # Real pulses are disrupted by motion. AI overlays often remain constant.
    # We ignore this if we detect shaky cam (since BVP is naturally noisy then).
    if not is_handheld:
        if motion_pulse_corr < 0.05: # Zero disruption = Overlay
            score += 0.45; ai_hits += 2
        elif motion_pulse_corr < 0.15:
            score += 0.2; ai_hits += 1

    # 5. Spatial Inconsistency (The Patchwork Test)
    # Organic Chaos: Major relaxation for handheld noise
    # Handheld humans (data_10) can have insane ROI Variance due to bin shift.
    roi_thresh_high = 45.0 if is_handheld else (18.0 if is_organic_grain else 12.0)
    roi_thresh_mid = 25.0 if is_handheld else (12.0 if is_organic_grain else 7.0)
    
    if roi_variance > roi_thresh_high: 
        score += 0.5; ai_hits += 2
    elif roi_variance > roi_thresh_mid:
        score += 0.25; ai_hits += 1

    # 6. Robotic Stability (Legacy Check)
    if bpm_drift < 0.03: 
        score += 0.35; ai_hits += 1

    # --- BIOLOGICAL REWARDS (v17.0 Real World Aware) ---
    reward_multiplier = 0.0 if ai_hits >= 2 else (0.5 if ai_hits == 1 else 1.0)
    
    # Reward Anchor Correlation (Camera Shake evidence)
    if anchor_correlation > 0.85:
        score -= 0.5 * reward_multiplier
        
    # Reward Natural Chaos (Handheld evidence)
    if is_handheld:
        score -= 0.25 * reward_multiplier
        
    # Reward Uniform Grain (Sensor Noise evidence)
    if is_organic_grain:
        score -= 0.25 * reward_multiplier
        
    # Reward GRD (Green signal surviving in noise)
    if grd_delta > 2.5:
        score -= 0.35 * reward_multiplier
        
    # Reward Natural Complexity (Entropy & Channel Jitter)
    if spectral_entropy > 1.8 and gr_lockstep < 0.99:
        score -= 0.3 * reward_multiplier
        
    # Reward Natural Vitality (IBI Drift)
    # Talking humans have drift. Shaky recordings have massive drift.
    drift_limit_min = 0.5 if is_handheld else 2.0
    drift_limit_max = 50.0 if is_handheld else 15.0
    if drift_limit_min < bpm_drift < drift_limit_max and roi_variance < roi_thresh_high:
        score -= 0.5 * reward_multiplier

    final_score = float(np.clip(score, 0.0, 1.0))
    tags = {
        "filtered": master_bvp.tolist() if isinstance(master_bvp, np.ndarray) else master_bvp, 
        "xf": (xf*60).tolist(), "yf": yf.tolist(), "bpm": peak_bpm, 
        "snr": snr, "drift": bpm_drift, "gr_var": avg_gr_var, 
        "gr_lock": gr_lockstep, "gr_lag": gr_phase_lag, 
        "entropy": spectral_entropy, "roi_var": roi_variance, "fps": fs,
        "rg_ratio": avg_rg_ratio, "anchor_corr": anchor_correlation
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
        plt.suptitle(f"Persona Real-World Forensic [v17.0]\nResult: {label} (Score: {score:.4f})", fontsize=14)
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
    # cap.release(); cv2.destroyAllWindows() # GUI cleanup removed for headless
    cap.release()
    if len(frames) >= 75:
        actual_fps = len(frames) / duration if duration > 0 else 30.0
        print(f"[*] Analyzing {len(frames)} frames (@{actual_fps:.1f} FPS)...")
        score, sigs = analyze(frames, return_signals=True, source="webcam", fps=actual_fps)
        label = "DEEPFAKE" if score > 0.5 else "HUMAN"
        print(f"--- Result: {label} (Score: {score:.4f}) ---")
        # plot_report(sigs, score) # Headless mode

if __name__ == "__main__":
    import time
    print("Persona rPPG Forensic [v17.0]\n1. Upload | 2. Webcam")
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
