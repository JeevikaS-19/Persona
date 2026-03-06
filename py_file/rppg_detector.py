import cv2
import numpy as np
import mediapipe_compat  # noqa
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

def analyze(frames, return_signals=False, source="auto", fps=30.0, env_flags=None, callback=None, precomputed_landmarks=None):
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
    if not frames or frame_count < 20:  # Lowered: 20 frames = ~3s min at 6fps
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
    
    bg_means = []
    nose_y_first_pass = []  # Capture nose during ROI pass — avoids second FaceMesh open
    is_cropped = (precomputed_landmarks is not None)
    stride_extraction = 1 if is_cropped else max(1, int(fps / 15.0))
    
    if is_cropped:
        for i in range(0, frame_count, stride_extraction):
            frame = frames[i]
            if frame is None: continue
            h, w, _ = frame.shape
            
            # Disable background sampling since it's a tight face ROI
            bg_means.append([0.0, 0.0, 0.0])
            
            lms = precomputed_landmarks[i] if i < len(precomputed_landmarks) else None
            if lms is None:
                for name in ROIS: roi_means[name].append([np.nan, np.nan, np.nan])
                nose_y_first_pass.append(0)
                continue
            
            nose_y_first_pass.append(lms[1, 1])  # Nose-tip Y reused below
            for name, indices in ROIS.items():
                points = np.array([(int(lms[idx, 0] * w), int(lms[idx, 1] * h)) for idx in indices])
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(poly_mask, points, 255)
                m = cv2.mean(frame, mask=poly_mask)[:3]
                roi_means[name].append([m[2], m[1], m[0]]) # RGB
    else:
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
                    nose_y_first_pass.append(0)
                    continue
                
                landmarks = results.multi_face_landmarks[0].landmark
                nose_y_first_pass.append(landmarks[1].y)  # Nose-tip Y reused below
                for name, indices in ROIS.items():
                    points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices])
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(poly_mask, points, 255)
                    m = cv2.mean(frame, mask=poly_mask)[:3]
                    roi_means[name].append([m[2], m[1], m[0]]) # RGB

    fs = float(fps) / stride_extraction
    nyq = 0.5 * fs
    lowcut = min(0.75, nyq * 0.9)
    highcut = min(2.5, nyq * 0.95)
    
    # Adaptive filter order: filtfilt needs signal > 3*(2*order+1) samples.
    # At 6fps/stride-3, BVP is ~10 points — order 2 gives padlen=15, order 4 gives 27.
    estimated_bvp_len = frame_count // stride_extraction
    filt_order = 2 if estimated_bvp_len < 30 else 4
    if lowcut < highcut:
        b, a = butter(filt_order, [lowcut / nyq, highcut / nyq], btype='band')
    else:
        b, a = butter(1, 0.99, btype='low')
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
        
        # Full CHROM Algorithm — de Haan & Jeanne (2013)
        # Two orthogonal projections; std ratio cancels motion-correlated specular noise
        X = 3*Rn - 2*Gn                       # Hue-plane component
        Y = 1.5*Rn + Gn - 1.5*Bn             # Luminance-orthogonal component
        std_x, std_y = np.std(X), np.std(Y)
        alpha = (std_x / (std_y + 1e-8))       # Noise-cancellation ratio
        bvp_chrom = X - alpha * Y              # Motion-noise cancelled BVP
        filtered_bvp = filtfilt(b, a, bvp_chrom)
        
        # Signal quality: SNR at peak vs band mean
        yf_roi = np.abs(rfft(filtered_bvp))
        xf_roi = rfftfreq(len(filtered_bvp), 1/fs)
        band_mask_roi = (xf_roi >= 0.75) & (xf_roi <= 2.5)
        roi_snr = (np.max(yf_roi[band_mask_roi]) / (np.mean(yf_roi[band_mask_roi]) + 1e-6)) \
                  if np.any(band_mask_roi) else 1.0
        
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
            "red_dominant": is_red_dominant, "snr": roi_snr
        }

    # Windowed Phase Stability Check (Adaptive Window)
    win_len = min(150, int(5.0 * fs)) if fs > 20 else min(75, int(5.0 * fs))
    win_len = max(win_len, 40) # Safety floor
    stride = win_len // 5
    phase_lags = []
    bpms = []
    bvp_len = len(roi_data["forehead"]["bvp"])
    # Recompute win_len relative to decimated BVP length (stride_extraction=3)
    win_len = min(win_len, bvp_len)
    stride = max(1, win_len // 5)
    for i in range(0, bvp_len - win_len, stride):
        # Phase of forehead vs mean cheeks
        f_win = roi_data["forehead"]["bvp"][i:i+win_len]
        c_win = (roi_data["cheek_r"]["bvp"][i:i+win_len] + roi_data["cheek_l"]["bvp"][i:i+win_len])/2
        actual_len = len(f_win)
        if actual_len < 10: continue  # Skip too-short slices
        
        f_fft = rfft(f_win)
        c_fft = rfft(c_win)
        xf_w = rfftfreq(actual_len, 1/fs)  # Match to actual slice, not win_len
        m_w = (xf_w >= 0.75) & (xf_w <= 2.5)
        
        if not np.any(m_w): continue  # No valid band at this FPS/length combo
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
    
    # v17.0 - Context & Background Analysis
    if is_cropped:
        # Task 3: Background Noise Fix - Use internal reference (forehead) to calibrate lighting noise on tight crops
        fh_arr = np.array(roi_means.get("forehead", []))
        if len(fh_arr) > 0 and not np.any(np.isnan(fh_arr)):
            bg_var_global = np.mean(np.var(fh_arr, axis=0))
        else:
            bg_var_global = 0.0
            
        # Check if cheeks have similar variance to forehead (uniform thermal/sensor noise)
        cheeks_var = np.mean([np.var(np.array(m)) for k, m in roi_means.items() if "cheek" in k and len(m) > 0])
        noise_uniformity = abs(cheeks_var - bg_var_global) / (bg_var_global + 1e-6)
        is_organic_grain = noise_uniformity < 0.4 and bg_var_global > 0.5
        
        anchor_correlation = 0.0
        is_handheld = env.get("shaky", False) or (bg_var_global > 2.0)
    else:
        bg_arr = np.array(bg_means)
        if len(bg_arr) > 0:
            bg_var_global = np.mean(np.var(bg_arr, axis=0))
        else:
            bg_var_global = 0.0
        # Spatial Continuity (Uniform Noise Check)
        face_var_avg = np.mean([np.var(np.array(m)) for m in roi_means.values()])
        noise_uniformity = abs(face_var_avg - bg_var_global) / (bg_var_global + 1e-6)
        is_organic_grain = noise_uniformity < 0.4 and bg_var_global > 0.5
        
        # Use nose_y from first extraction pass (no second FaceMesh needed)
        nose_y = nose_y_first_pass
        anchor_correlation = 0.0
        if len(nose_y) > 2 and len(bg_means) > 2:
            bg_intensity = np.mean(bg_arr, axis=1)
            n_y = np.array(nose_y)
            bg_i = np.interp(np.linspace(0, 1, len(n_y)), np.linspace(0, 1, len(bg_intensity)), bg_intensity)
            if np.std(n_y) > 1e-7 and np.std(bg_i) > 1e-7:
                anchor_correlation = float(np.abs(np.corrcoef(n_y, bg_i)[0, 1]))
        
        # Global Motion (Handheld detection)
        is_handheld = bg_var_global > 1.2 or anchor_correlation > 0.85
    
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

    # --- v17.1 CONTINUOUS WEIGHTED PROBABILITY SCORING ---
    # Task 2: Replace binary score snapping with continuous sigmoid mapping
    
    # 1. Sine-Wave Perfection (Biological Impossibility)
    # Higher correlation = higher suspicion. Sigmoid centered around 0.985
    sine_suspicion = 1.0 / (1.0 + np.exp(np.clip(-50 * (sine_correlation - 0.985), -200, 200)))
    
    # 2. Fixed Frequency / HRV Stability
    # Low std is bad. Centered around 0.012
    hrv_suspicion = 1.0 / (1.0 + np.exp(np.clip(300 * (ibi_std - 0.012), -200, 200)))
    
    # 3. Blue Ghosting / Channel Leakage
    blue_suspicion = 1.0 / (1.0 + np.exp(np.clip(-15 * (blue_pulse_strength - 0.6), -200, 200)))
    
    # 4. Motion-Pulse Correlation (Floating Overlay)
    if is_handheld:
        overlay_suspicion = 0.0 # Handheld invalidates overlay check
    else:
        overlay_suspicion = 1.0 / (1.0 + np.exp(np.clip(40 * (motion_pulse_corr - 0.1), -200, 200)))
        
    # 5. Spatial Inconsistency (The Patchwork Test)
    roi_thresh = 35.0 if is_handheld else (15.0 if is_organic_grain else 9.0)
    spatial_suspicion = 1.0 / (1.0 + np.exp(np.clip(-0.2 * (roi_variance - roi_thresh), -200, 200)))
    
    # 6. Robotic Stability (Legacy Check)
    drift_suspicion = 1.0 / (1.0 + np.exp(np.clip(100 * (bpm_drift - 0.04), -200, 200)))

    # Weighted sum of suspicions (Weights total 1.0)
    weights = {
        "sine": 0.25,
        "hrv": 0.20,
        "blue": 0.20,
        "overlay": 0.15,
        "spatial": 0.15,
        "drift": 0.05
    }
    
    raw_score = (
        sine_suspicion * weights["sine"] + 
        hrv_suspicion * weights["hrv"] + 
        blue_suspicion * weights["blue"] + 
        overlay_suspicion * weights["overlay"] + 
        spatial_suspicion * weights["spatial"] + 
        drift_suspicion * weights["drift"]
    )
    
    # Smooth decimal out mapping. Map [0.0, 1.0] to a wider sigmoid to stretch small changes.
    final_score = float(1.0 / (1.0 + np.exp(np.clip(-10 * (raw_score - 0.45), -200, 200))))
    tags = {
        "filtered": master_bvp.tolist() if isinstance(master_bvp, np.ndarray) else master_bvp, 
        "xf": (xf*60).tolist(), "yf": yf.tolist(), "bpm": peak_bpm, 
        "snr": snr, "drift": bpm_drift, "gr_var": avg_gr_var, 
        "gr_lock": gr_lockstep, "gr_lag": gr_phase_lag, 
        "entropy": spectral_entropy, "roi_var": roi_variance, "fps": fs,
        "rg_ratio": avg_rg_ratio, "anchor_corr": anchor_correlation
    }
    return (final_score, tags) if return_signals else final_score

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
        
        # Task 1: Standardize Preprocessing (15 FPS + Tight Crop)
        frames, actual_fps = preprocess_video(frames, actual_fps)
        
        print(f"[*] Analyzing {len(frames)} frames (@{actual_fps:.1f} FPS)...")
        score, sigs = analyze(frames, return_signals=True, source="webcam", fps=actual_fps)
        label = "DEEPFAKE" if score > 0.5 else "HUMAN"
        print(f"--- Result: {label} (Score: {score:.4f}) ---")
        plot_report(sigs, score)

def preprocess_video(frames, fps):
    """Mimic main.py optimization: 15 FPS + Tight Face Crop (224x224)"""
    target_fps = 15.0
    stride = max(1, int(fps / target_fps))
    frames = frames[::stride]
    actual_fps = fps / stride
    
    if not frames: return frames, actual_fps
    
    mp_face_detection = mp.solutions.face_detection
    face_roi = None
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frames[0].shape
            x = int(max(0, (bbox.xmin - 0.1) * w))
            y = int(max(0, (bbox.ymin - 0.1) * h))
            xw = int(min(w, (bbox.width + 0.2) * w))
            yh = int(min(h, (bbox.height + 0.2) * h))
            face_roi = (x, y, xw, yh)

    if face_roi:
        x, y, w_roi, h_roi = face_roi
        frames = [f[y:y+h_roi, x:x+w_roi] for f in frames]

    return frames, actual_fps

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
            raw_frames = []
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(frame)
                if len(raw_frames) >= 600: break
            cap.release()
            
            # Task 1: Standardize Preprocessing (15 FPS + Tight Crop)
            frames, actual_fps = preprocess_video(raw_frames, fps)
            
            # Task 4: Standardize Duration (main.py only processes first ~5 seconds)
            max_frames = int(actual_fps * 5.0)
            frames = frames[:max_frames]
            
            if len(frames) >= 75:
                score, sigs = analyze(frames, return_signals=True, source="upload", fps=actual_fps)
                label = "DEEPFAKE" if score > 0.5 else "HUMAN"
                print(f"\n--- Result: {label} (Score: {score:.4f}) ---")
                plot_report(sigs, score)
    elif choice == '2': run_webcam()
