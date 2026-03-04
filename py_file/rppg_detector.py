import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def analyze(frames, return_signals=False, source="auto", fps=30.0):
    """
    Forensic rPPG Analysis v13.0 - Entropy Forensic.
    - Phase Jitter (SD_Phase): Detects incoherent AI patch generation.
    - Spectral Shape: Distinguishes Bio-Shoulders from AI Delta peaks.
    - Ultra-Lockstep: Detects subtle R/G synthetic unison.
    """
    frame_count = len(frames)
    if not frames or frame_count < 150:
        return (0.5, None) if return_signals else 0.5
    
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

    # Windowed Phase Stability Check
    win_len, stride = 150, 30
    phase_lags = []
    for i in range(0, frame_count - win_len, stride):
        # Phase of forehead vs mean cheeks
        f_win = roi_data["forehead"]["bvp"][i:i+win_len]
        c_win = (roi_data["cheek_r"]["bvp"][i:i+win_len] + roi_data["cheek_l"]["bvp"][i:i+win_len])/2
        
        f_fft = rfft(f_win)
        c_fft = rfft(c_win)
        xf_w = rfftfreq(win_len, 1/fs)
        idx = np.argmax(np.abs(f_fft)[(xf_w >= 0.75) & (xf_w <= 2.5)])
        
        f_phase = np.angle(f_fft[idx])
        c_phase = np.angle(c_fft[idx])
        phase_lags.append(np.abs(f_phase - c_phase))
    
    phase_jitter = np.std(phase_lags) if len(phase_lags) > 1 else 0.0

    master_bvp = (roi_data["forehead"]["bvp"] + roi_data["cheek_r"]["bvp"] + roi_data["cheek_l"]["bvp"]) / 3.0
    avg_gr_var = np.mean([v["gr_var"] for v in roi_data.values()])
    avg_lockstep = np.mean([v["lockstep"] for v in roi_data.values()])
    
    yf = np.abs(rfft(master_bvp))
    xf = rfftfreq(len(master_bvp), 1/fs)
    band_mask = (xf >= 0.75) & (xf <= 2.5)
    peak_idx_rel = np.argmax(yf[band_mask])
    peak_val = yf[band_mask][peak_idx_rel]
    peak_bpm = xf[band_mask][peak_idx_rel] * 60
    
    # v12.0 - Spectral Shape check (Bio-Shoulder)
    # Binary spectrum check: AI peaks have near-zero energy in adjacent bins.
    peak_idx_abs = np.where(band_mask)[0][peak_idx_rel]
    adj_energy = (yf[peak_idx_abs-1] + yf[peak_idx_abs+1]) / (peak_val + 1e-6)
    
    snr = peak_val / (np.mean(yf[band_mask]) + 1e-6)
    
    bpms = []
    for i in range(0, len(master_bvp) - win_len, stride):
        win = master_bvp[i:i+win_len]
        y_w = np.abs(rfft(win))
        x_w = rfftfreq(win_len, 1/fs)
        m_w = (x_w >= 0.75) & (x_w <= 2.5)
        bpms.append(x_w[m_w][np.argmax(y_w[m_w])] * 60)
    bpm_drift = np.std(bpms) if len(bpms) > 1 else 0.0

    sync_score = np.corrcoef(roi_data["forehead"]["bvp"], (roi_data["cheek_r"]["bvp"]+roi_data["cheek_l"]["bvp"])/2)[0,1]

    # --- v13.1 SCORING MATRIX (Dual Profile) ---
    score = 0.5
    ai_hits = 0
    
    # 1. PROFILE SELECTION
    is_live = (source == "webcam")
    
    # 2. SPECTRAL BROADENING (The "Human" Flag)
    # AI Delta Peak: adj_energy < 0.1 (Single surgical bin)
    # Human Shoulder: adj_energy > 0.2 (Natural frequency bleed)
    is_broad = (adj_energy > 0.15)
    
    # 3. ROBOTIC STABILITY ANCHOR
    if bpm_drift < 0.1:
        score += 0.45; ai_hits += 2
    elif bpm_drift < 0.25:
        score += 0.25
        
    # 4. INTENSITY GATING
    if is_live:
        # Profile A: Live (Strict)
        if snr < 1.8: score += 0.35; ai_hits += 1 # Low quality is suspicious in live stream
    else:
        # Profile B: Upload (Relaxed/Compressed)
        if snr < 1.1 and not is_broad: score += 0.4; ai_hits += 1 # Only penalize low SNR if it's NOT broad
        
    # 5. BIOLOGICAL SIGNATURES (G/R Ratio)
    if avg_gr_var < 1.1:
        score += 0.35; ai_hits += 1
    elif avg_gr_var < 1.02:
        score += 0.5; ai_hits += 2

    # --- BIOLOGICAL REWARDS (Adaptive) ---
    reward_multiplier = 0.0 if ai_hits >= 2 else (0.5 if ai_hits == 1 else 1.0)
    
    # Spectral Broadening Reward (Elite Human Marker)
    if is_broad and avg_gr_var > 1.2:
        score -= 0.4 * reward_multiplier # Strong proof of biological source
        
    # Natural Drift Reward
    if 0.4 < bpm_drift < 6.0:
        score -= 0.25 * reward_multiplier
        
    # Phase Jitter Penalties (Patchwork AI)
    if phase_jitter > 1.4:
        score += 0.4; ai_hits += 1

    final_score = float(np.clip(score, 0.0, 1.0))
    tags = {"filtered": master_bvp, "xf": xf*60, "yf": yf, "bpm": peak_bpm, "snr": snr, "drift": bpm_drift, "gr_var": avg_gr_var, "sync": sync_score, "jitter": phase_jitter, "adj_e": adj_energy, "fps": fs, "source": source}
    return final_score, tags if return_signals else final_score

def plot_report(signals, score):
    if not signals: return
    try:
        plt.figure(figsize=(10, 8))
        # CORRECTED v12.0 LABEL LOGIC
        label = "INCONCLUSIVE" if (signals["snr"] < 1.3) else ("DEEPFAKE" if score > 0.5 else "HUMAN")
        
        plt.subplot(3, 1, 1); plt.plot(signals["filtered"], color='red')
        plt.title(f"v13.0 Entropy Pulse (FPS: {signals.get('fps',0):.1f} | Jitter: {signals.get('jitter',0):.2f})")
        plt.subplot(3, 1, 2); plt.plot(signals["xf"], signals["yf"], color='blue')
        plt.axvline(x=signals["bpm"], color='orange', linestyle='--'); plt.xlim(40, 160)
        plt.title(f"FFT Spectrum (BPM: {signals['bpm']:.1f} | SNR: {signals['snr']:.2f})")
        plt.subplot(3, 1, 3); plt.bar(["Bio-Var", "Drift", "Jitter", "Adj-E"], [signals["gr_var"], signals["drift"], signals["jitter"], signals["adj_e"]], color=['green', 'orange', 'purple', 'black'])
        plt.axhline(y=1.3, color='gray', linestyle='--'); plt.axhline(y=1.2, color='red', linestyle=':')
        plt.suptitle(f"Persona Entropy Forensic [v13.0]\nResult: {label} (Score: {score:.4f})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    except: pass

def run_webcam():
    cap = cv2.VideoCapture(0)
    print("\n--- Live Capture v13.0 ---")
    frames = []
    while len(frames) < 300:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Persona Live v12.0", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frames.append(frame)
    cap.release(); cv2.destroyAllWindows()
    if len(frames) >= 150:
        score, sigs = analyze(frames, return_signals=True, source="webcam")
        label = "DEEPFAKE" if score > 0.5 else "HUMAN"
        print(f"--- Result: {label} (Score: {score:.4f}) ---")
        plot_report(sigs, score)

if __name__ == "__main__":
    print("Persona rPPG Forensic [v13.0]\n1. Upload | 2. Webcam")
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
