import cv2
import mediapipe as mp
import numpy as np
import librosa
import os
import sys
from scipy.signal import butter, lfilter, savgol_filter
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def analyze(frames, audio_data, sr=22050, fps=30.0):
    """
    Lip-Sync Logic v1.7 - Forensic Precision.
    - Articulatory Lag: Detects if mouth leads audio (Human) or follows (Deepfake).
    - Absolute Threshold: >= 0.1020 is Human | < 0.1020 is Deepfake.
    """
    frame_count = len(frames)
    if frame_count == 0 or audio_data is None:
        return 0.5, {}

    # Estimate FPS (heuristic if not provided)
    if fps is None: fps = 30.0 # Standard fallback
    
    # 1. Visual Feature Extraction (Headless Stride=3)
    mp_face_mesh = mp.solutions.face_mesh
    stride_extraction = 3
    
    v_distances = [] 
    h_distances = [] 
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        for i in range(0, frame_count, stride_extraction):
            frame = frames[i]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                l = results.multi_face_landmarks[0].landmark
                v_dist = np.sqrt((l[13].x - l[14].x)**2 + (l[13].y - l[14].y)**2)
                h_dist = np.sqrt((l[78].x - l[308].x)**2 + (l[78].y - l[308].y)**2)
                scale = np.sqrt((l[10].x - l[152].x)**2 + (l[10].y - l[152].y)**2)
                v_distances.append(v_dist / (scale + 1e-6))
                h_distances.append(h_dist / (scale + 1e-6))
            else:
                v_distances.append(v_distances[-1] if v_distances else 0.0)
                h_distances.append(h_distances[-1] if h_distances else 0.0)

    # 2. Audio Feature Extraction (RMS + Pitch)
    # Re-normalize audio to match decimated visual stream
    vis_count = len(v_distances)
    hop = len(audio_data) // vis_count
    audio_rms = librosa.feature.rms(y=audio_data, hop_length=hop)[0]
    audio_rms = audio_rms[:vis_count]
    
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, hop_length=hop)
    pitch_series = []
    for t in range(min(vis_count, pitches.shape[1])):
        idx = magnitudes[:, t].argmax()
        pitch_series.append(pitches[idx, t])
    pitch_series = np.array(pitch_series)
    if len(pitch_series) < vis_count:
        pitch_series = np.pad(pitch_series, (0, vis_count - len(pitch_series)), 'edge')

    # Vocal Gating (Focus on active speech)
    vocal_gate = audio_rms > (np.max(audio_rms) * 0.1)
    
    # 3. Cross-Correlation & Lag Analysis
    def norm(sig):
        std = np.std(sig)
        return (sig - np.mean(sig)) / (std + 1e-6)

    vis_v = norm(np.array(v_distances))
    aud_v = norm(audio_rms)
    
    correlation = np.correlate(vis_v, aud_v, mode='full')
    lags = np.arange(-(len(aud_v)-1), len(vis_v))
    
    window_frames = int(0.150 * (fps/stride_extraction))
    valid_mask = (lags >= -window_frames) & (lags <= window_frames)
    
    if np.any(valid_mask):
        valid_lags = lags[valid_mask]
        valid_corr = correlation[valid_mask]
        peak_idx = np.argmax(valid_corr)
        peak_shift = valid_lags[peak_idx]
        
        if len(vis_v[vocal_gate]) > 5:
            shifted_vis = np.roll(vis_v, -peak_shift)
            max_corr = np.corrcoef(shifted_vis[vocal_gate], aud_v[vocal_gate])[0,1]
        else:
            max_corr = valid_corr[peak_idx] / (vis_count + 1e-6)
    else:
        peak_shift = 0
        max_corr = 0.0

    lag_ms = (peak_shift / (fps/stride_extraction)) * 1000.0
    
    if len(h_distances) == len(pitch_series):
        corr_h = np.corrcoef(norm(h_distances), norm(pitch_series))[0,1]
    else:
        corr_h = 0.0

    # 4. Forensic Scoring v1.7
    score = 0.0 if max_corr >= 0.1020 else 1.0

    final_score = float(np.clip(score, 0.0, 1.0))
    tags = {
        "score": final_score,
        "v_dist": np.array(v_distances).tolist(),
        "audio_amp": audio_rms.tolist(),
        "lag_ms": float(lag_ms),
        "max_corr": float(max_corr),
        "pitch_corr": float(corr_h) if 'corr_h' in locals() else 0.0
    }
    return final_score, tags

def plot_report(tags, score):
    """Generates a forensic lip-sync report."""
    try:
        plt.figure(figsize=(10, 7))
        plt.subplot(2,1,1)
        plt.plot(tags['v_dist'], label='Inner Mouth Opening', color='blue', linewidth=2)
        plt.title(f"Forensic Lip-Sync v1.7 | Lag: {tags['lag_ms']:.2f}ms")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(tags['audio_amp'], label='Audio RMS Energy', color='red', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        label = 'DEEPFAKE' if score > 0.5 else 'HUMAN'
        plt.suptitle(f"Final Score: {score:.4f} - {label}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

def run_webcam():
    import sounddevice as sd
    import wavio
    import threading
    import queue

    print("\n--- Webcam Lip-Sync Test v1.1 ---")
    print("[*] Recording will start for 10 seconds...")
    
    fs_audio = 22050
    duration = 10  # seconds
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    frames = []
    audio_data_list = []
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[*] RECORDING... SPEAK NOW!")
    
    # Start Audio Recording
    with sd.InputStream(samplerate=fs_audio, channels=1, callback=audio_callback):
        start_time = cv2.getTickCount()
        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
            ret, frame = cap.read()
            if not ret: break
            
            # Show Progress
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            cv2.putText(frame, f"RECORDING: {duration - elapsed:.1f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow("LipSync Live v1.1", frame) # Headless

            frames.append(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Pull audio from queue
            while not audio_queue.empty():
                audio_data_list.append(audio_queue.get())

    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) < 150 or not audio_data_list:
        print("Error: Capture too short or no audio.")
        return

    # Combine audio chunks
    audio_data = np.concatenate(audio_data_list, axis=0).flatten()
    
    # Calculate actual FPS from recording
    actual_fps = len(frames) / duration
    print(f"[*] Analyzing {len(frames)} frames (@{actual_fps:.1f} FPS) and {len(audio_data)} audio samples...")
    score, tags = analyze(frames, audio_data, fs_audio, fps=actual_fps)
    
    print("-" * 30)
    print(f"RESULT: {'DEEPFAKE' if score > 0.5 else 'HUMAN'}")
    print(f"Sync Score: {score:.4f} (Correlation: {tags['max_corr']:.4f})")
    print(f"Articulation Lag: {tags['lag_ms']:.2f}ms ({'Human Lead' if tags['lag_ms'] < 0 else 'AI Delay'})")
    print("-" * 30)

    # Plotting for verification
    plt.figure(figsize=(10, 7))
    plt.subplot(2,1,1)
    plt.plot(tags['v_dist'], label='Inner Mouth Opening', color='blue')
    plt.title(f"Forensic Lip-Sync v1.2 | Lag: {tags['lag_ms']:.2f}ms")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(tags['audio_amp'], label='Audio RMS Energy', color='red')
    plt.legend()
    plt.suptitle(f"Final Score: {score:.4f} - {'DEEPFAKE' if score > 0.5 else 'HUMAN'}")
    plt.tight_layout()
    # plt.show() # Headless


def run_file_upload(video_path=None):
    if not video_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        video_path = filedialog.askopenfilename()
        root.destroy()
    
    if not video_path or not os.path.exists(video_path):
        print("Invalid path.")
        return

    print(f"[*] Analyzing: {os.path.basename(video_path)}")
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        audio = clip.audio.to_soundarray(fps=22050)
        if audio.shape[1] > 1: audio = audio.mean(axis=1)
        sr = 22050
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        # Robust FPS: Frames / Duration
        fps = len(frames) / duration if duration > 0 else 30.0
        
        print(f"[*] Analyzing {len(frames)} frames (@{fps:.1f} FPS) and {len(audio)} audio samples...")
        score, tags = analyze(frames, audio, sr, fps=fps)
        
        print("-" * 30)
        print(f"RESULT: {'DEEPFAKE' if score > 0.5 else 'HUMAN'}")
        print(f"Sync Score: {score:.4f} (Correlation: {tags['max_corr']:.4f})")
        print("-" * 30)
        plot_report(tags, score)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Persona Lip-Sync Detector [v1.7]\n1. Upload | 2. Webcam")
    choice = input("Choice: ").strip()
    if choice == '1':
        run_file_upload()
    elif choice == '2':
        run_webcam()