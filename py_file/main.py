import os
import sys
import warnings

# Suppress MediaPipe / TensorFlow Lite internal C++ log noise (env vars must be first)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# GLOG writes directly to the stderr file-descriptor, bypassing Python env vars.
# Redirect stderr to devnull before any mediapipe/TFLite import to silence it.
_devnull = open(os.devnull, 'w')
sys.stderr = _devnull

# Suppress librosa / audioread deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import asyncio
import numpy as np
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Import our forensic specialists
from rppg_detector import analyze as analyze_rppg
from sync_detector import analyze as analyze_sync
from biometric_detector import analyze as analyze_biometric
from reflection_detector import analyze as analyze_reflection
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe_compat  # noqa
import mediapipe as mp
import pandas as pd

class EnvironmentalAnalyzer:
    """Detects organic chaos: Motion, ISO Grain, and Luminance."""
    @staticmethod
    def analyze_environment(frames):
        if not frames: return {"low_light": False, "shaky": False, "grainy": False, "lux": 100}
        
        # 1. Luminance Check
        lux_values = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames[::10]]
        avg_lux = np.mean(lux_values)
        is_low_light = avg_lux < 45 # Threshold 45/255
        
        # 2. Motion Analysis (Shaky Cam)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        nose_tips = []
        for f in frames[::5]:
            res = face_mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                l = res.multi_face_landmarks[0].landmark[1] # Nose tip
                nose_tips.append([l.x, l.y])
        
        shaky = False
        if len(nose_tips) > 10:
            variance = np.var([nt[0] for nt in nose_tips]) + np.var([nt[1] for nt in nose_tips])
            shaky = variance > 0.0012 # Tightened (0.0005 -> 0.0012) to avoid false positives on subtle head tilt
            
        # 3. Sensor Noise Check (Grain)
        # Check background Laplacian variance (outside face region)
        try:
            sample_frame = frames[len(frames)//2]
            gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
            # Use top-left corner as background sample
            bg_sample = gray[50:150, 50:150]
            laplacian_var = cv2.Laplacian(bg_sample, cv2.CV_64F).var()
            grainy = laplacian_var > 150 # High Laplacian var in static BG = Grain
        except: grainy = False
        
        return {
            "low_light": bool(is_low_light), 
            "shaky": bool(shaky), 
            "grainy": bool(grainy), 
            "lux": float(round(avg_lux, 1))
        }

# --- Pillar 4: Security & Hygiene ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # stdout only — stderr is devnull
)
logger = logging.getLogger("PersonaEngine")

# --- Pillar 3: Server Stability & Resource Guarding ---
MAX_CONCURRENT_JOBS = 1
MAX_VIDEO_DURATION = 15.0  # seconds
MAX_FILE_SIZE_MB = 100.0  # Increased for buffers
# Semaphore to limit CPU congestion (Single worker for thermal safety)
engine_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

def sanitize_filename(filename: str) -> str:
    """Strip path metadata to prevent directory traversal."""
    return Path(filename).name

async def get_video_stream(video_path, decimate=False):
    """
    Pillar 3: Memory Management (Generator).
    Pillar 2: Optimization (Selective Resolution Normalization).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video stream.")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pillar 2: Selective Decimation removed from core to protect rPPG integrity.
        # We now decimate inside the specialist calls if needed.
        if decimate and count % 2 != 0:
            count += 1
            continue
            
        # Pillar 2: Resolution Normalization (Protect small pulse signals)
        # Directive: Cap at 640x480 for thermal efficiency
        h, w = frame.shape[:2]
        if h > 480 or w > 640:
            scale = min(640 / w, 480 / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
        yield frame
        count += 1
        
    cap.release()

def resample_to_fps(frames, original_fps, target_fps=10):
    """Downsamples frames to a target frequency for CPU safety."""
    if original_fps <= target_fps:
        return frames
    
    stride = max(1, int(original_fps / target_fps))
    return frames[::stride]

async def analyze_video_production(video_path, source_type="upload", progress_callback=None, signal_callback=None):
    """
    Sentinel Production Engine:
    - 5s Trimming for memory safety
    - Independent Specialists for fault tolerance
    - 10 FPS Resampling for CPU economy
    """
    filename = sanitize_filename(video_path)
    start_time = time.time()
    temp_trim_path = None
    
    try:
        if not os.path.exists(video_path):
            return {"status": "error", "message": "Source missing", "filename": filename}
            
        async with engine_semaphore:
            logger.info(f"Sentinel Audit Initiated: {filename}")
            
            # Step 1: 5-Second Trimming (Memory Safety)
            try:
                from moviepy.video.io.VideoFileClip import VideoFileClip
                temp_trim_path = video_path.replace('.', '_trimmed.')
                with VideoFileClip(video_path) as clip:
                    end = min(5, clip.duration)
                    # moviepy v2.x renamed subclip -> subclipped
                    trimmed = clip.subclipped(0, end) if hasattr(clip, 'subclipped') else clip.subclip(0, end)
                    trimmed.write_videofile(temp_trim_path, codec="libx264", audio_codec="aac", logger=None)
                video_path = temp_trim_path
                logger.info("Trimmed to 5s for deployment stability.")
            except Exception as e:
                logger.warning(f"Trim failed, proceeding with raw: {e}")

            # Step 2: Extract Frames (Headless)
            cap = cv2.VideoCapture(video_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            raw_frames = []
            while len(raw_frames) < 150: # Cap at ~15s max even if trim failed
                ret, frame = cap.read()
                if not ret: break
                raw_frames.append(cv2.resize(frame, (640, 480)))
            cap.release()

            if not raw_frames:
                return {"status": "error", "message": "Empty stream"}

            # Step 3: Optimization Stack
            frames = resample_to_fps(raw_frames, orig_fps, target_fps=6)
            fs = 6.0
            duration = len(frames) / fs
            
            # Fast-ROI
            face_roi = None
            mp_face_detection = mp.solutions.face_detection
            # HEADLESS: MediaPipe is headless by default unless we use drawing_utils
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

            # Final ROI Crop
            if face_roi:
                x, y, w_roi, h_roi = face_roi
                frames = [f[y:y+h_roi, x:x+w_roi] for f in frames]

            # Step 4: Independent Specialists (Fault Tolerance)
            rppg_score, rppg_tags = 0.5, {}
            sync_score, sync_tags = 0.5, {}
            spatial_score = 0.5
            env = {}

            # Specialist A: rPPG
            try:
                env = EnvironmentalAnalyzer.analyze_environment(raw_frames)
                rppg_data = await asyncio.to_thread(analyze_rppg, frames, return_signals=True, fps=fs, env_flags=env)
                rppg_score, rppg_tags = rppg_data if isinstance(rppg_data, tuple) else (rppg_data, {})
            except Exception as e:
                logger.error(f"rPPG Specialist Failure: {e}")
                rppg_tags = {"error": str(e)}

            # Specialist B: Sync
            try:
                import librosa
                audio, sr = librosa.load(video_path, sr=22050, duration=5)
                if audio.size > 0 and np.max(np.abs(audio)) > 0.005:
                    sync_data = await asyncio.to_thread(analyze_sync, frames, audio, sr=sr, fps=fs)
                    sync_score, sync_tags = sync_data if isinstance(sync_data, tuple) else (sync_data, {})
            except Exception as e:
                logger.debug(f"Sync Specialist: no usable audio ({e or 'silent/no audio track'})")
                sync_tags = {"error": str(e)}

            # Specialist C: Biometric (Saccade)
            biometric_score, biometric_tags = 0.5, {}
            try:
                biometric_data = await asyncio.to_thread(analyze_biometric, frames, fps=fs, return_signals=True)
                biometric_score, biometric_tags = biometric_data if isinstance(biometric_data, tuple) else (biometric_data, {})
            except Exception as e:
                logger.error(f"Biometric Specialist Failure: {e}")
                biometric_tags = {"error": str(e)}

            # Specialist D: Reflection
            reflection_score, reflection_tags = 0.5, {}
            try:
                reflection_data = await asyncio.to_thread(analyze_reflection, frames, fps=fs, return_signals=True)
                reflection_score, reflection_tags = reflection_data if isinstance(reflection_data, tuple) else (reflection_data, {})
            except Exception as e:
                logger.error(f"Reflection Specialist Failure: {e}")
                reflection_tags = {"error": str(e)}

            # All specialists now return 0=human, 1=deepfake (suspicion scale)
            # Dynamic ensemble: skip specialists stuck at 0.5 (faulted/no data)
            # so they don't dilute the result with a meaningless neutral vote.
            candidates = [
                (rppg_score,       0.35, "error" not in rppg_tags),
                (sync_score,       0.25, "error" not in sync_tags),
                (biometric_score,  0.30, "error" not in biometric_tags),
                (reflection_score, 0.10, "error" not in reflection_tags),
            ]
            active = [(s, w) for s, w, ok in candidates if ok and s != 0.5]
            if not active:
                active = [(s, w) for s, w, _ in candidates]  # fallback: use all
            total_weight = sum(w for _, w in active)
            ensemble_score = sum(s * w for s, w in active) / total_weight
            
            # Suspicion threshold: >= 0.65 → DEEPFAKE
            classification = "DEEPFAKE" if ensemble_score >= 0.65 else "HUMAN"
            
            # Majority-vote override: if >2 of 4 pillars are >= 0.5, force DEEPFAKE
            all_scores = [rppg_score, sync_score, biometric_score, reflection_score]
            suspicious_count = sum(1 for s in all_scores if s >= 0.5)
            if suspicious_count > 2:
                classification = "DEEPFAKE"

            
            return {
                "status": "completed",
                "filename": filename,
                "metrics": {
                    "ensemble_score": round(float(ensemble_score), 4),
                    "rppg_score": round(float(rppg_score), 4),
                    "sync_score": round(float(sync_score), 4),
                    "biometric_score": round(float(biometric_score), 4),
                    "reflection_score": round(float(reflection_score), 4),
                    "classification": classification
                },
                "environment": env,
                "forensics": {
                    "bpm": rppg_tags.get("bpm", 0),
                    "jitter": biometric_tags.get("jitter_avg"),
                    "morphology_r2": reflection_tags.get("morphology_r2", 0.0),
                    "parallax_fails": reflection_tags.get("parallax_fails", 0),
                    "blink_fails": reflection_tags.get("blink_persistence", 0),
                    "rppg_fault": "error" in rppg_tags,
                    "sync_fault": "error" in sync_tags,
                    "biometric_fault": "error" in biometric_tags
                },
                "telemetry": {
                    "compute_time": round(time.time() - start_time, 2),
                    "resampled_fps": fs,
                    "frames_analyzed": len(frames),
                    "trimmed": True
                }
            }
            
    except Exception as e:
        logger.error(f"SENTINEL_PANIC: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        if temp_trim_path and os.path.exists(temp_trim_path):
            try: os.unlink(temp_trim_path)
            except: pass

async def run_cli_audit(source_type="webcam", file_path=None):
    """CLI entry point for a full Persona forensic audit."""
    print(f"\n--- Persona Sentinel Audit v2.5 [{source_type.upper()}] ---")
    
    frames = []
    fps = 30.0
    
    if source_type == "webcam":
        cap = cv2.VideoCapture(0)
        print("[*] Monitoring Webcam... (Press 'Q' to analyze)")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, "PERSONA AUDIT ACTIVE", (20, 40), 1, 1, (0, 0, 255), 2)
            cv2.imshow("Persona Engine CLI", frame)
            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or len(frames) >= 300: break
        cap.release()
        cv2.destroyAllWindows()
    else:
        if not file_path or not os.path.exists(file_path):
            print("[!] Invalid File Path."); return
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[*] Ingesting: {os.path.basename(file_path)} (@{fps:.1f} FPS)")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            if len(frames) >= 600: break
        cap.release()

    if len(frames) < 75:
        print("[!] Data stream too short for specialized forensic analysis.")
        return

    print("[*] Running Ensemble Specialists (rPPG, Sync, Bio, Physics)...")
    # For CLI, we save as a temp file to allow Specialists to use their native file loaders (like librosa for Sync)
    temp_path = "cli_audit_temp.mp4"
    try:
        # Create a basic video for specialists that need file access
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames: out.write(f)
        out.release()
        
        result = await analyze_video_production(temp_path, source_type=source_type)
        
        if result["status"] == "completed":
            m   = result["metrics"]
            env = result.get("environment", {})
            fos = result.get("forensics", {})
            tel = result.get("telemetry", {})

            verdict = m["classification"]
            score   = m["ensemble_score"]
            bar_len = 40
            filled  = int(score * bar_len)
            bar     = "█" * filled + "░" * (bar_len - filled)

            print("\n" + "╔" + "═"*58 + "╗")
            print(f"║  {'PERSONA SENTINEL — FORENSIC AUDIT REPORT':^54}  ║")
            print("╠" + "═"*58 + "╣")
            print(f"║  File : {result.get('filename','unknown'):<48}  ║")
            print(f"║  Time : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S'):<48}  ║")
            print("╠" + "═"*58 + "╣")

            # Verdict banner
            flag = "🔴  DEEPFAKE DETECTED" if verdict == "DEEPFAKE" else "🟢  GENUINE / HUMAN"
            print(f"║  VERDICT  :  {flag:<44}  ║")
            print(f"║  Ensemble :  [{bar}] {score:.2%}  ║")
            print("╠" + "═"*58 + "╣")

            # Specialist scores
            print("║  SPECIALIST BREAKDOWN                                    ║")
            print("╠" + "─"*58 + "╣")
            specialists = [
                ("rPPG  (Heart-Rate)",    m["rppg_score"]),
                ("Sync  (Lip-Sync)",      m["sync_score"]),
                ("Bio   (Eye-Jitter)",    m["biometric_score"]),
                ("Refl  (Corneal Phys.)", m["reflection_score"]),
            ]
            for name, s in specialists:
                mini_bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
                flag_sp  = " ⚠" if s >= 0.65 else "  "
                print(f"║  {name:<22} [{mini_bar}] {s:.4f}{flag_sp}  ║")

            # Environment
            print("╠" + "═"*58 + "╣")
            print("║  ENVIRONMENT                                             ║")
            print("╠" + "─"*58 + "╣")
            env_items = [
                ("Avg Luminance (lux)", f"{env.get('lux', 'N/A')}"),
                ("Low-Light Flag",      "YES ⚠" if env.get("low_light") else "No"),
                ("Shaky Camera",        "YES ⚠" if env.get("shaky")     else "No"),
                ("Film Grain Detected", "YES ⚠" if env.get("grainy")    else "No"),
            ]
            for k, v in env_items:
                print(f"║  {k:<28}  {v:<26}  ║")

            # Forensic signals
            print("╠" + "═"*58 + "╣")
            print("║  FORENSIC SIGNALS                                        ║")
            print("╠" + "─"*58 + "╣")
            jitter = fos.get("jitter", None)
            fos_items = [
                ("Eye-Jitter Avg (px)", f"{jitter:.4f}" if jitter is not None else "N/A"),
                ("Morphology Fit (R2)", f"{fos.get('morphology_r2', 0):.2%}"),
                ("Zero Parallax Frames", f"{fos.get('parallax_fails', 0)}"),
                ("Baked Blink Fails", f"{fos.get('blink_fails', 0)}"),
                ("rPPG Specialist Fault", "YES ⚠" if fos.get("rppg_fault")      else "No"),
                ("Sync Specialist Fault", "YES ⚠" if fos.get("sync_fault")      else "No"),
                ("Bio  Specialist Fault", "YES ⚠" if fos.get("biometric_fault") else "No"),
            ]
            for k, v in fos_items:
                print(f"║  {k:<28}  {v:<26}  ║")
                
            # Telemetry
            print("╠" + "═"*58 + "╣")
            print("║  TELEMETRY                                               ║")
            print("╠" + "─"*58 + "╣")
            tel_items = [
                ("Compute Time (s)",   f"{tel.get('compute_time', '?')}"),
                ("Frames Analysed",    f"{tel.get('frames_analyzed', '?')}"),
                ("Resampled FPS",      f"{tel.get('resampled_fps', '?')}"),
                ("Video Trimmed",      "Yes (5s)" if tel.get("trimmed") else "No"),
            ]
            for k, v in tel_items:
                print(f"║  {k:<28}  {v:<26}  ║")

            print("╚" + "═"*58 + "╝\n")
        else:
            print(f"[!] Engine Error: {result.get('message')}")
            
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    print("Persona Forensic Engine [v2.5]\n1. Upload | 2. Webcam")
    choice = input("Select: ").strip()
    if choice == '1':
        import tkinter as tk; from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(); root.destroy()
        if path: asyncio.run(run_cli_audit("upload", path))
    elif choice == '2':
        asyncio.run(run_cli_audit("webcam"))
