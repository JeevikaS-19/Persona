import cv2
import asyncio
import numpy as np
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Import our forensic specialists
from rppg_detector import analyze as analyze_rppg
from sync_detector import analyze as analyze_sync
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe as mp

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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PersonaEngine")

# --- Pillar 3: Server Stability & Resource Guarding ---
MAX_CONCURRENT_JOBS = 3
MAX_VIDEO_DURATION = 15.0  # seconds
MAX_FILE_SIZE_MB = 50.0
# Semaphore to limit CPU congestion
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
        h, w = frame.shape[:2]
        if h > 720 or w > 1280:
            scale = 720 / h
            frame = cv2.resize(frame, (int(w * scale), 720))
            
        yield frame
        count += 1
        
    cap.release()

async def analyze_video_production(video_path, progress_callback=None, signal_callback=None):
    """Orchestrates the parallel forensic analysis."""
    
    # Pillar 4: Filename Sanitization
    filename = sanitize_filename(video_path)
    start_time = time.time()
    
    # Pillar 3: Resource Guarding (Size/Duration)
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_MB:
        return {"error": f"File too large: {file_size:.1f}MB (Max {MAX_FILE_SIZE_MB}MB)", "filename": filename}
    
    async with engine_semaphore:
        logger.info(f"Starting analysis for: {filename}")
        
        try:
            # Pillar 3: Memory Efficient Processing
            frames = []
            async for frame in get_video_stream(video_path):
                frames.append(frame)
                
            # Pillar 4: Duration Limit
            # (Heuristic: frames / estimated fps)
            clip = VideoFileClip(video_path)
            duration = clip.duration
            if duration > MAX_VIDEO_DURATION:
                return {"error": f"Video too long: {duration:.1f}s (Max {MAX_VIDEO_DURATION}s)"}
            
            # Extract Audio for Sync Detector
            audio = clip.audio.to_soundarray(fps=22050)
            if audio.shape[1] > 1: audio = audio.mean(axis=1) # Mono
            sr = 22050
            
            # ENVIRONMENTAL CALIBRATION (NEW)
            env = EnvironmentalAnalyzer.analyze_environment(frames)
            logger.info(f"Environment: {env}")
            
            # Pillar 1: Asynchronous Parallelism (launch both simultaneously)
            # We run the heavy compute in threads to avoid blocking the event loop
            # Pillar 1 & 3: Audio Silence Handling
            is_silent = np.max(np.abs(audio)) < 0.005
            
            # Pillar 1 & 2: Selective Decimation
            # rPPG gets FULL frames for frequency stability.
            # Sync gets DECIMATED frames for performance.
            fps_full = len(frames) / duration
            fps_decimated = fps_full / 2.0
            
            rppg_task = asyncio.to_thread(analyze_rppg, frames, return_signals=True, fps=fps_full, env_flags=env, callback=signal_callback)
            
            # Only run sync if there's audio energy (use decimated slice)
            if not is_silent:
                sync_frames = frames[::2]
                sync_task = asyncio.to_thread(analyze_sync, sync_frames, audio, sr=sr, fps=fps_decimated)
                rppg_res, sync_res = await asyncio.gather(rppg_task, sync_task)
            else:
                rppg_res = await rppg_task
                sync_res = (0.5, {"max_corr": 0, "lag_ms": 0})
            
            rppg_score, rppg_tags = rppg_res if isinstance(rppg_res, tuple) else (rppg_res, {})
            sync_score, sync_tags = sync_res if isinstance(sync_res, tuple) else (sync_res, {})
            
            rppg_tags = rppg_tags or {}
            sync_tags = sync_tags or {}

            # v17.0 REAL-WORLD HARDENING
            is_consensus_fake = (rppg_score > 0.85 and sync_score > 0.85)
            
            # Extract new v17.0 tags
            rgv = rppg_tags.get("rg_ratio", 1.0)
            anc = rppg_tags.get("anchor_corr", 0.0)
            
            # Update environment flags based on specialized rPPG analysis
            if anc > 0.85: env["shaky"] = True
            if rgv > 2.0: env["grainy"] = True # Use grainy/organic as proxy for red-dominance

            # Adaptive Weighting (Organic Chaos)
            if env["low_light"]:
                ensemble_score = (rppg_score * 0.2) + (sync_score * 0.8)
                audit_type = "LOW-LIGHT ADAPTIVE AUDIT"
            elif env["shaky"] or env["grainy"] or rgv > 1.8:
                # Handheld/Grainy/Orange real videos
                # Directive: 60% rPPG, 40% Lip-Sync if R/G > 2.0
                ensemble_score = (rppg_score * 0.6) + (sync_score * 0.4)
                # Apply "Chaos Discount" only if no consensus deepfake is detected
                if (env["shaky"] or anc > 0.85) and not is_consensus_fake: 
                    ensemble_score -= 0.4
                audit_type = "ORGANIC CHAOS AUDIT"
            elif is_silent:
                ensemble_score = rppg_score 
                audit_type = "AUDIO-AGNOSTIC BIOMETRIC AUDIT"
            else:
                ensemble_score = (rppg_score * 0.6) + (sync_score * 0.4)
                audit_type = "FULL MULTI-MODAL FORENSIC AUDIT"
                
            # Final threshold relaxation if ORGANIC
            ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))
            classification = "DEEPFAKE" if ensemble_score > 0.5 else "HUMAN"
            
            # REFINED RELAXATION: Only relax if there's no consensus fake and score is borderline
            if (env["shaky"] or env["grainy"]) and ensemble_score < 0.65 and not is_consensus_fake:
                classification = "HUMAN" # Narrowed window (0.75 -> 0.65)
            
            execution_time = time.time() - start_time
            
            # Pillar 4: Structured JSON Logging
            result = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "status": "COMPLETED",
                "audit_mode": audit_type,
                "metrics": {
                    "ensemble_score": round(ensemble_score, 4),
                    "rppg_score": round(rppg_score, 4),
                    "sync_score": round(sync_score, 4),
                    "classification": classification
                },
                "environment": env,
                "forensics": {
                    "bpm": round(rppg_tags.get("bpm", 0) if rppg_tags else 0, 1),
                    "snr": round(rppg_tags.get("snr", 0) if rppg_tags else 0, 2),
                    "entropy": round(rppg_tags.get("entropy", 0) if rppg_tags else 0, 2),
                    "gr_lock": round(rppg_tags.get("gr_lock", 0) if rppg_tags else 0, 3),
                    "roi_var": round(rppg_tags.get("roi_var", 0) if rppg_tags else 0, 2),
                    "sync_lag_ms": round(sync_tags.get("lag_ms", 0) if sync_tags else 0, 2),
                    "sync_corr": round(sync_tags.get("max_corr", 0) if sync_tags else 0, 4)
                },
                "server_telemetry": {
                    "frames_processed": len(frames),
                    "duration_s": round(duration, 2),
                    "compute_time_s": round(execution_time, 2)
                }
            }
            logger.info(f"Analysis Finished: {filename} -> {result['metrics']['classification']}")
            return result
            
        except Exception as e:
            logger.error(f"Engine Failure on {filename}: {str(e)}")
            return {"error": str(e), "filename": filename}

async def demo_ui():
    """Simple UI wrapper for the production engine."""
    import tkinter as tk
    from tkinter import filedialog
    
    print("-" * 50)
    print("PERSONA PRODUCTION ENGINE v2.0")
    print("Asynchronous | Organic Chaos | Bio-Aware")
    print("-" * 50)
    
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(); root.destroy()
    
    if path:
        result = await analyze_video_production(path)
        
        # ERROR HANDLING
        if "error" in result:
            print("\n" + "!"*50)
            print(f" ERROR: {result.get('filename', 'Unknown File')}")
            print(f" MESSAGE: {result['error']}")
            print("!"*50 + "\n")
            return

        # TERMINAL REPORT POP-UP
        m = result.get("metrics", {})
        f = result.get("forensics", {})
        cls = m.get("classification", "ERROR")
        color = "\033[91m" if cls == "DEEPFAKE" else "\033[92m"
        reset = "\033[0m"
        
        print("\n" + "="*50)
        print(f" FORENSIC AUDIT REPORT: {result['filename']}")
        print(f" MODE: {result.get('audit_mode', 'UNKNOWN')}")
        env = result.get("environment", {})
        env_str = f"[{'LOW LIGHT' if env.get('low_light') else 'OPTIMAL'}] [{'SHAKY' if env.get('shaky') else 'STATIC'}] [{'GRAINY' if env.get('grainy') else 'SMOOTH'}]"
        print(f" ENV: {env_str}")
        print("="*50)
        
        print(f" VERDICT: {color}{cls}{reset} (Score: {m.get('ensemble_score', 0):.4f})")
        print("-" * 50)
        print(f" BIOMETRIC (rPPG): {m.get('rppg_score', 0):.2f} (BPM: {f.get('bpm', 0)})")
        print(f" BEHAVIORAL (Sync): {m.get('sync_score', 0):.2f} (Corr: {f.get('sync_corr', 0):.3f})")
        print("-" * 50)
        print(f" TELEMETRY: {result.get('server_telemetry', {}).get('compute_time_s', 0)}s compute")
        print("="*50)

        print("\n[FULL JSON LOG]")
        print(json.dumps(result, indent=4))

if __name__ == "__main__":
    asyncio.run(demo_ui())
