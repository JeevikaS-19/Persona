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

async def analyze_video_production(video_path):
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
            
            # Pillar 1: Asynchronous Parallelism (launch both simultaneously)
            # We run the heavy compute in threads to avoid blocking the event loop
            # Pillar 1 & 3: Audio Silence Handling
            is_silent = np.max(np.abs(audio)) < 0.005
            
            # Pillar 1 & 2: Selective Decimation
            # rPPG gets FULL frames for frequency stability.
            # Sync gets DECIMATED frames for performance.
            fps_full = len(frames) / duration
            fps_decimated = fps_full / 2.0
            
            rppg_task = asyncio.to_thread(analyze_rppg, frames, return_signals=True, fps=fps_full)
            
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

            # Weighted Ensembling (Adaptive Weighting)
            if is_silent:
                ensemble_score = rppg_score # 100% rPPG for silent videos
                audit_type = "AUDIO-AGNOSTIC BIOMETRIC AUDIT"
            else:
                ensemble_score = (rppg_score * 0.6) + (sync_score * 0.4)
                audit_type = "FULL MULTI-MODAL FORENSIC AUDIT"
            
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
                    "classification": "DEEPFAKE" if ensemble_score > 0.5 else "HUMAN"
                },
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
    print("PERSONA PRODUCTION ENGINE v1.2")
    print("Asynchronous | Hybrid-FPS | Bio-Fidelity")
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
