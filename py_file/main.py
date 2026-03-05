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

async def analyze_video_production(video_path, progress_callback=None, signal_callback=None):
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
                    # Immediately trim to first 5s to prevent laptop overflow
                    trimmed = clip.subclip(0, min(5, clip.duration))
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
            frames = resample_to_fps(raw_frames, orig_fps, target_fps=10)
            fs = 10.0
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
                logger.error(f"Sync Specialist Failure: {e}")
                sync_tags = {"error": str(e)}

            # Specialist C: Keyframe Spatial Artifacts (Optimization Directive)
            try:
                import random
                key_indices = random.sample(range(len(frames)), min(3, len(frames)))
                key_scores = []
                for idx in key_indices:
                    # Simple Laplacian variance check as a proxy for 'blurry/AI soft' artifacts
                    # Synthetic faces often have inconsistent edge sharpness
                    var = cv2.Laplacian(frames[idx], cv2.CV_64F).var()
                    key_scores.append(1.0 if var < 100 else 0.2) # Low var = potential AI smoothness
                spatial_score = np.mean(key_scores)
                logger.info(f"Keyframe Strategy: Spatial audit (3 frames) complete. Score: {spatial_score}")
            except Exception as e:
                logger.error(f"Spatial Specialist Failure: {e}")

            # Final Summary (Ensemble weighted)
            ensemble_score = (rppg_score * 0.4) + (sync_score * 0.4) + (spatial_score * 0.2)
            classification = "DEEPFAKE" if ensemble_score > 0.5 else "HUMAN"
            
            return {
                "status": "completed",
                "filename": filename,
                "metrics": {
                    "ensemble_score": round(float(ensemble_score), 4),
                    "rppg_score": round(float(rppg_score), 4),
                    "sync_score": round(float(sync_score), 4),
                    "spatial_score": round(float(spatial_score), 4),
                    "classification": classification
                },
                "environment": env,
                "forensics": {
                    "bpm": rppg_tags.get("bpm", 0),
                    "filtered": rppg_tags.get("filtered", []),
                    "audio_amp": sync_tags.get("audio_amp", []),
                    "v_dist": sync_tags.get("v_dist", []),
                    "rppg_fault": "error" in rppg_tags,
                    "sync_fault": "error" in sync_tags
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

# Headless entry point
if __name__ == "__main__":
    print("PERSONA ENGINE: Headless Mode Active")
