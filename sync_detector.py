import cv2
import mediapipe as mp
import numpy as np
import librosa
from scipy.signal import butter, lfilter, savgol_filter
from moviepy.video.io.VideoFileClip import VideoFileClip


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def analyze(frames, audio_data, sr=22050):
    # 1. Audio Pre-processing (Vocal Isolation: 300Hz - 3400Hz)
    filtered_audio = bandpass_filter(audio_data, 300, 3400, sr)
    
    # 2. Extract Visual Signal (Lip Distance)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    lip_distances = []
    for frame in frames:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            l = results.multi_face_landmarks[0].landmark
            # Euclidean distance for 3D depth resilience
            dist = np.sqrt((l[13].x - l[14].x)**2 + (l[13].y - l[14].y)**2 + (l[13].z - l[14].z)**2)
            lip_distances.append(dist)
        else:
            lip_distances.append(lip_distances[-1] if lip_distances else 0.0)

    # 3. Apply Savitzky-Golay Smoothing to Lip Data
    # Window size must be odd and less than the number of frames
    if len(lip_distances) > 11:
        lip_distances = savgol_filter(lip_distances, window_length=11, polyorder=3)
    
    # 4. Extract Acoustic Signal (Energy Envelope)
    hop = len(filtered_audio) // len(frames)
    audio_env = librosa.onset.onset_strength(y=filtered_audio, sr=sr, hop_length=hop)
    audio_env = audio_env[:len(lip_distances)]

    # 5. Normalization
    def norm(sig):
        std = np.std(sig)
        return (sig - np.mean(sig)) / std if std > 1e-6 else sig - np.mean(sig)

    vis_sig = norm(np.array(lip_distances))
    aud_sig = norm(audio_env)

    # 6. Cross-Correlation (Check for time-aligned patterns)
    # Using 'valid' mode focuses on the most overlapping section
    correlation = np.correlate(vis_sig, aud_sig, mode='same')
    max_corr = np.max(correlation) / len(vis_sig) 

    # 7. Final Scoring Logic
    # 0.0 (Human) to 1.0 (Deepfake)
    # Most real speech hits a normalized correlation between 0.3 and 0.6
    score = 1.0 - np.clip(max_corr * 2.5, 0.0, 1.0) 

    return float(np.round(score, 4))
    return score
    
def run_detection(video_path):
    # 1. Load Audio using MoviePy (No FFmpeg installation required!)
    print(f"[*] Extracting audio from {video_path}...")
    try:
        video_clip = VideoFileClip(video_path)
        # Convert to a format librosa-style logic can understand (Mono, 22050Hz)
        audio_data = video_clip.audio.to_soundarray(fps=22050)
        
        # If stereo (2 channels), average them to make it mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        sr = 22050
        video_clip.close() # Free up the file
    except Exception as e:
        print(f"Audio Error: {e}")
        return

    # 2. Load Video Frames (OpenCV)
    print(f"[*] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Error: No frames found. Check your video path.")
        return

    # 3. Run the Specialist
    print(f"[*] Analyzing {len(frames)} frames...")
    final_score = analyze(frames, audio_data, sr)
    
    print("-" * 30)
    print(f"RESULT FOR: {video_path}")
    print(f"Deepfake Score: {final_score}")
    print("Interpretation:", "Deepfake" if final_score > 0.5 else "Human")
    print("-" * 30)

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace 'my_video.mp4' with the name of a video in your folder
    run_detection(r"C:/Users/disha/Videos/Recording 2026-03-04 134122.mp4")