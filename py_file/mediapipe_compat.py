"""
mediapipe_compat.py  — Compatibility shim for mediapipe >= 0.10
---------------------------------------------------------------
mediapipe 0.10 removed `mp.solutions`. This module re-attaches a
`solutions` namespace to the mediapipe package so legacy code that
calls `mp.solutions.face_mesh.FaceMesh(...)` keeps working unchanged.

Usage (call ONCE before any detector imports):
    import mediapipe_compat  # noqa — registers shim
"""

import types
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import cv2
import urllib.request, os, sys


# ── helpers ──────────────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def _download_model():
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    if not os.path.exists(_MODEL_PATH):
        print("[COMPAT] Downloading face_landmarker.task…")
        urllib.request.urlretrieve(url, _MODEL_PATH)
        print("[COMPAT] Downloaded face_landmarker.task")


# ── FaceMesh compatibility wrapper ───────────────────────────────────────────

class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

class _FakeFaceLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks          # list of _FakeLandmark

class _FaceMeshResult:
    def __init__(self, face_landmarks_list):
        if face_landmarks_list:
            self.multi_face_landmarks = [
                _FakeFaceLandmarks([_FakeLandmark(lm.x, lm.y, lm.z)
                                    for lm in fl])
                for fl in face_landmarks_list
            ]
        else:
            self.multi_face_landmarks = None

class FaceMesh:
    """Drop-in replacement for mp.solutions.face_mesh.FaceMesh"""

    def __init__(self, static_image_mode=False, max_num_faces=1,
                 refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        _download_model()
        base_opts = _mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        running_mode = (
            _mp_vision.RunningMode.IMAGE
            if static_image_mode
            else _mp_vision.RunningMode.VIDEO
        )
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
        self._static = static_image_mode
        self._ts = 0   # monotonic timestamp (ms) for VIDEO mode

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static:
            detection = self._landmarker.detect(mp_image)
        else:
            self._ts += 33          # ~30 fps assumed; good enough for analysis
            detection = self._landmarker.detect_for_video(mp_image, self._ts)
        return _FaceMeshResult(detection.face_landmarks)

    def close(self):
        try: self._landmarker.close()
        except: pass

    # context-manager support (with FaceMesh(...) as mesh:)
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ── Assemble the fake solutions namespace ─────────────────────────────────────

_face_mesh_module = types.SimpleNamespace(FaceMesh=FaceMesh)
_solutions_ns = types.SimpleNamespace(face_mesh=_face_mesh_module)

# Attach to the live mediapipe module object so `import mediapipe as mp;
# mp.solutions.face_mesh` works everywhere
mp.solutions = _solutions_ns

print("[COMPAT] mediapipe.solutions shim installed (face_mesh -> FaceLandmarker)")
