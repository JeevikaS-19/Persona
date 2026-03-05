import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import math
import io
from collections import deque


class ReflectionSpecialist:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.audit_data = []
        self.last_eyes = [None, None]

        self.pupil_history = deque(maxlen=30)
        self.score_buffer = deque(maxlen=10)

    # -----------------------------
    # Improved glint detection
    # -----------------------------
    def get_eye_glint(self, eye_crop):

        if eye_crop is None or eye_crop.size == 0:
            return None, False

        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        min_val, max_val, _, max_loc = cv2.minMaxLoc(blurred)

        mean_val = np.mean(blurred)
        std_val = np.std(blurred)

        # adaptive highlight detection
        if max_val > mean_val + (1.5 * std_val):
            return max_loc, True

        return None, False


    # -----------------------------
    # Improved pupil metric
    # -----------------------------
    def get_pupil_metric(self, eye_crop):

        if eye_crop is None:
            return 0

        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7,7), 0)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return 0

        c = max(contours, key=cv2.contourArea)

        (x, y), radius = cv2.minEnclosingCircle(c)

        if radius < 2 or radius > 40:
            return 0

        return radius


    # -----------------------------
    # Frame analysis
    # -----------------------------
    def analyze_frame(self, frame, frame_id=0, is_webcam=False):

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results or not results.multi_face_landmarks:
            return 0.5, None, None, 0

        mesh = results.multi_face_landmarks[0].landmark

        l_center = (int(mesh[468].x * w), int(mesh[468].y * h))
        r_center = (int(mesh[473].x * w), int(mesh[473].y * h))


        def crop_eye(center):

            cx, cy = center

            if 40 < cx < w-40 and 40 < cy < h-40:
                return frame[cy-40:cy+40, cx-40:cx+40]

            return None


        l_eye = crop_eye(l_center)
        r_eye = crop_eye(r_center)

        if l_eye is not None and r_eye is not None:
            self.last_eyes = [l_eye.copy(), r_eye.copy()]


        # detect glints
        l_glint, l_light = self.get_eye_glint(l_eye)
        r_glint, r_light = self.get_eye_glint(r_eye)

        # pupil sizes
        l_pupil = self.get_pupil_metric(l_eye)
        r_pupil = self.get_pupil_metric(r_eye)

        pupil_size = np.mean([l_pupil, r_pupil])

        self.pupil_history.append(pupil_size)

        raw_score = 0.3


        # better webcam logic
        if is_webcam:

            glint_count = sum([l_light, r_light])

            if glint_count == 0:
                raw_score = 0.8

            if len(self.pupil_history) > 10:

                variance = np.var(self.pupil_history)

                if variance < 0.5:
                    raw_score = max(raw_score, 0.7)


        self.score_buffer.append(raw_score)

        final_score = np.mean(self.score_buffer)

        self.audit_data.append({
            "frame": frame_id,
            "pupil": pupil_size,
            "score": final_score
        })

        return final_score, l_eye, r_eye, pupil_size


    # -----------------------------
    # Graph generation
    # -----------------------------
    def generate_graph(self):

        df = pd.DataFrame(self.audit_data)

        plt.figure(figsize=(8,4), dpi=100)

        plt.style.use("dark_background")

        plt.plot(df.index, df["pupil"], linewidth=2)

        plt.title("Pupil Physics Variance")

        buf = io.BytesIO()

        plt.savefig(buf, format="png", bbox_inches="tight")

        buf.seek(0)

        plt.close()

        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return cv2.resize(img, (800,400))


    # -----------------------------
    # Final report
    # -----------------------------
    def show_final_report(self):

        if not self.audit_data:
            print("No audit data available")
            return

        df = pd.DataFrame(self.audit_data)

        avg_score = df["score"].mean()

        verdict = "REAL" if avg_score < 0.5 else "DEEPFAKE"

        color = (0,255,0) if verdict=="REAL" else (0,0,255)

        raw_conf = ((1-avg_score) if avg_score<0.5 else avg_score)*100

        conf = min(100, math.ceil(raw_conf/20)*20)

        report = np.zeros((1080,1920,3), dtype=np.uint8)

        cv2.putText(report,"PHYSICS FORENSIC AUDIT SYSTEM",(600,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)

        cv2.putText(report,verdict,(750,300),
                    cv2.FONT_HERSHEY_SIMPLEX,5,color,10)

        cv2.putText(report,f"CONFIDENCE: {conf}%",(780,420),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3)

        if self.last_eyes[0] is not None:

            l_zoom = cv2.resize(self.last_eyes[0], (400,400))
            r_zoom = cv2.resize(self.last_eyes[1], (400,400))

            report[500:900,1100:1500] = l_zoom
            report[500:900,1520:1920] = r_zoom

        graph = self.generate_graph()

        report[500:900,50:850] = graph

        cv2.imshow("FINAL REPORT", report)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


    # -----------------------------
    # Webcam mode
    # -----------------------------
    def analyze_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera not found")
            return

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            score,_,_,pupil = self.analyze_frame(
                frame,
                len(self.audit_data),
                True
            )

            color = (0,255,0) if score<0.5 else (0,0,255)

            cv2.putText(
                frame,
                f"Pupil Metric: {pupil:.2f}",
                (30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.imshow("Physics Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        self.face_mesh.close()

        cv2.destroyAllWindows()

        self.show_final_report()


    # -----------------------------
    # Video mode
    # -----------------------------
    def analyze_video(self, path):

        cap = cv2.VideoCapture(path)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            self.analyze_frame(frame, len(self.audit_data))

            cv2.imshow("Processing", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        self.face_mesh.close()

        cv2.destroyAllWindows()

        self.show_final_report()


    # -----------------------------
    # Image mode (FIXED)
    # -----------------------------
    def analyze_image(self, path):

        frame = cv2.imread(path)

        if frame is None:
            print("Image not found")
            return

        self.analyze_frame(frame,0)

        self.show_final_report()



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    spec = ReflectionSpecialist()

    print("1. Image | 2. Webcam | 3. Video")

    choice = input("Mode: ").strip()

    if choice == "1":

        path = input("Image path: ")

        spec.analyze_image(path)

    elif choice == "2":

        spec.analyze_webcam()

    elif choice == "3":

        path = input("Video path: ")

        spec.analyze_video(path)