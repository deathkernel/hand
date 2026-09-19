import os
import time
import urllib.request
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

HAND_CONNECTIONS = (
    (0, 1), (0, 5), (0, 17), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def ensure_model() -> None:
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe hand model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def count_fingers(landmarks, handedness: str) -> int:
    fingers = 0
    if handedness == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers += 1

    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if landmarks[tip].y < landmarks[pip].y:
            fingers += 1
    return fingers


def gesture_name(count: int) -> str:
    return {
        0: "FIST",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "OPEN HAND",
    }.get(count, "UNKNOWN")


def draw_hand(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    points: List[Tuple[int, int]] = []
    for landmark in landmarks:
        x = max(0, min(w - 1, int(landmark.x * w)))
        y = max(0, min(h - 1, int(landmark.y * h)))
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (80, 220, 120), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 220, 255), -1)


def draw_panel(frame, hands_info: List[str], fps: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 140), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.putText(frame, "HAND DETECTION", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    y = 88
    if not hands_info:
        cv2.putText(frame, "No hand detected", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1)
    else:
        for info in hands_info[:2]:
            cv2.putText(frame, info, (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 1)
            y += 22


def main() -> None:
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = time.perf_counter()
    timestamp_ms = 0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                infos: List[str] = []

                for index, landmarks in enumerate(result.hand_landmarks):
                    handedness = "Unknown"
                    confidence = 0.0
                    if index < len(result.handedness) and result.handedness[index]:
                        category = result.handedness[index][0]
                        handedness = category.category_name or category.display_name or "Unknown"
                        confidence = category.score

                    count = count_fingers(landmarks, handedness)
                    draw_hand(frame, landmarks)
                    gesture = gesture_name(count)

                    wrist = landmarks[0]
                    h, w = frame.shape[:2]
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, f"{handedness}: {gesture}",
                                (max(10, x - 60), max(30, y - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                    infos.append(f"{handedness}: {count} fingers | {confidence * 100:.0f}%")

                now = time.perf_counter()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                draw_panel(frame, infos, fps)
                cv2.putText(frame, "Q / ESC to quit", (18, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                cv2.imshow("Hand Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
