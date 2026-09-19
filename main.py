import math
import os
import random
import time
import urllib.request
from collections import deque
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

COLORS = [(255, 90, 180), (80, 220, 255), (120, 255, 150), (255, 190, 70), (220, 120, 255)]


def ensure_model() -> None:
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe hand model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle(a, b, c) -> float:
    ab_x, ab_y = a.x - b.x, a.y - b.y
    cb_x, cb_y = c.x - b.x, c.y - b.y
    ab_len = math.hypot(ab_x, ab_y)
    cb_len = math.hypot(cb_x, cb_y)
    if ab_len == 0 or cb_len == 0:
        return 0.0
    cosine = (ab_x * cb_x + ab_y * cb_y) / (ab_len * cb_len)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def finger_is_extended(landmarks, mcp: int, pip: int, dip: int, tip: int) -> bool:
    wrist = landmarks[0]
    pip_angle = joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
    dip_angle = joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
    straight = pip_angle >= 160 and dip_angle >= 155
    extended = distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist) * 1.08
    return straight and extended


def thumb_is_extended(landmarks) -> bool:
    wrist = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    mcp_angle = joint_angle(landmarks[1], thumb_mcp, thumb_ip)
    ip_angle = joint_angle(thumb_mcp, thumb_ip, thumb_tip)
    length_ok = distance(thumb_tip, wrist) > distance(thumb_ip, wrist) * 1.15
    away_from_palm = distance(thumb_tip, index_mcp) > distance(thumb_ip, index_mcp) * 1.10
    return mcp_angle >= 145 and ip_angle >= 150 and length_ok and away_from_palm


def finger_states(landmarks) -> List[bool]:
    return [
        thumb_is_extended(landmarks),
        finger_is_extended(landmarks, 5, 6, 7, 8),
        finger_is_extended(landmarks, 9, 10, 11, 12),
        finger_is_extended(landmarks, 13, 14, 15, 16),
        finger_is_extended(landmarks, 17, 18, 19, 20),
    ]


def count_fingers(landmarks, handedness: str) -> int:
    del handedness
    return sum(finger_states(landmarks))


def gesture_name(count: int) -> str:
    return {0: "FIST", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "OPEN HAND"}.get(count, "UNKNOWN")


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


def draw_panel(frame, title: str, lines: List[str], fps: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, 175), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.76, frame, 0.24, 0)
    cv2.putText(frame, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    y = 84
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 245, 210), 1)
        y += 22


def fingertip_pixel(landmarks, frame_shape) -> Tuple[int, int]:
    h, w = frame_shape[:2]
    tip = landmarks[8]
    return max(0, min(w - 1, int(tip.x * w))), max(0, min(h - 1, int(tip.y * h)))


def pinch_ratio(landmarks) -> float:
    palm = max(distance(landmarks[0], landmarks[9]), 1e-5)
    return distance(landmarks[4], landmarks[8]) / palm


def add_particles(particles, point, color, amount=8, speed=2.4):
    x, y = point
    for _ in range(amount):
        angle = random.random() * math.tau
        velocity = random.uniform(0.4, speed)
        particles.append([
            float(x), float(y),
            math.cos(angle) * velocity, math.sin(angle) * velocity,
            random.randint(8, 20), color,
        ])


def update_particles(canvas, particles):
    alive = []
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[3] += 0.025
        p[4] -= 1
        if p[4] > 0:
            radius = max(1, p[4] // 5)
            cv2.circle(canvas, (int(p[0]), int(p[1])), radius, p[5], -1)
            alive.append(p)
    particles[:] = alive[-900:]


def draw_glow_line(canvas, a, b, color, size):
    glow = canvas.copy()
    cv2.line(glow, a, b, color, max(size * 3, 8), cv2.LINE_AA)
    cv2.addWeighted(glow, 0.14, canvas, 0.86, 0, canvas)
    cv2.line(canvas, a, b, color, size, cv2.LINE_AA)


def air_drawing(frame, result, canvas, particles, state):
    hands = result.hand_landmarks
    if not hands:
        state["prev"] = None
        update_particles(canvas, particles)
        return "NO HAND", state["color"], state["size"]

    landmarks = hands[0]
    states = finger_states(landmarks)
    count = sum(states)
    gesture = gesture_name(count)
    point = fingertip_pixel(landmarks, frame.shape)
    pinch = pinch_ratio(landmarks)

    # Pinch changes brush size continuously: close pinch = thin, open pinch = thick.
    if states[0] and states[1] and pinch < 1.0:
        state["size"] = max(2, min(28, int(5 + (1.0 - pinch) * 24)))
        state["prev"] = None
        state["mode"] = "SIZE"
    elif states[1] and not any(states[2:]):
        # Index only = draw with a glowing motion trail.
        state["mode"] = "DRAW"
        if state["prev"] is not None:
            draw_glow_line(canvas, state["prev"], point, state["color"], state["size"])
            dx = point[0] - state["prev"][0]
            dy = point[1] - state["prev"][1]
            speed = min(7.0, math.hypot(dx, dy) / 3.0)
            if speed > 1.0:
                add_particles(particles, point, state["color"], min(7, int(speed)), speed)
        state["prev"] = point
    elif states[1] and states[2] and not any(states[3:]) and not states[0]:
        # Two fingers = erase with a soft brush.
        state["mode"] = "ERASE"
        state["prev"] = None
        erase = max(12, state["size"] * 2)
        cv2.circle(canvas, point, erase, (0, 0, 0), -1)
    elif count == 0:
        # Fist = pause drawing and emit a small burst.
        state["mode"] = "PAUSE"
        state["prev"] = None
    elif count == 5:
        # Open palm clears the canvas only after being held briefly.
        state["mode"] = "CLEAR"
        state["prev"] = None
        state["clear_frames"] += 1
        if state["clear_frames"] >= 12:
            canvas[:] = 0
            particles.clear()
            state["clear_frames"] = 0
    else:
        state["mode"] = gesture
        state["prev"] = None
        state["clear_frames"] = 0

    update_particles(canvas, particles)
    return state["mode"], state["color"], state["size"]


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

    canvas = None
    particles = []
    state = {"prev": None, "color": COLORS[0], "color_index": 0, "size": 6, "mode": "PAUSE", "clear_frames": 0}
    prev = time.perf_counter()
    timestamp_ms = 0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                if canvas is None or canvas.shape != frame.shape:
                    canvas = frame.copy()
                    canvas[:] = 0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                mode, color, size = air_drawing(frame, result, canvas, particles, state)

                # Composite the artwork over the camera feed.
                mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                inverse = cv2.bitwise_not(mask)
                camera_part = cv2.bitwise_and(frame, frame, mask=inverse)
                frame = cv2.add(camera_part, canvas)

                infos = []
                for index, landmarks in enumerate(result.hand_landmarks[:2]):
                    handedness = "Unknown"
                    confidence = 0.0
                    if index < len(result.handedness) and result.handedness[index]:
                        category = result.handedness[index][0]
                        handedness = category.category_name or category.display_name or "Unknown"
                        confidence = category.score
                    count = count_fingers(landmarks, handedness)
                    draw_hand(frame, landmarks)
                    infos.append(f"{handedness}: {count} fingers | {confidence * 100:.0f}%")

                now = time.perf_counter()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now

                draw_panel(frame, "ADVANCED AIR DRAWING", [
                    f"Mode: {mode}",
                    f"Brush: {size}px | Color: {state['color_index'] + 1}",
                    "Index = draw | 2 fingers = erase",
                    "Pinch = size | Palm = clear | Fist = pause",
                ], fps)

                y = 190
                for info in infos:
                    cv2.putText(frame, info, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 160), 1)
                    y += 22

                cv2.putText(frame, "C = color  |  X = clear  |  Q / ESC = quit", (18, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                cv2.imshow("Hand - Advanced Air Drawing", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    state["color_index"] = (state["color_index"] + 1) % len(COLORS)
                    state["color"] = COLORS[state["color_index"]]
                elif key == ord("x"):
                    canvas[:] = 0
                    particles.clear()
                elif key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
