import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

CYAN = (255, 215, 40)
BLUE = (255, 130, 20)
ORANGE = (30, 170, 255)
WHITE = (235, 245, 255)
DARK = (5, 8, 15)


def ensure_model():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe face model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) <= 100_000:
        raise RuntimeError("Face model download failed or is incomplete.")


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check Windows Camera permissions and close other apps using the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def blendshape_map(categories):
    return {c.category_name: float(c.score) for c in categories}


def expression(bs):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    brow = (bs.get("browInnerUp", 0) + bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)) / 3
    squint = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
    if smile > 0.42:
        return "HAPPY", smile
    if jaw > 0.48 and wide > 0.25:
        return "SURPRISED", max(jaw, wide)
    if frown > 0.32 and brow < 0.30:
        return "SAD", frown
    if brow > 0.45 and wide > 0.18:
        return "ALERT", brow
    if squint > 0.42 and frown > 0.18:
        return "SERIOUS", squint
    return "NEUTRAL", 1.0 - min(1.0, max(smile, frown, jaw, wide))


def point(landmarks, idx, w, h):
    p = landmarks[idx]
    return int(p.x * w), int(p.y * h)


def line_chain(layer, landmarks, indices, w, h, color, thickness=1):
    pts = [point(landmarks, i, w, h) for i in indices if i < len(landmarks)]
    if len(pts) > 1:
        import numpy as np
        cv2.polylines(layer, [np.array(pts, dtype="int32")], False, color, thickness, cv2.LINE_AA)


def arc(layer, center, radius, start, end, color, thickness=1):
    cv2.ellipse(layer, center, (radius, radius), 0, start, end, color, thickness, cv2.LINE_AA)


def draw_face_hud(frame, landmarks, bs, phase):
    h, w = frame.shape[:2]
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1 = max(0, int(min(xs) * w) - 18)
    y1 = max(0, int(min(ys) * h) - 18)
    x2 = min(w - 1, int(max(xs) * w) + 18)
    y2 = min(h - 1, int(max(ys) * h) + 18)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    radius = max(70, min(175, int(face_w * 0.62)))
    name, score = expression(bs)

    glow = frame.copy()
    arm = 30
    for a, b in [((x1, y1), (x1 + arm, y1)), ((x1, y1), (x1, y1 + arm)),
                 ((x2, y1), (x2 - arm, y1)), ((x2, y1), (x2, y1 + arm)),
                 ((x1, y2), (x1 + arm, y2)), ((x1, y2), (x1, y2 - arm)),
                 ((x2, y2), (x2 - arm, y2)), ((x2, y2), (x2, y2 - arm))]:
        cv2.line(glow, a, b, CYAN, 2, cv2.LINE_AA)

    arc(glow, (cx, cy), radius, phase, phase + 92, CYAN, 2)
    arc(glow, (cx, cy), radius + 12, -phase * 1.7, -phase * 1.7 + 48, BLUE, 1)
    arc(glow, (cx, cy), radius + 27, phase * 0.55, phase * 0.55 + 22, ORANGE, 1)
    cv2.circle(glow, (cx, cy), 5, WHITE, -1)
    cv2.line(glow, (max(0, cx - radius - 35), cy), (min(w - 1, cx + radius + 35), cy), BLUE, 1, cv2.LINE_AA)
    cv2.line(glow, (cx, max(0, cy - 22)), (cx, min(h - 1, cy + 22)), BLUE, 1, cv2.LINE_AA)

    for chain in (
        (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378),
        (10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149),
        (33, 7, 163, 144, 145, 153, 154, 155, 133),
        (263, 249, 390, 373, 374, 380, 381, 382, 362),
    ):
        line_chain(glow, landmarks, chain, w, h, BLUE, 1)

    for idx in (33, 263):
        px, py = point(landmarks, idx, w, h)
        cv2.circle(glow, (px, py), 11, CYAN, 1, cv2.LINE_AA)
        cv2.line(glow, (px - 17, py), (px + 17, py), CYAN, 1, cv2.LINE_AA)
        cv2.line(glow, (px, py - 17), (px, py + 17), CYAN, 1, cv2.LINE_AA)

    beam_y = y1 + int(((math.sin(phase * 0.045) + 1) * 0.5) * max(1, y2 - y1))
    cv2.line(glow, (x1, beam_y), (x2, beam_y), CYAN, 2, cv2.LINE_AA)
    cv2.putText(glow, "TARGET ACQUIRED", (x1, max(18, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .38, CYAN, 1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.92, frame, 0.08, 0, frame)

    panel_w, panel_h = 250, 125
    px = x2 + 28
    if px + panel_w >= w:
        px = max(18, x1 - panel_w - 28)
    py = max(60, min(y1, h - panel_h - 18))
    panel = frame.copy()
    cv2.rectangle(panel, (px, py), (px + panel_w, py + panel_h), DARK, -1)
    cv2.addWeighted(panel, .84, frame, .16, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), BLUE, 1, cv2.LINE_AA)

    lines = [
        ("J.A.R.V.I.S.", .58, WHITE, 2),
        ("FACIAL INTERFACE // ONLINE", .32, CYAN, 1),
        (f"EXPRESSION   {name}", .38, ORANGE, 1),
        (f"CONFIDENCE   {score * 100:04.1f}%", .36, WHITE, 1),
    ]
    y = py + 27
    for text, size, color, thickness in lines:
        cv2.putText(frame, text, (px + 14, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
        y += 25
    return name


def main():
    ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_face_blendshapes=True,
    )

    cap = open_camera()
    phase = 0.0
    previous = time.perf_counter()
    expression_history = deque(maxlen=7)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                phase = (phase + 2.2) % 360

                detected = bool(result.face_landmarks)
                name = "SEARCHING"
                if detected:
                    bs = blendshape_map(result.face_blendshapes[0]) if result.face_blendshapes else {}
                    raw_name = draw_face_hud(frame, result.face_landmarks[0], bs, phase)
                    expression_history.append(raw_name)
                    name = max(set(expression_history), key=expression_history.count)
                else:
                    expression_history.clear()

                now = time.perf_counter()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now

                cv2.putText(frame, "J.A.R.V.I.S. // FACIAL INTERFACE", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, .48, WHITE, 1, cv2.LINE_AA)
                status = "TARGET ACQUIRED" if detected else "SCANNING FOR TARGET"
                status_color = CYAN if detected else ORANGE
                cv2.putText(frame, f"{status}   |   {name}   |   {fps:.0f} FPS", (22, 51), cv2.FONT_HERSHEY_SIMPLEX, .34, status_color, 1, cv2.LINE_AA)

                hh, ww = frame.shape[:2]
                for i in range(0, 360, 45):
                    a = math.radians(i + phase)
                    r1, r2 = 18, 25
                    p1 = (int(ww / 2 + math.cos(a) * (ww / 2 - r1)), int(hh / 2 + math.sin(a) * (hh / 2 - r1)))
                    p2 = (int(ww / 2 + math.cos(a) * (ww / 2 - r2)), int(hh / 2 + math.sin(a) * (hh / 2 - r2)))
                    cv2.line(frame, p1, p2, BLUE, 1, cv2.LINE_AA)

                cv2.imshow("JARVIS // Facial Interface", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
