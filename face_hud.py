import math
import os
import time
import urllib.request

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
DARK = (8, 10, 18)


def ensure_model():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe face model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.isfile(MODEL_PATH) or os.path.getsize(MODEL_PATH) <= 100_000:
        raise RuntimeError("Face model download failed or is incomplete.")
    print("Face model downloaded.")


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


def face_box(landmarks, shape):
    h, w = shape[:2]
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1 = max(0, int(min(xs) * w) - 14)
    y1 = max(0, int(min(ys) * h) - 14)
    x2 = min(w - 1, int(max(xs) * w) + 14)
    y2 = min(h - 1, int(max(ys) * h) + 14)
    return x1, y1, x2, y2


def arc(layer, center, radius, start, end, color, thickness=1):
    cv2.ellipse(layer, center, (radius, radius), 0, start, end, color, thickness, cv2.LINE_AA)


def draw_face_hud(frame, landmarks, bs, phase):
    x1, y1, x2, y2 = face_box(landmarks, frame.shape)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = max(55, min(150, int((x2 - x1) * 0.58)))
    expression_name, confidence = expression(bs)

    glow = frame.copy()
    arm = 22
    for a, b in [((x1, y1), (x1 + arm, y1)), ((x1, y1), (x1, y1 + arm)),
                 ((x2, y1), (x2 - arm, y1)), ((x2, y1), (x2, y1 + arm)),
                 ((x1, y2), (x1 + arm, y2)), ((x1, y2), (x1, y2 - arm)),
                 ((x2, y2), (x2 - arm, y2)), ((x2, y2), (x2, y2 - arm))]:
        cv2.line(glow, a, b, CYAN, 2, cv2.LINE_AA)

    arc(glow, (cx, cy), radius, phase % 360, (phase + 70) % 360, CYAN, 2)
    arc(glow, (cx, cy), radius + 10, (-phase * 1.6) % 360, (-phase * 1.6 + 38) % 360, BLUE, 1)
    cv2.circle(glow, (cx, cy), 4, WHITE, -1)
    cv2.line(glow, (cx - radius - 18, cy), (cx + radius + 18, cy), BLUE, 1, cv2.LINE_AA)

    # Robust eye scan markers: use the actual eye-region landmarks.
    for idx in (33, 263):
        if idx < len(landmarks):
            p = landmarks[idx]
            px, py = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
            cv2.circle(glow, (px, py), 8, CYAN, 1, cv2.LINE_AA)
            cv2.line(glow, (px - 13, py), (px + 13, py), CYAN, 1, cv2.LINE_AA)
            cv2.line(glow, (px, py - 13), (px, py + 13), CYAN, 1, cv2.LINE_AA)

    cv2.addWeighted(glow, 0.88, frame, 0.12, 0, frame)

    panel_w = 267
    px = min(frame.shape[1] - panel_w - 18, max(18, x2 + 28))
    py = max(55, y1)
    if px + panel_w > frame.shape[1]:
        px = max(18, x1 - panel_w - 28)
    panel = frame.copy()
    cv2.rectangle(panel, (px, py), (px + panel_w, py + 132), DARK, -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "FACE ANALYSIS", (px + 14, py + 25), cv2.FONT_HERSHEY_SIMPLEX, .52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "STATUS   TRACKING", (px + 14, py + 47), cv2.FONT_HERSHEY_SIMPLEX, .39, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"EXPRESSION  {expression_name}", (px + 14, py + 70), cv2.FONT_HERSHEY_SIMPLEX, .43, ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SIGNAL      {confidence * 100:04.1f}%", (px + 14, py + 92), cv2.FONT_HERSHEY_SIMPLEX, .39, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"BLINK L/R   {bs.get('eyeBlinkLeft', 0):.2f} / {bs.get('eyeBlinkRight', 0):.2f}", (px + 14, py + 112), cv2.FONT_HERSHEY_SIMPLEX, .36, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"JAW OPEN    {bs.get('jawOpen', 0):.2f}", (px + 14, py + 128), cv2.FONT_HERSHEY_SIMPLEX, .36, WHITE, 1, cv2.LINE_AA)
    return expression_name


def open_camera():
    # DirectShow is more reliable on Windows than the default backend for webcams.
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


def main():
    ensure_model()

    # IMAGE mode is intentionally used here. It avoids timestamp/tracking issues
    # and is very reliable for a single webcam face while still running in real time.
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
    frames = 0
    detected = False

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                phase = (phase + 2.2) % 360
                frames += 1

                expression_name = "NO FACE"
                if result.face_landmarks:
                    detected = True
                    bs = blendshape_map(result.face_blendshapes[0]) if result.face_blendshapes else {}
                    expression_name = draw_face_hud(frame, result.face_landmarks[0], bs, phase)
                else:
                    detected = False

                now = time.perf_counter()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now

                overlay = frame.copy()
                cv2.rectangle(overlay, (18, 18), (435, 112), DARK, -1)
                cv2.addWeighted(overlay, .78, frame, .22, 0, frame)
                status = "TRACKING" if detected else "SEARCHING..."
                status_color = CYAN if detected else ORANGE
                cv2.putText(frame, "J.A.R.V.I.S. // FACE MODULE", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, .62, WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"FACE: {status}   EXP: {expression_name}   FPS: {fps:.0f}", (34, 73), cv2.FONT_HERSHEY_SIMPLEX, .36, status_color, 1, cv2.LINE_AA)
                cv2.putText(frame, "LOCAL ANALYSIS // NO IMAGE UPLOAD", (34, 96), cv2.FONT_HERSHEY_SIMPLEX, .34, WHITE, 1, cv2.LINE_AA)

                if frames == 30 and not detected:
                    print("No face detected after 30 frames. Move closer, face the camera, and improve lighting.")

                cv2.imshow("JARVIS - Face HUD", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
