import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_hud import blendshape_map, draw_face_hud, ensure_model as ensure_face_model

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
WINDOW = "J.A.R.V.I.S. // Hand + Face Interface"

CYAN = (255, 215, 40)
BLUE = (255, 130, 20)
ORANGE = (30, 170, 255)
WHITE = (235, 245, 255)
DARK = (8, 10, 18)


class VisualState:
    """State for the clean HUD effects; intentionally contains no particles."""

    def __init__(self):
        self.energy = 72.0
        self.rotation = 0.0
        self.lock = 0.0
        self.last_two_distance = None

    def update(self, has_hands):
        self.rotation = (self.rotation + 0.018) % math.tau
        self.energy = min(100.0, self.energy + 0.08)
        self.lock = min(1.0, self.lock) if has_hands else self.lock * 0.94


def ensure_hand_model():
    if os.path.isfile(HAND_MODEL_PATH) and os.path.getsize(HAND_MODEL_PATH) > 100_000:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe hand model (first run only)...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle(a, b, c):
    ux, uy = a.x - b.x, a.y - b.y
    vx, vy = c.x - b.x, c.y - b.y
    den = math.hypot(ux, uy) * math.hypot(vx, vy)
    if den == 0:
        return 0
    return math.degrees(math.acos(max(-1, min(1, (ux * vx + uy * vy) / den))))


def finger_states(l):
    wrist = l[0]
    out = [angle(l[1], l[2], l[3]) > 140 and angle(l[2], l[3], l[4]) > 145 and dist(l[4], wrist) > dist(l[3], wrist) * 1.10]
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        out.append(angle(l[mcp], l[pip], l[dip]) > 158 and angle(l[pip], l[dip], l[tip]) > 153 and dist(l[tip], wrist) > dist(l[pip], wrist) * 1.07)
    return out


def classify(l):
    s = finger_states(l)
    pinch = dist(l[4], l[8]) / max(dist(l[0], l[9]), 1e-5)
    if pinch < 0.40:
        return "PINCH"
    if sum(s) == 0:
        return "FIST"
    if sum(s) >= 4:
        return "OPEN"
    if s[1] and not any(s[2:]) and not s[0]:
        return "POINT"
    if s[1] and s[2] and not any(s[3:]) and not s[0]:
        return "TWO"
    return "MOVE"


def xy(l, shape):
    h, w = shape[:2]
    return int(l[8].x * w), int(l[8].y * h)


def square_metrics(points):
    points = list(points)
    if len(points) < 20:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y
    if width < 35 or height < 35:
        return None
    scale = max(width, height)
    closure = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    aspect_error = abs(width - height) / scale
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    path_ratio = path / max(2.0 * (width + height), 1.0)
    score = 1 - min(1, aspect_error / .35) * .35 - min(1, (closure / scale) / .25) * .40 - min(1, abs(path_ratio - 1) / .8) * .25
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "score": score,
            "valid": closure / scale < .25 and aspect_error < .35 and .7 < path_ratio < 2.2 and score >= .58}


def clean_square(m):
    side = int(max(m["max_x"] - m["min_x"], m["max_y"] - m["min_y"]))
    cx, cy = int((m["min_x"] + m["max_x"]) / 2), int((m["min_y"] + m["max_y"]) / 2)
    half = side // 2
    return cx - half, cy - half, cx + half, cy + half


def draw_hand(frame, l, gesture):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in l]
    links = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16))
    for a, b in links:
        cv2.line(frame, pts[a], pts[b], CYAN, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, WHITE, -1)
    x, y = pts[8]
    for r in (16, 24):
        cv2.circle(frame, (x, y), r, BLUE, 1, cv2.LINE_AA)
    cv2.putText(frame, gesture, (x + 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, .42, CYAN, 1, cv2.LINE_AA)


def arc_reactor(frame, center, radius, phase, energy):
    cx, cy = center
    layer = frame.copy()
    for r, speed, width in ((radius, 1, 2), (radius * .78, -1.7, 1), (radius * .56, 2.2, 1)):
        cv2.ellipse(layer, center, (int(r), int(r)), 0, phase * speed, phase * speed + 300, CYAN, width, cv2.LINE_AA)
    cv2.circle(layer, center, int(radius * .36), BLUE, 2)
    cv2.circle(layer, center, int(radius * .22), CYAN, -1)
    cv2.addWeighted(layer, .92, frame, .08, 0, frame)
    cv2.putText(frame, f"ARC ENERGY {energy:03.0f}%", (cx - 70, cy + int(radius * 1.55)), cv2.FONT_HERSHEY_SIMPLEX, .43, CYAN, 1, cv2.LINE_AA)


def hud(frame, gesture, hands, fps, energy, locked, face_detected, face_expression):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (440, 174), DARK, -1)
    cv2.addWeighted(overlay, .72, frame, .28, 0, frame)
    rows = [
        ("J.A.R.V.I.S.", .85, WHITE, 2),
        ("HAND + FACIAL INTERFACE // ONLINE", .38, CYAN, 1),
        (f"GESTURE   {gesture:<8}   HANDS {hands}", .42, WHITE, 1),
        (f"ENERGY    {energy:05.1f}%   FPS {fps:04.0f}", .42, WHITE, 1),
        (f"TARGET    {'LOCKED' if locked else 'SEARCHING'}", .42, ORANGE if locked else CYAN, 1),
        (f"FACE      {'DETECTED' if face_detected else 'SEARCHING'}  {face_expression}", .40, CYAN if face_detected else ORANGE, 1),
    ]
    y = 48
    for text, size, color, thick in rows:
        cv2.putText(frame, text, (34, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y += 23
    cv2.putText(frame, "POINT: DRAW SQUARE   PINCH: GRAB   OPEN: REPULSOR   F11: FULLSCREEN", (max(20, w // 2 - 390), h - 24), cv2.FONT_HERSHEY_SIMPLEX, .39, CYAN, 1, cv2.LINE_AA)


def expression_name(value):
    """Normalize facial-HUD return values to a hashable expression string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("expression", "name", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return "NEUTRAL"
    if isinstance(value, (tuple, list)):
        for item in value:
            name = expression_name(item)
            if name != "NEUTRAL":
                return name
    return "NEUTRAL"


def main():
    ensure_hand_model()
    ensure_face_model()
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=.62, min_hand_presence_confidence=.62, min_tracking_confidence=.62)
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=.35, min_face_presence_confidence=.35,
        min_tracking_confidence=.35, output_face_blendshapes=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    visual = VisualState()
    drawing_points = deque(maxlen=180)
    square_rect = None
    square_hold = 0
    expression_history = deque(maxlen=7)
    timestamp = 0
    prev = time.perf_counter()

    try:
        with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp += 33

                hand_result = hand_landmarker.detect_for_video(image, timestamp)
                face_result = face_landmarker.detect_for_video(image, timestamp)
                visual.rotation = (visual.rotation + 2.2) % 360

                face_detected = bool(face_result.face_landmarks)
                face_expression = "SEARCHING"
                if face_detected:
                    bs = blendshape_map(face_result.face_blendshapes[0]) if face_result.face_blendshapes else {}
                    raw = draw_face_hud(frame, face_result.face_landmarks[0], bs, visual.rotation)
                    raw_name = expression_name(raw)
                    expression_history.append(raw_name)
                    face_expression = max(set(expression_history), key=expression_history.count)
                else:
                    expression_history.clear()

                gestures = []
                for l in hand_result.hand_landmarks:
                    gesture = classify(l)
                    x, y = xy(l, frame.shape)
                    gestures.append(gesture)
                    draw_hand(frame, l, gesture)
                    if gesture == "POINT":
                        drawing_points.append((x, y))
                        if len(drawing_points) >= 20:
                            m = square_metrics(drawing_points)
                            if m and m["valid"]:
                                square_rect, square_hold = clean_square(m), 18
                        visual.lock = min(1, visual.lock + .08)
                    elif gesture == "PINCH":
                        visual.energy = max(0, visual.energy - .04)
                    elif gesture == "OPEN":
                        visual.energy = min(100, visual.energy + .16)
                    elif gesture == "FIST":
                        visual.energy = min(100, visual.energy + .3)

                visual.lock = min(1, visual.lock) if hand_result.hand_landmarks else visual.lock * .94
                if len(hand_result.hand_landmarks) >= 2:
                    a, b = xy(hand_result.hand_landmarks[0], frame.shape), xy(hand_result.hand_landmarks[1], frame.shape)
                    cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                    d = math.hypot(a[0] - b[0], a[1] - b[1])
                    visual.last_two_distance = d
                    arc_reactor(frame, (cx, cy), max(45, min(125, d * .25)), visual.rotation, visual.energy)
                    cv2.line(frame, a, b, BLUE, 1, cv2.LINE_AA)
                else:
                    visual.last_two_distance = None

                if square_hold > 0 and square_rect:
                    x1, y1, x2, y2 = square_rect
                    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2, cv2.LINE_AA)
                    cv2.putText(frame, "SQUARE DETECTED", (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .48, CYAN, 1, cv2.LINE_AA)
                    square_hold -= 1
                elif not gestures or gestures[0] != "POINT":
                    drawing_points.clear()

                visual.update(bool(hand_result.hand_landmarks))

                now = time.perf_counter()
                fps = 1 / max(now - prev, 1e-6)
                prev = now
                primary = gestures[0] if gestures else "NO HAND"
                hud(frame, primary, len(hand_result.hand_landmarks), fps, visual.energy, visual.lock > .65, face_detected, face_expression)
                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == 0x7A:
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key in (ord('r'), ord('R')):
                    visual = VisualState()
                    drawing_points.clear()
                    square_rect = None
                    square_hold = 0
                if key in (ord('b'), ord('B')):
                    visual.energy = min(100, visual.energy + 8)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
