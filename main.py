import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
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
MAX_PARTICLES = 1400
MODES = ["GALAXY", "PLASMA", "FIRE", "WATER", "NEON"]
COLORS = [(255, 90, 180), (80, 220, 255), (100, 255, 150), (255, 180, 70), (220, 100, 255)]


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: float
    color: Tuple[int, int, int]


class ParticleEngine:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.mode = 0
        self.color_index = 0
        self.trail: List[Tuple[int, int]] = []
        self.prev_point = None
        self.clear_hold = 0

    @property
    def color(self):
        return COLORS[self.color_index]

    def clear(self):
        self.particles.clear()
        self.trail.clear()
        self.prev_point = None

    def emit(self, x, y, count=8, speed=2.5, color=None):
        color = color or self.color
        for _ in range(count):
            a = random.random() * math.tau
            s = random.uniform(0.4, speed)
            self.particles.append(Particle(
                x + random.uniform(-3, 3), y + random.uniform(-3, 3),
                math.cos(a) * s, math.sin(a) * s,
                random.uniform(35, 110), random.uniform(1.0, 3.5), color
            ))
        if len(self.particles) > MAX_PARTICLES:
            self.particles = self.particles[-MAX_PARTICLES:]

    def explosion(self, x, y):
        self.emit(x, y, 140, 7.0)
        for p in self.particles[-140:]:
            p.life += 60

    def attract_or_repel(self, p, x, y, strength, repel=False):
        dx, dy = x - p.x, y - p.y
        d2 = dx * dx + dy * dy + 120
        force = min(1.8, strength * 900 / d2)
        if repel:
            force = -force
        p.vx += dx * force
        p.vy += dy * force

    def update(self, hands):
        for p in self.particles:
            if self.mode == 0:  # Galaxy: orbit around the canvas center.
                cx, cy = self.width * 0.5, self.height * 0.5
                dx, dy = cx - p.x, cy - p.y
                p.vx += dx * 0.000018 - dy * 0.000010
                p.vy += dy * 0.000018 + dx * 0.000010
                p.vx *= 0.998
                p.vy *= 0.998
            elif self.mode == 1:  # Plasma: curl field.
                p.vx += math.sin(p.y * 0.018 + p.life * 0.02) * 0.045
                p.vy += math.cos(p.x * 0.018) * 0.045
                p.vx *= 0.997
                p.vy *= 0.997
            elif self.mode == 2:  # Fire: rising turbulence.
                p.vy -= 0.055
                p.vx += math.sin(p.y * 0.025) * 0.015
                p.vx *= 0.995
            elif self.mode == 3:  # Water: gravity + wave motion.
                p.vy += 0.018
                p.vx += math.sin(p.y * 0.025) * 0.012
                p.vy *= 0.998
            else:  # Neon: flowing electric field.
                p.vx += math.sin(p.y * 0.03 + p.life * 0.04) * 0.035
                p.vy += math.cos(p.x * 0.025) * 0.025
                p.vx *= 0.996
                p.vy *= 0.996

            for x, y, gesture, strength in hands:
                if gesture == "PINCH":
                    self.attract_or_repel(p, x, y, 1.0 + strength, False)
                elif gesture == "OPEN":
                    self.attract_or_repel(p, x, y, 0.8 + strength, True)

            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            p.size *= 0.997

        self.particles = [
            p for p in self.particles
            if p.life > 0 and -80 < p.x < self.width + 80 and -80 < p.y < self.height + 80
        ]

    def draw(self, canvas):
        glow = canvas.copy()
        for p in self.particles:
            x, y = int(p.x), int(p.y)
            if 0 <= x < self.width and 0 <= y < self.height:
                r = max(1, int(p.size))
                cv2.circle(canvas, (x, y), r, p.color, -1)
                cv2.circle(glow, (x, y), r * 4, p.color, -1)
        cv2.addWeighted(glow, 0.07, canvas, 0.93, 0, canvas)


def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe hand model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle(a, b, c):
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    al = math.hypot(abx, aby)
    cl = math.hypot(cbx, cby)
    if not al or not cl:
        return 0.0
    value = (abx * cbx + aby * cby) / (al * cl)
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def extended(landmarks, mcp, pip, dip, tip):
    wrist = landmarks[0]
    return (
        angle(landmarks[mcp], landmarks[pip], landmarks[dip]) >= 160
        and angle(landmarks[pip], landmarks[dip], landmarks[tip]) >= 155
        and distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist) * 1.08
    )


def thumb_extended(l):
    return (
        angle(l[1], l[2], l[3]) >= 145
        and angle(l[2], l[3], l[4]) >= 150
        and distance(l[4], l[0]) > distance(l[3], l[0]) * 1.15
        and distance(l[4], l[5]) > distance(l[3], l[5]) * 1.10
    )


def finger_states(l):
    return [
        thumb_extended(l),
        extended(l, 5, 6, 7, 8),
        extended(l, 9, 10, 11, 12),
        extended(l, 13, 14, 15, 16),
        extended(l, 17, 18, 19, 20),
    ]


def pinch_ratio(l):
    palm = max(distance(l[0], l[9]), 1e-5)
    return distance(l[4], l[8]) / palm


def classify(l):
    states = finger_states(l)
    count = sum(states)
    pinch = pinch_ratio(l)
    if pinch < 0.42:
        return "PINCH", count
    if count == 0:
        return "FIST", count
    if count >= 4:
        return "OPEN", count
    if states[1] and not any(states[2:]) and not states[0]:
        return "DRAW", count
    if states[1] and states[2] and not any(states[3:]) and not states[0]:
        return "TWO", count
    return "MOVE", count


def point(l, shape):
    h, w = shape[:2]
    return max(0, min(w - 1, int(l[8].x * w))), max(0, min(h - 1, int(l[8].y * h)))


def hand_skeleton(frame, l):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in l]
    for a, b in ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16)):
        cv2.line(frame, pts[a], pts[b], (90, 150, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (235, 235, 245), -1)
    cv2.circle(frame, pts[8], 13, (255, 255, 255), 2)


def ui(frame, engine, mode, fps, hands):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 130), (7, 7, 16), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, "HAND // PARTICLE UNIVERSE", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245,245,255), 2)
    cv2.putText(frame, f"{MODES[engine.mode]}  |  {len(engine.particles)} particles  |  {fps:.0f} FPS", (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.47, engine.color, 1)
    cv2.putText(frame, f"GESTURE: {mode}  |  HANDS: {hands}", (18, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220,220,230), 1)
    cv2.putText(frame, "1-5: worlds  C: color  X: clear  B: burst  Q: quit", (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190,190,205), 1)


def main():
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

    engine = ParticleEngine(1280, 720)
    prev_time = time.perf_counter()
    timestamp = 0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if engine.width != w or engine.height != h:
                    engine = ParticleEngine(w, h)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp += 33
                result = landmarker.detect_for_video(image, timestamp)

                hands = []
                primary_gesture = "NO HAND"
                for i, l in enumerate(result.hand_landmarks):
                    gesture, count = classify(l)
                    if i == 0:
                        primary_gesture = gesture
                    x, y = point(l, frame.shape)
                    speed = 0.0
                    if engine.prev_point is not None:
                        speed = math.hypot(x - engine.prev_point[0], y - engine.prev_point[1])

                    strength = min(1.0, speed / 80.0)
                    hands.append((x, y, gesture, strength))
                    hand_skeleton(frame, l)

                    if gesture == "DRAW":
                        engine.emit(x, y, 5 + min(8, int(speed / 8)), 1.5)
                        engine.trail.append((x, y))
                        if len(engine.trail) > 75:
                            engine.trail = engine.trail[-75:]
                    elif gesture == "PINCH":
                        engine.emit(x, y, 3, 0.8)
                    elif gesture == "FIST" and speed > 18:
                        engine.explosion(x, y)
                    elif gesture == "OPEN":
                        engine.emit(x, y, 2, 0.5)

                    engine.prev_point = (x, y)

                # Draw a fading trajectory behind the fingertip.
                for i in range(1, len(engine.trail)):
                    thickness = max(1, int(i / 16))
                    cv2.line(frame, engine.trail[i - 1], engine.trail[i], engine.color, thickness, cv2.LINE_AA)

                engine.update(hands)
                engine.draw(frame)

                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                ui(frame, engine, primary_gesture, fps, len(result.hand_landmarks))
                cv2.imshow("Hand - Particle Universe", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if ord('1') <= key <= ord('5'):
                    engine.mode = key - ord('1')
                    engine.clear()
                elif key in (ord('c'), ord('C')):
                    engine.color_index = (engine.color_index + 1) % len(COLORS)
                elif key in (ord('x'), ord('X')):
                    engine.clear()
                elif key in (ord('b'), ord('B')):
                    engine.explosion(w // 2, h // 2)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
