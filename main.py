import math
import os
import random
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

CYAN = (255, 215, 40)
BLUE = (255, 130, 20)
ORANGE = (30, 170, 255)
WHITE = (235, 245, 255)
DARK = (8, 10, 18)


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    life: float
    size: float


class Hologram:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.particles: List[Particle] = []
        self.trails = deque(maxlen=22)
        self.energy = 72.0
        self.rotation = 0.0
        self.pulse = 0.0
        self.lock = 0.0
        self.explosion = 0.0
        self.last_two_distance = None

    def spawn(self, x, y, count=10, burst=1.0):
        for _ in range(count):
            a = random.random() * math.tau
            speed = random.uniform(1.0, 4.0) * burst
            self.particles.append(Particle(x + random.uniform(-7, 7), y + random.uniform(-7, 7), random.uniform(-1, 1), math.cos(a) * speed, math.sin(a) * speed, random.uniform(-1.5, 1.5), random.uniform(30, 100), random.uniform(1.0, 3.0)))
        if len(self.particles) > 1800:
            del self.particles[:-1800]

    def blast(self, x, y, amount=180):
        self.spawn(x, y, amount, 3.2)
        self.pulse = 1.0
        self.explosion = 1.0

    def update(self, hands):
        self.rotation += 0.018
        self.pulse *= 0.93
        self.explosion *= 0.94
        self.energy += 0.08
        self.energy = min(100, self.energy)

        for p in self.particles:
            # Floating 3D-space illusion.
            p.vx += math.sin(p.y * 0.012 + p.z * 2.0) * 0.012
            p.vy += math.cos(p.x * 0.010 - p.z) * 0.012
            p.vz += math.sin(p.x * 0.008 + p.y * 0.006) * 0.008

            for x, y, gesture, strength in hands:
                dx, dy = x - p.x, y - p.y
                d2 = dx * dx + dy * dy + 500
                if gesture == "PINCH":
                    f = min(1.8, 1200 / d2) * (1 + strength)
                    p.vx += dx * f * 0.015
                    p.vy += dy * f * 0.015
                elif gesture == "OPEN":
                    f = min(1.4, 900 / d2) * (1 + strength)
                    p.vx -= dx * f * 0.012
                    p.vy -= dy * f * 0.012
                elif gesture == "FIST":
                    p.vx += (self.w / 2 - p.x) * 0.0008
                    p.vy += (self.h / 2 - p.y) * 0.0008

            p.x += p.vx
            p.y += p.vy
            p.z += p.vz
            p.vx *= 0.988
            p.vy *= 0.988
            p.vz *= 0.985
            p.life -= 1
            p.size *= 0.998

        self.particles = [p for p in self.particles if p.life > 0 and -100 < p.x < self.w + 100 and -100 < p.y < self.h + 100]

    def draw(self, frame):
        glow = frame.copy()
        for p in self.particles:
            scale = 1.0 / max(0.55, 1.0 + p.z * 0.15)
            x, y = int(p.x), int(p.y)
            r = max(1, int(p.size * scale))
            if 0 <= x < self.w and 0 <= y < self.h:
                cv2.circle(frame, (x, y), r, CYAN, -1)
                cv2.circle(glow, (x, y), r * 5, BLUE, -1)
        cv2.addWeighted(glow, 0.08, frame, 0.92, 0, frame)


def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


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
    out = [
        angle(l[1], l[2], l[3]) > 140 and angle(l[2], l[3], l[4]) > 145 and dist(l[4], wrist) > dist(l[3], wrist) * 1.10,
    ]
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
    if len(points) < 20:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width < 35 or height < 35:
        return None
    closure = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    scale = max(width, height)
    aspect_error = abs(width - height) / scale
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    perimeter = 2.0 * (width + height)
    path_ratio = path / max(perimeter, 1.0)
    score = 1.0
    score -= min(1.0, aspect_error / 0.35) * 0.35
    score -= min(1.0, (closure / scale) / 0.25) * 0.40
    score -= min(1.0, abs(path_ratio - 1.0) / 0.8) * 0.25
    return {
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        "score": score,
        "valid": closure / scale < 0.25 and aspect_error < 0.35 and 0.7 < path_ratio < 2.2 and score >= 0.58,
    }


def clean_square(metrics):
    side = int(max(metrics["max_x"] - metrics["min_x"], metrics["max_y"] - metrics["min_y"]))
    cx = int((metrics["min_x"] + metrics["max_x"]) / 2)
    cy = int((metrics["min_y"] + metrics["max_y"]) / 2)
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
    cv2.circle(frame, (x, y), 16, BLUE, 1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 24, BLUE, 1, cv2.LINE_AA)
    cv2.line(frame, (x - 34, y), (x - 10, y), BLUE, 1)
    cv2.line(frame, (x + 10, y), (x + 34, y), BLUE, 1)
    cv2.line(frame, (x, y - 34), (x, y - 10), BLUE, 1)
    cv2.line(frame, (x, y + 10), (x, y + 34), BLUE, 1)
    cv2.putText(frame, gesture, (x + 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)


def ring(layer, center, radius, color=CYAN, thickness=1, segments=64, phase=0):
    cx, cy = center
    pts = []
    for i in range(segments + 1):
        a = phase + math.tau * i / segments
        pts.append((int(cx + math.cos(a) * radius), int(cy + math.sin(a) * radius)))
    cv2.polylines(layer, [__import__('numpy').array(pts, dtype='int32')], False, color, thickness, cv2.LINE_AA)


def arc_reactor(frame, center, radius, phase, energy):
    layer = frame.copy()
    cx, cy = center
    for r, speed, width in ((radius, 1.0, 2), (radius * .78, -1.7, 1), (radius * .56, 2.2, 1)):
        ring(layer, center, r, CYAN, width, 48, phase * speed)
        ring(layer, center, r * 1.08, BLUE, 1, 36, -phase * speed * 0.7)
    for i in range(8):
        a = phase * (1.0 if i % 2 else -0.8) + i * math.tau / 8
        x = int(cx + math.cos(a) * radius * 1.22)
        y = int(cy + math.sin(a) * radius * 1.22)
        cv2.circle(layer, (x, y), 3, WHITE, -1)
    cv2.circle(layer, (cx, cy), int(radius * .36), BLUE, 2)
    cv2.circle(layer, (cx, cy), int(radius * .22), CYAN, -1)
    cv2.addWeighted(layer, 0.92, frame, 0.08, 0, frame)
    cv2.putText(frame, f"ARC ENERGY {energy:03.0f}%", (cx - 70, cy + int(radius * 1.55)), cv2.FONT_HERSHEY_SIMPLEX, .43, CYAN, 1, cv2.LINE_AA)


def hud(frame, gesture, hands, fps, energy, locked):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (390, 148), DARK, -1)
    cv2.addWeighted(overlay, .72, frame, .28, 0, frame)
    cv2.putText(frame, "J.A.R.V.I.S.", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, .85, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "HAND INTERFACE // ONLINE", (34, 70), cv2.FONT_HERSHEY_SIMPLEX, .40, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"GESTURE   {gesture:<8}   HANDS {hands}", (34, 94), cv2.FONT_HERSHEY_SIMPLEX, .42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ENERGY    {energy:05.1f}%   FPS {fps:04.0f}", (34, 116), cv2.FONT_HERSHEY_SIMPLEX, .42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"TARGET    {'LOCKED' if locked else 'SEARCHING'}", (34, 138), cv2.FONT_HERSHEY_SIMPLEX, .42, ORANGE if locked else CYAN, 1, cv2.LINE_AA)

    # Minimal bottom command strip.
    cv2.putText(frame, "POINT: DRAW SQUARE   PINCH: GRAB   OPEN: REPULSOR   FIST: CHARGE", (w // 2 - 300, h - 24), cv2.FONT_HERSHEY_SIMPLEX, .39, CYAN, 1, cv2.LINE_AA)


def main():
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=.62,
        min_hand_presence_confidence=.62,
        min_tracking_confidence=.62,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    holo = Hologram(1280, 720)
    prev = time.perf_counter()
    timestamp = 0
    previous_center = None
    drawing_points = deque(maxlen=180)
    square_rect = None
    square_hold = 0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if (w, h) != (holo.w, holo.h):
                    holo = Hologram(w, h)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp += 33
                result = landmarker.detect_for_video(image, timestamp)
                hand_data = []
                gestures = []

                for l in result.hand_landmarks:
                    gesture = classify(l)
                    x, y = xy(l, frame.shape)
                    gestures.append(gesture)
                    hand_data.append((x, y, gesture, 0.0))
                    draw_hand(frame, l, gesture)

                    if gesture == "POINT":
                        drawing_points.append((x, y))
                        if len(drawing_points) >= 20:
                            metrics = square_metrics(drawing_points)
                            if metrics and metrics["valid"]:
                                square_rect = clean_square(metrics)
                                square_hold = 18
                        holo.lock = min(1.0, holo.lock + .08)
                        holo.spawn(x, y, 3, .5)
                        holo.trails.append((x, y))
                    elif gesture == "PINCH":
                        holo.energy = max(0, holo.energy - .04)
                        holo.spawn(x, y, 7, 1.1)
                        holo.trails.append((x, y))
                    elif gesture == "OPEN":
                        holo.energy = min(100, holo.energy + .16)
                        holo.spawn(x, y, 4, .7)
                    elif gesture == "FIST":
                        holo.energy = min(100, holo.energy + .3)
                        if random.random() < .28:
                            holo.spawn(x, y, 5, .8)

                if not result.hand_landmarks:
                    holo.lock *= .94
                else:
                    holo.lock = min(1, holo.lock)

                # Two hands = floating Arc Reactor between the hands.
                if len(result.hand_landmarks) >= 2:
                    a = xy(result.hand_landmarks[0], frame.shape)
                    b = xy(result.hand_landmarks[1], frame.shape)
                    cx, cy = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                    d = math.hypot(a[0] - b[0], a[1] - b[1])
                    radius = max(45, min(125, d * .25))
                    if holo.last_two_distance is not None and d - holo.last_two_distance > 70:
                        holo.blast(cx, cy, 220)
                    holo.last_two_distance = d
                    holo.pulse = max(holo.pulse, min(1, d / 500))
                    arc_reactor(frame, (cx, cy), radius, holo.rotation, holo.energy)
                    cv2.line(frame, a, b, BLUE, 1, cv2.LINE_AA)
                else:
                    holo.last_two_distance = None

                if square_hold > 0 and square_rect:
                    x1, y1, x2, y2 = square_rect
                    glow = frame.copy()
                    cv2.rectangle(glow, (x1, y1), (x2, y2), CYAN, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1, cv2.LINE_AA)
                    cv2.putText(frame, "SQUARE DETECTED", (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .48, CYAN, 1, cv2.LINE_AA)
                    cv2.addWeighted(glow, .22, frame, .78, 0, frame)
                    square_hold -= 1
                elif not gestures or gestures[0] != "POINT":
                    drawing_points.clear()

                # Fingertip trail and target lock box.
                for i in range(1, len(holo.trails)):
                    cv2.line(frame, holo.trails[i - 1], holo.trails[i], BLUE, max(1, i // 6), cv2.LINE_AA)
                if holo.trails:
                    tx, ty = holo.trails[-1]
                    size = int(38 + 12 * math.sin(time.perf_counter() * 7))
                    cv2.rectangle(frame, (tx-size, ty-size), (tx+size, ty+size), CYAN, 1, cv2.LINE_AA)

                holo.update(hand_data)
                holo.draw(frame)

                # Screen-center scanning reticle gives the camera a Stark-lab feel.
                cx, cy = w // 2, h // 2
                scan = frame.copy()
                r = 80 + int(8 * math.sin(time.perf_counter() * 3))
                cv2.circle(scan, (cx, cy), r, BLUE, 1, cv2.LINE_AA)
                cv2.line(scan, (cx-r-18, cy), (cx-r+5, cy), BLUE, 1)
                cv2.line(scan, (cx+r-5, cy), (cx+r+18, cy), BLUE, 1)
                cv2.line(scan, (cx, cy-r-18), (cx, cy-r+5), BLUE, 1)
                cv2.line(scan, (cx, cy+r-5), (cx, cy+r+18), BLUE, 1)
                cv2.addWeighted(scan, .28, frame, .72, 0, frame)

                now = time.perf_counter()
                fps = 1 / max(now - prev, 1e-6)
                prev = now
                primary = gestures[0] if gestures else "NO HAND"
                hud(frame, primary, len(result.hand_landmarks), fps, holo.energy, holo.lock > .65)
                cv2.imshow("J.A.R.V.I.S. // Hand Interface", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in (ord('r'), ord('R')):
                    holo = Hologram(w, h)
                if key in (ord('b'), ord('B')):
                    holo.blast(w // 2, h // 2, 250)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
