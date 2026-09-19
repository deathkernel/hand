import time
from typing import List

import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, handedness: str) -> int:
    lm = hand_landmarks.landmark
    fingers = 0

    # Thumb: compare x direction based on hand side.
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            fingers += 1
    else:
        if lm[4].x > lm[3].x:
            fingers += 1

    # Index, middle, ring, pinky: fingertip is above PIP.
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    return fingers


def gesture_name(count: int) -> str:
    names = {
        0: "FIST",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "OPEN HAND",
    }
    return names.get(count, "UNKNOWN")


def draw_panel(frame, hands_info: List[str], fps: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 125), (20, 20, 20), -1)
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = time.perf_counter()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            infos: List[str] = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    side = handedness_info.classification[0].label
                    confidence = handedness_info.classification[0].score
                    count = count_fingers(hand_landmarks, side)
                    gesture = gesture_name(count)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(thickness=2),
                    )

                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    x, y = int(wrist.x * w), int(wrist.y * h)

                    cv2.putText(
                        frame,
                        f"{side}: {gesture}",
                        (max(10, x - 60), max(30, y - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )

                    infos.append(
                        f"{side}: {count} fingers | {confidence * 100:.0f}%"
                    )

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            draw_panel(frame, infos, fps)
            cv2.putText(
                frame,
                "Q / ESC to quit",
                (18, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )

            cv2.imshow("Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
