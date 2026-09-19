# Hand Detection

Real-time hand detection and finger counting using Python, OpenCV and MediaPipe.

## Features
- Real-time webcam hand landmarks
- Left/right hand tracking
- Finger counting
- Gesture recognition for common gestures
- Air-drawn square detection and clean square rendering
- FPS and confidence display
- Clean exit with Q / ESC

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

## Gestures

Point with your index finger and draw a closed, roughly square shape. The project detects the trajectory and renders a clean square.

- OPEN HAND
- FIST
- ONE
- TWO
- THREE
- FOUR
- FIVE

Keep the webcam permissions enabled for your Python application.
