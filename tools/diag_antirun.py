"""Проверяет, не режет ли анти-разлёт нормальные шаги мяча (реальное видео)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from app.config import settings
from app.services.detector import BallDetector

vid = "videos/VID_20260925_163505.mp4"
det = BallDetector()
det.reset()
import cv2
cap = cv2.VideoCapture(vid)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev = None; f = 0
raw_track = {}
while True:
    ok, frame = cap.read()
    if not ok: break
    f += 1
    balls, persons = det.detect(frame)
    b = balls[0] if balls else None
    pos = None
    if b is not None:
        pos = b.center
    if pos is not None and prev is not None:
        step = math.hypot(pos[0]-prev[0], pos[1]-prev[1])
        dy_max = settings.max_jump_frac * min(W, Hh)
        dx_max = 0.6 * dy_max
        blocked = abs(pos[1]-prev[1]) > dy_max or abs(pos[0]-prev[0]) > dx_max
        if blocked and 30 <= f <= 70:
            print(f"f={f:3d} RAW jump {prev[0]:.0f},{prev[1]:.0f} -> {pos[0]:.0f},{pos[1]:.0f} step={step:.0f} BLOCKED")
    if pos is not None:
        raw_track[f] = pos
        prev = pos
cap.release()
xs=[v[0] for v in raw_track.values()]; ys=[v[1] for v in raw_track.values()]
print("raw detector track pts:", len(raw_track), f"x[{min(xs):.0f},{max(xs):.0f}] y[{min(ys):.0f},{max(ys):.0f}]")
for k in sorted(raw_track)[::8]:
    print(f"  f={k:4d} x={raw_track[k][0]:7.1f} y={raw_track[k][1]:7.1f}")
