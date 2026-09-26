import os, sys
from app.services.detector import BallDetector
from app.services.pipeline import analyze_video
det = BallDetector()
a = analyze_video("videos/VID_20260925_163505.mp4", detector=det)
by_frame = {p['frame']: (p['x'], p['y']) for p in a.ball_track_points}
n = len(a.ball_track_points)
print("track pts:", n, "/", a.n_frames)
# покрытие по блокам кадров
tot = a.n_frames
for s in range(0, tot, 20):
    have = sum(1 for f in range(s+1, min(s+20, tot)+1) if f in by_frame)
    print(f"frames {s+1:4d}-{min(s+20,tot):4d}: {have:2d}/20")
print("passes:", [(p.release_frame, p.catch_frame) for p in a.passes])
