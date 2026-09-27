import os, json
from collections import defaultdict
from app.services.detector import BallDetector
from app.services.pipeline import analyze_video

det = BallDetector()
a = analyze_video("videos/VID_20260925_163505.mp4", detector=det)
by_frame = {p['frame']: (p['x'], p['y']) for p in a.ball_track_points}
# ground truth из прошлой сессии: мяч у левого 0-34, полёт 35-102, покой у правого
for f in list(range(1, 163, 8)):
    p = by_frame.get(f)
    print(f"{f:4d} -> {('%.0f,%.0f' % p) if p else 'None'}")
print("passes:", [(p.release_frame, p.catch_frame) for p in a.passes])
