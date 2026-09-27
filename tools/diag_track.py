import sys, math
sys.path.insert(0, ".")
import numpy as np, cv2
from app.services.detector import BallDetector
import app.services.pipeline as P

VIDEO = "videos/VID_20260925_163505.mp4"
segs = []
orig = P._is_pass
def spy(pts, passer, catcher, w, h, fps):
    segs.append((pts[0][0], pts[-1][0], len(pts)))
    return orig(pts, passer, catcher, w, h, fps)
P._is_pass = spy
det = BallDetector()
res = P.analyze_video(VIDEO, detector=det)
tp = res.ball_track_points
print("segments (f0,f1,n):", segs)
print("track pts:", len(tp))
# разрывы в треке
prev = None
for p in tp:
    if prev is not None and p["frame"] - prev["frame"] > 1:
        print(f"gap frames {prev['frame']}->{p['frame']}: ({prev['x']:.0f},{prev['y']:.0f})->({p['x']:.0f},{p['y']:.0f})")
    prev = p
# траектория по 10 точкам
for i in range(0, len(tp), 8):
    p = tp[i]; print(p["frame"], round(p["x"]), round(p["y"]))
