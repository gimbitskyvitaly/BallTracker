import sys; sys.path.insert(0,".")
import numpy as np, cv2
from app.services.detector import BallDetector
import app.services.pipeline as P

VIDEO="videos/VID_20260925_163505.mp4"
segs=[]
orig=P._finalize_segment
def spy(analysis, pts, passer, catcher, pid, cid, fps, w, h):
    segs.append([(p[0],round(p[1]),round(p[2])) for p in pts])
    return orig(analysis, pts, passer, catcher, pid, cid, fps, w, h)
P._finalize_segment = spy
res=P.analyze_video(VIDEO, detector=BallDetector())
for s in segs:
    print("SEG", len(s))
    print(s)
