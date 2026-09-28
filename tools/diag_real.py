"""Диагностика _is_pass на реальном видео: какие проверки отсеивают сегменты."""
import math, sys
import numpy as np
sys.path.insert(0, ".")
from app.config import settings
from app.services.detector import BallDetector
from app.services.tracker import SORTTracker
from app.services.physics import set_gravity_px
import app.services.pipeline as P
import cv2

VIDEO = "videos/VID_20260925_163505.mp4"

# перехватываем _is_pass: пишем метрики каждого сегмента и вердикт
orig_is_pass = P._is_pass
segments = []

def spy(pts, passer, catcher, width, height, fps):
    res = orig_is_pass(pts, passer, catcher, width, height, fps)
    xs = np.array([p[1] for p in pts], float); ys = np.array([p[2] for p in pts], float)
    dx_range = float(xs.max() - xs.min())
    span = abs(xs[-1] - xs[0])
    k = min(3, len(pts) - 1); dt = max(pts[k][0]-pts[0][0], 1)
    v = math.hypot(xs[k]-xs[0], ys[k]-ys[0]) / dt
    i_apex = int(np.argmin(ys))
    apex_lift = min(ys[0], ys[-1]) - float(ys[i_apex]) if 0 < i_apex < len(pts)-1 else -1
    segments.append(dict(n=len(pts), f0=pts[0][0], f1=pts[-1][0],
                        span=round(span,1), dx_range=round(dx_range,1),
                        v=round(v,2), i_apex=i_apex, apex=round(apex_lift,1),
                        passer=passer is not None, catcher=catcher is not None,
                        verdict=res))
    return res

P._is_pass = spy

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"video {W}x{H} @ {fps:.1f}, thresholds: dx_frac={settings.pass_min_dx_frac} -> {settings.pass_min_dx_frac*W:.0f}px, "
      f"no_catch_dx={settings.pass_no_catch_min_dx_frac} -> {settings.pass_no_catch_min_dx_frac*W:.0f}px, "
      f"v>={settings.pass_min_speed_px_f}, apex>={settings.pass_min_apex_px}, require_contact={settings.require_release_contact}")
cap.release()

det = BallDetector()
res = P.analyze_video(VIDEO, detector=det)
print("n_frames:", res.n_frames, "track pts:", len(res.ball_track_points), "passes:", len(res.passes))
for s in segments:
    print(s)
