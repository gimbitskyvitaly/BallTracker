import time, os
os.environ.setdefault("BT_USE_FLOW", "1")
from app.services.detector import BallDetector
from app.services.pipeline import analyze_video

t0 = time.time()
det = BallDetector()
a = analyze_video("videos/VID_20260925_163505.mp4", detector=det)
print(f"frames={a.n_frames} ball_pts={len(a.ball_track_points)} passes={len(a.passes)} dt={time.time()-t0:.1f}s")
xs = [p['x'] for p in a.ball_track_points]
ys = [p['y'] for p in a.ball_track_points]
if xs: print(f"x range [{min(xs):.0f},{max(xs):.0f}] y range [{min(ys):.0f},{max(ys):.0f}]")
# покрытие кадра метками
fr = [p['frame'] for p in a.ball_track_points]
print(f"first={fr[0]} last={fr[-1]} coverage={len(fr)/a.n_frames:.2f}")
for p in a.passes:
    print("PASS:", p.release_frame, "->", p.catch_frame, "passer_id", p.passer_id, "catcher_id", p.catcher_id)
