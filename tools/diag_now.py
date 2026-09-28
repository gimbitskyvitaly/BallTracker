"""Диагностика текущего пайплайна на реальном видео: где теряется пас."""
import sys, math
sys.path.insert(0, '.')
from app.services.pipeline import analyze_video
from app.config import settings

vid = "videos/VID_20260925_163505.mp4"
a = analyze_video(vid)
print(f"fps={a.fps:.2f} size={a.width}x{a.height} frames={a.n_frames}")
print(f"ball_track_points: {len(a.ball_track_points)}")
if a.ball_track_points:
    xs=[p['x'] for p in a.ball_track_points]; ys=[p['y'] for p in a.ball_track_points]
    print(f"x range [{min(xs):.0f},{max(xs):.0f}] y range [{min(ys):.0f},{max(ys):.0f}]")
    # show every 10th point
    for p in a.ball_track_points[::10]:
        print(f"  f={p['frame']:4d} x={p['x']:7.1f} y={p['y']:7.1f}")
print(f"passes: {len(a.passes)}")
for pe in a.passes:
    print(f"  release={pe.release_frame} catch={pe.catch_frame}")
