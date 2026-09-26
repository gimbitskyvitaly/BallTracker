from app.services.detector import BallDetector
from app.services.pipeline import analyze_video
det = BallDetector()
a = analyze_video("videos/VID_20260925_163505.mp4", detector=det)
by_frame = {p['frame']: (p['x'], p['y']) for p in a.ball_track_points}
# GT: мяч у левого игрока 0-34, полёт 35-102, покой у правого 103+
import sys
for f in range(1, 163):
    p = by_frame.get(f)
    mark = ''
    if p:
        x,y=p
        # грубая оценка: левый игрок ~x<500? проверим по фактическим данным ниже
    print(f"{f:3d} {('%7.1f,%7.1f'%p) if p else '   None'}")
print("passes:", [(p.release_frame, p.catch_frame) for p in a.passes])
