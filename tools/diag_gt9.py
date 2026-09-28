"""Собираем ГТ-траекторию мяча (сцепка motion-треков) и прогоняем через _is_pass."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from app.config import settings
from app.services.pipeline import _is_pass, _drop_outliers

vid = "videos/VID_20260925_163505.mp4"
cap = cv2.VideoCapture(vid)
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
H, W = frames[0].shape[:2]
print("W,H:", W, H, "n:", len(frames))

def cands(i):
    g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    md = cv2.absdiff(g, cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY))
    mb = cv2.GaussianBlur(md, (0,0), 3)
    m = (mb > 25).astype(np.uint8)*255
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 40 or a > 0.01*W*H: continue
        (cx,cy),r = cv2.minEnclosingCircle(c)
        if r > 70: continue
        fill = a/(np.pi*r*r+1e-6)
        out.append((a*max(fill,0.2), cx, cy, r))
    out.sort(reverse=True)
    return out[:4]

# глобальный трек: от сида вперёд с velocity gate, затем назад; склейка разрывов <=6
def track(seed_i, seed_xy, gate=140.0):
    pos = np.array(seed_xy, float); vel = np.zeros(2)
    hits = {seed_i: pos.copy()}
    for i in range(seed_i+1, len(frames)):
        pred = pos + vel
        cs = cands(i)
        best = min(cs, key=lambda c: math.hypot(c[1]-pred[0], c[2]-pred[1])) if cs else None
        if best is None or math.hypot(best[1]-pred[0], best[2]-pred[1]) > gate:
            # одна попытка по инерции
            continue
        newpos = np.array([best[1], best[2]])
        vel = 0.5*vel + 0.5*(newpos-pos)
        pos = newpos
        hits[i] = pos.copy()
    return hits

best_hits = {}
for si in range(1, len(frames)-2):
    for c in cands(si):
        h = track(si, (c[1], c[2]))
        if len(h) > len(best_hits):
            best_hits = h
print("GT track points:", len(best_hits))
xs=[v[0] for v in best_hits.values()]; ys=[v[1] for v in best_hits.values()]
if xs: print(f"x[{min(xs):.0f},{max(xs):.0f}] y[{min(ys):.0f},{max(ys):.0f}]")
for k in sorted(best_hits)[::6]:
    print(f"  f={k:4d} x={best_hits[k][0]:7.1f} y={best_hits[k][1]:7.1f}")

# Полёт = точки вне «стояния» у игроков: берём весь трек как pts сегмента
pts = [(k, float(v[0]), float(v[1])) for k, v in sorted(best_hits.items())]
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# контакт-зона: прикинем позиции игроков из крайних точек трека
class D:
    def __init__(self, c): self.center=c; self.bbox=np.array([c[0]-60,c[1]-190,c[0]+60,c[1]+190])
passer = D(pts[0][1:]) if pts else None
catcher = D(pts[-1][1:]) if pts else None
v = _is_pass(pts, passer, catcher, W, H, fps)
print("_is_pass verdict:", v if not isinstance(v, tuple) else (v[0], f"kept {len(v[1])}/{len(pts)}"))

# заодно метрики по частям
tol = settings.outlier_tol_px if settings.outlier_tol_px>0 else max(30.0, 0.05*W)
xa = np.array([p[1] for p in pts]); ya = np.array([p[2] for p in pts])
xk, yk, keep = _drop_outliers(xa, ya, tol)
print(f"outliers dropped: {(~keep).sum()} of {len(pts)}, tol={tol:.0f}")
