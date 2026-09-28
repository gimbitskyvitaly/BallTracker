"""ГТ-трек (кешированные кандидаты) -> прогон через _is_pass с логом причин отказа."""
import sys, os, math, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from app.config import settings

vid = "videos/VID_20260925_163505.mp4"
CACHE = "/tmp/cands.pkl"
if os.path.exists(CACHE):
    cands_map = pickle.load(open(CACHE, "rb"))
else:
    cap = cv2.VideoCapture(vid)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    H, W = frames[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cands_map = {}
    for i in range(1, len(frames)):
        md = cv2.absdiff(grays[i], grays[i-1])
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
        cands_map[i] = [(c[1], c[2]) for c in out[:4]]
    pickle.dump(cands_map, open(CACHE,"wb"))

def track(seed_i, seed_xy, gate=140.0):
    pos = np.array(seed_xy, float); vel = np.zeros(2)
    hits = {seed_i: pos.copy()}
    for i in range(seed_i+1, len(cands_map)+1):
        cs = cands_map.get(i, [])
        pred = pos + vel
        best = min(cs, key=lambda c: math.hypot(c[0]-pred[0], c[1]-pred[1])) if cs else None
        if best is None or math.hypot(best[0]-pred[0], best[1]-pred[1]) > gate:
            continue
        newpos = np.array([best[0], best[1]])
        vel = 0.5*vel + 0.5*(newpos-pos)
        pos = newpos
        hits[i] = pos.copy()
    return hits

best_hits = {}
for si in sorted(cands_map):
    for c in cands_map[si]:
        h = track(si, c)
        if len(h) > len(best_hits):
            best_hits = h
print("GT points:", len(best_hits))
pts = [(k, float(v[0]), float(v[1])) for k, v in sorted(best_hits.items())]
for p in pts[::6]:
    print(f"  f={p[0]:4d} x={p[1]:7.1f} y={p[2]:7.1f}")
fps = 30.19; W, H = 1920, 1080

# --- ручная копия шагов _is_pass с логом -----------------------------------
from app.services.pipeline import _drop_outliers, _median_filter
def dbg(pts, passer=None, catcher=None):
    print(f"\n=== verdict path, n={len(pts)} ===")
    if len(pts) < settings.min_flight_frames:
        print("FAIL len"); return
    xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    tol = settings.outlier_tol_px if settings.outlier_tol_px>0 else max(30.0, 0.05*W)
    alive = list(range(len(pts))); xc, yc = xs, ys
    for it in range(12):
        xk, yk, keep = _drop_outliers(xc, yc, tol)
        if bool(keep.all()): print(f"outliers converged iter={it}, dropped={len(pts)-len(alive)}"); break
        surv=[alive[i] for i,k in enumerate(keep) if k]
        if len(surv) < max(settings.min_flight_frames, 0.5*len(alive)):
            print(f"STOP: would drop >half ({len(alive)}->{len(surv)})"); break
        alive, xc, yc = surv, xk, yk
    xf, yf = xc, yc
    frames_k=[pts[i][0] for i in alive]
    dt_min = min((frames_k[j+1]-frames_k[j]) for j in range(len(frames_k)-1)) if len(frames_k)>1 else 1
    dt_s = max(dt_min/fps, 1e-3)
    g_px = settings.gravity_ratio*fps*fps
    max_step = settings.max_step_speed_px_s*dt_s + 0.5*g_px*dt_s*dt_s*settings.max_step_sigma
    steps=[math.hypot(xf[j+1]-xf[j], yf[j+1]-yf[j]) for j in range(len(xf)-1)]
    gaps=[j for j,s in enumerate(steps) if s>max_step]
    print(f"kept={len(xf)} max_step={max_step:.0f}px big_gaps={len(gaps)} allowed<{settings.max_step_gaps_allowed}")
    if gaps:
        for j in gaps[:6]:
            print(f"   gap@{j}: step={steps[j]:.0f} f {frames_k[j]}({xf[j]:.0f},{yf[j]:.0f}) -> f {frames_k[j+1]}({xf[j+1]:.0f},{yf[j+1]:.0f})")
    xs2=_median_filter(xf,5); ys2=_median_filter(yf,5)
    dx=float(np.max(xs2)-np.min(xs2))
    print(f"dx={dx:.0f} need>={settings.pass_min_dx_frac*W:.0f}")
    n=len(xs2); k=min(3,n-1); vmax=0
    for i0 in range(n-k):
        dt=max(frames_k[i0+k]-frames_k[i0],1)
        vmax=max(vmax, math.hypot(xs2[i0+k]-xs2[i0], ys2[i0+k]-ys2[i0])/dt)
    print(f"v_max={vmax:.1f}px/f need>={settings.pass_min_speed_px_f}")
    ia=int(np.argmin(ys2)); lift=max(ys2[0],ys2[-1])-ys2[ia]
    print(f"apex idx={ia}/{n-1} lift={lift:.0f}px need>={settings.pass_min_apex_px}")

dbg(pts)
from app.services.pipeline import _is_pass
class D:
    def __init__(self,c): self.center=c; self.bbox=np.array([c[0]-60,c[1]-190,c[0]+60,c[1]+190])
v=_is_pass(pts, D(pts[0][1:]), D(pts[-1][1:]), W,H,fps)
print("_is_pass full:", v if not isinstance(v,tuple) else (v[0], f"kept {len(v[1])}/{len(pts)}"))
