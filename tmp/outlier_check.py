import json
import numpy as np
traj = json.load(open('tmp/traj_user.json'))
pts = [(t['t'], t['x_px'], t['y_px']) for t in traj]
T = np.array([p[0] for p in pts]); X = np.array([p[1] for p in pts]); Y = np.array([p[2] for p in pts])
rng = np.random.default_rng(0)
best_inl = None
for _ in range(3000):
    idx = rng.choice(len(pts), 6, replace=False)
    c = np.polyfit(T[idx], X[idx], 1); d = np.polyfit(T[idx], Y[idx], 2)
    rx = np.abs(np.polyval(c, T) - X); ry = np.abs(np.polyval(d, T) - Y)
    inl = (rx < 12) & (ry < 12)
    if best_inl is None or inl.sum() > best_inl.sum(): best_inl = inl
print("inliers:", int(best_inl.sum()), "of", len(pts))
out = [i for i,v in enumerate(best_inl) if not v]
print("outlier indices:", out)
for i in out: print(i, pts[i])
