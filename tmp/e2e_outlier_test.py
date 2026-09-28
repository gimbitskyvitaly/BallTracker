"""E2E-проверка постобработки: берём РЕАЛЬНУЮ траекторию из ответа API
(job 75884ccef424) и прогоняем через _finalize_pass как в пайплайне."""
import json, sys, math
sys.path.insert(0, '.')
from app.services.pipeline import VideoAnalysis, _finalize_pass, PassEvent
from app.services.detector import Detection
import numpy as np

traj = json.load(open('tmp/traj_user.json'))
if isinstance(traj, dict): traj = traj.get('trajectory', traj)
pts = [(84 + i, p['x_px'], p['y_px']) for i, p in enumerate(traj)]

def v_outliers(seq, thr):
    bad=[]
    for i in range(1,len(seq)-1):
        dpr=math.hypot(seq[i][1]-seq[i-1][1], seq[i][2]-seq[i-1][2])
        dn =math.hypot(seq[i][1]-seq[i+1][1], seq[i][2]-seq[i+1][2])
        dnbr=math.hypot(seq[i+1][1]-seq[i-1][1], seq[i+1][2]-seq[i-1][2])
        if min(dpr,dn)>thr and dnbr < max(dpr,dn)*0.5: bad.append(i)
    return bad

print("V-outliers на входе:", [pts[i] for i in v_outliers(pts, 30)])

an = VideoAnalysis(fps=30.0, width=1920, height=1080, n_frames=len(pts))
passer = Detection(np.array([881.,359.,1065.,741.]), 0.9, 0)
catcher = Detection(np.array([1487.,421.,1635.,724.]), 0.9, 0)
_finalize_pass(an, pts, passer, catcher, fps=30.0, ball_diam_px=12.0)
assert len(an.passes) == 1, "пас должен остаться!"
fl = an.passes[0].flight
assert fl is not None and len(fl.trajectory) > 0, "траектория не должна быть пустой!"
out = [(p.get('frame'), p['x_px'], p['y_px']) for p in fl.trajectory]
print(f"method={fl.method} tof={fl.time_of_flight_s} apex={fl.apex_height_m} dist={fl.distance_m}")
print("точек на выходе:", len(fl.trajectory), "(было", len(pts), ")")
rem = sorted(set(p[0] for p in pts) - {84+i for i,p in enumerate(fl.trajectory)}) if fl.method=='tracked_direct' else []
print("удалённые кадры (tracked_direct):", rem)
# для tracked_direct индексы совпадают с порядком
if fl.method == 'tracked_direct':
    kept = [(84+i, p['x_px'], p['y_px']) for i,p in enumerate(fl.trajectory)]
    print("оставшиеся V-выбросы:", [kept[i] for i in v_outliers(kept, 30)])
