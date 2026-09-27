import cv2, numpy as np, math
cap = cv2.VideoCapture("videos/VID_20260925_163505.mp4")
W = int(cap.get(3)); H = int(cap.get(4))
print(W, H)
# HSV-кандидаты мяча: выучим палитру по кадру 20 (мяч заведомо виден ~ (884,271))
def ball_px(frame, f):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv
ok, fr = cap.read(); ok, fr = cap.read()
for i in range(18): ok, fr = cap.read()
hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
patch = hsv[271-12:271+12, 884-12:884+12]
print("frame20 patch median HSV:", np.median(patch.reshape(-1,3), axis=0))
# сканируем кадры: ищем blob ближайший к медиане патча, размерная полоса
lo = np.array([int(np.percentile(patch[...,0],10))-8, int(np.percentile(patch[...,1],10))-30, 60])
hi = np.array([int(np.percentile(patch[...,0],90))+8, 255, 255])
print("range", lo, hi)
results = {}
idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, frame = cap.read()
    if not ok: break
    idx += 1
    if idx % 3 and idx not in (35,36,37,38,39,40,45,50,60,70,80,90,100,102,103): 
        continue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None
    for c in cnts:
        a=cv2.contourArea(c)
        if a < 20 or a > 0.01*W*H: continue
        (cx,cy),r = cv2.minEnclosingCircle(c)
        circ = a/(math.pi*r*r+1e-6)
        if circ<0.5 or 2*r < 0.006*H or 2*r > 0.06*H: continue
        if best is None or a>best[0]: best=(a,cx,cy,2*r)
    results[idx]=best
    if best: print(f"{idx:3d} GT ball {best[1]:6.0f},{best[2]:6.0f} d={best[3]:4.0f}")
    else: print(f"{idx:3d} GT none")
cap.release()
