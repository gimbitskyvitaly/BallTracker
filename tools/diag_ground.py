import sys; sys.path.insert(0, ".")
import cv2, numpy as np
cap = cv2.VideoCapture("videos/VID_20260925_163505.mp4")
fps = cap.get(cv2.CAP_PROP_FPS); W = int(cap.get(3)); H = int(cap.get(4))
prev = None; i = 0
hits = []
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        d = cv2.absdiff(g, prev)
        m = (d > 25).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w*h < 40 or max(w,h) > 0.5*W: continue
            hits.append((i, x+w//2, y+h//2, w, h))
    prev = cv2.GaussianBlur(g, (5,5), 0)
    i += 1
cap.release()
# сгруппировать по кадрам: показать крупные движущиеся объекты (мяч летит быстро -> большие смещения)
byframe = {}
for fr,x,y,w,h in hits: byframe.setdefault(fr, []).append((x,y,w,h))
for fr in sorted(byframe):
    objs = byframe[fr]
    big = [o for o in objs if o[2]*o[3] > 200]
    print(fr, [(x,y,w,h) for x,y,w,h in big][:4])
