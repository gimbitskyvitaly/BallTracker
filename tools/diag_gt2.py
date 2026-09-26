import sys; sys.path.insert(0,".")
import cv2, numpy as np
cap = cv2.VideoCapture("videos/VID_20260925_163505.mp4")
W = int(cap.get(3)); H = int(cap.get(4))
prev=None; i=0; rows=[]
while True:
    ok,f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(5,5),0)
    if prev is not None:
        d = cv2.absdiff(g, prev)
        m = (d>30).astype(np.uint8)*255
        m = cv2.dilate(m, np.ones((5,5),np.uint8))
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs=[]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            a=w*h
            if a<100 or w>0.4*W or h>0.4*H: continue
            objs.append((a,x+w//2,y+h//2,w,h))
        objs.sort(reverse=True)
        rows.append((i,objs[:3]))
    prev=g; i+=1
cap.release()
for fr,objs in rows:
    print(fr, [(x,y,w,h) for a,x,y,w,h in objs])
