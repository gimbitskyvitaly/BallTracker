import sys; sys.path.insert(0,".")
import cv2, numpy as np
cap = cv2.VideoCapture("videos/VID_20260925_163505.mp4")
W=int(cap.get(3)); H=int(cap.get(4))
prev=None; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); g=cv2.GaussianBlur(g,(5,5),0)
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    col=cv2.inRange(hsv,(20,40,80),(55,255,255))
    col=cv2.morphologyEx(col,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    if prev is not None and 115<=i<=161:
        d=cv2.absdiff(g,prev); m=(d>30).astype(np.uint8)*255
        m=cv2.dilate(m,np.ones((7,7),np.uint8))
        inter=cv2.bitwise_and(m,col)
        cnts,_=cv2.findContours(inter,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        objs=[]
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            a=w*h
            if a<100 or w>0.15*W or h>0.15*H: continue
            (cx,cy),rad=cv2.minEnclosingCircle(c)
            circ=a/(np.pi*rad*rad+1e-6)
            objs.append((circ,a,x+w//2,y+h//2,w,h))
        objs=[o for o in objs if 0.7<=o[0] and 12<=o[4]<=60 and 12<=o[5]<=60]
        objs.sort(reverse=True)
        print(i,[(x,y,w,h,round(c,2)) for c,a,x,y,w,h in objs[:4]])
    prev=g; i+=1
cap.release()
