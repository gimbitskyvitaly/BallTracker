import sys; sys.path.insert(0,".")
import cv2, numpy as np
cap = cv2.VideoCapture("videos/VID_20260925_163505.mp4")
W=int(cap.get(3)); H=int(cap.get(4))
prev=None; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    # bright-ish yellow-green (tennis ball) mask
    m1=cv2.inRange(hsv,(20,60,80),(45,255,255))
    m2=cv2.inRange(hsv,(25,40,150),(55,180,255))
    m=cv2.bitwise_or(m1,m2)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    objs=[]
    for c in cnts:
        x,y,w,h=cv2.boundingRect(c)
        a=w*h
        if a<80 or w>0.2*W or h>0.2*H: continue
        objs.append((a,x+w//2,y+h//2,w,h))
    objs.sort(reverse=True)
    print(i, [(x,y,w,h) for a,x,y,w,h in objs[:3]])
    prev=f; i+=1
cap.release()
