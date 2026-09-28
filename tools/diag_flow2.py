"""Качество кандидатов RobustFlowBall на реальном видео (с кэшем кадров)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, pickle
from app.services.pipeline import RobustFlowBall
from app.config import settings

vid = "videos/VID_20260925_163505.mp4"
CACHE="/tmp/frames_small.pkl"
if os.path.exists(CACHE):
    frames = pickle.load(open(CACHE,'rb'))
else:
    cap=cv2.VideoCapture(vid); frames=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        frames.append(cv2.resize(f,None,fx=0.5,fy=0.5))
    cap.release(); pickle.dump(frames, open(CACHE,'wb'))
print("cached frames:", len(frames), frames[0].shape)
rf = RobustFlowBall(settings and 1920 or 1920, 1080)
n_with=0
for i,fr in enumerate(frames, start=1):
    cands = rf.update(fr)
    if cands: n_with+=1
    if i%10==0:
        print(f"f={i:4d} cands={[(round(c[0]),round(c[1])) for c in cands[:4]]}")
print("frames with candidates:", n_with, "/", len(frames))
