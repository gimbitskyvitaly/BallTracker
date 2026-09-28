"""Ground-truth & pipeline accuracy check on the real video.

Builds an independent, motion-consistent "ball" trajectory (frame-diff
candidates + velocity prediction gate) and compares the pipeline output
against it: coverage, median/percentile position error, pass windows.

Usage: python tools/gt_check.py [video]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def _cands_at(frames, i):
    """Compact round blobs in the frame difference of frames i-1..i."""
    f = frames[i]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    md = cv2.absdiff(g, cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY))
    mb = cv2.GaussianBlur(md, (0, 0), 3)
    m = (mb > 25).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = f.shape[:2]
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 40 or a > 0.01 * W * H:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r > 70:
            continue
        fill = a / (np.pi * r * r + 1e-6)
        out.append((a * max(fill, 0.2), cx, cy, r))
    out.sort(reverse=True)
    return out[:4]


def _run_track(frames, seed_i, seed_c, gate=120.0):
    pos = np.array([seed_c[1], seed_c[2]], dtype=float)
    vel = np.zeros(2)
    hits = [(seed_i + 1, pos[0], pos[1])]
    for i in range(seed_i + 1, len(frames)):
        pred = pos + vel
        cs = _cands_at(frames, i)
        if not cs:
            continue
        best = min(cs, key=lambda c: np.hypot(c[1] - pred[0], c[2] - pred[1]))
        d = np.hypot(best[1] - pred[0], best[2] - pred[1])
        if d > gate:
            break
        newpos = np.array([best[1], best[2]], dtype=float)
        vel = 0.5 * vel + 0.5 * (newpos - pos)
        pos = newpos
        hits.append((i + 1, best[1], best[2]))
    return hits


def ground_truth(path: str, min_len: int = 15):
    """Motion-consistent trajectories: every candidate seeds a forward track;
    tracks with >= min_len consecutive (gapped<=3) hits are real trajectories.
    Returns (total_frames, [segment,...]) where segment = [(frame,x,y),...]."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    total = len(frames)

    tracks = []
    for si in range(1, total - 6):
        for c in _cands_at(frames, si):
            h = _run_track(frames, si, c)
            if len(h) >= min_len:
                tracks.append(h)

    # greedy dedup: keep longest first, drop those overlapping an kept one
    tracks.sort(key=len, reverse=True)
    chosen = []
    used = set()
    for t in tracks:
        overlap = sum(1 for fr, _, _ in t if fr in used)
        if overlap <= 0.3 * len(t):
            chosen.append(t)
            used.update(fr for fr, _, _ in t)
    chosen.sort(key=lambda t: t[0][0])
    return total, chosen


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "videos/VID_20260925_163505.mp4"
    total, segs = ground_truth(path)
    print("video frames:", total)
    print("GT segments:", len(segs))
    for s in segs:
        xs = [p[1] for p in s]
        ys = [p[2] for p in s]
        print("  fr %d-%d n=%d x:%d->%d y:%d->%d" %
              (s[0][0], s[-1][0], len(s), xs[0], xs[-1], ys[0], ys[-1]))

    from app.services.pipeline import analyze_video
    an = analyze_video(path)
    print("pipeline: track pts =", len(an.ball_track_points),
          "passes =", [(p.release_frame, p.catch_frame) for p in an.passes])

    gtmap = {}
    for s in segs:
        for fr, x, y in s:
            gtmap[fr] = (x, y)
    errs = []
    matched = 0
    for pt in an.ball_track_points:
        fr = int(pt["frame"])
        if fr in gtmap:
            gx, gy = gtmap[fr]
            e = float(np.hypot(pt["x"] - gx, pt["y"] - gy))
            errs.append(e)
            matched += 1
    errs = np.array(errs) if errs else np.array([np.nan])
    print("track coverage vs GT: %d/%d GT frames matched" % (matched, len(gtmap)))
    print("err median=%.0f p90=%.0f max=%.0f  (<30px: %.0f%%)" %
          (np.nanmedian(errs), np.nanpercentile(errs, 90), np.nanmax(errs),
           100.0 * float(np.mean(errs < 30))))


if __name__ == "__main__":
    main()
