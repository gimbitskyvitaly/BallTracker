"""Video rendering of ball trajectories (ball-time-ai style overlay).

The ``DRAW_ALL_TRAJECTORIES`` flag decides what is drawn:

* ``true``  - every cleaned ball trajectory; passes highlighted in green,
              ordinary flight segments in gray.
* ``false`` - only the trajectories classified as setting passes (green),
              everything else stays untouched.

Additionally the current ball marker, a fading tail, removed outliers (red
crosses, once) and the configured net geometry are overlaid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config.settings import Settings
from core.models import NetGeometry, Trajectory

LOG = logging.getLogger(__name__)

COLOR_PASS = (0, 220, 0)        # BGR - green: setting pass
COLOR_OTHER = (160, 160, 160)   # BGR - gray: non-pass trajectory
COLOR_BALL = (0, 165, 255)      # BGR - orange: current ball position
COLOR_OUTLIER = (0, 0, 255)     # BGR - red: rejected outlier
COLOR_NET = (255, 200, 0)       # BGR - cyan-ish: net geometry


class _FrameIndex:
    """Lookup: frame number -> trajectory points active on that frame."""

    def __init__(self, trajectories: List[Trajectory], fps: float, tail_frames: int) -> None:
        self.by_frame: Dict[int, List[Tuple[Trajectory, int]]] = {}
        for traj in trajectories:
            for pos, point in enumerate(traj.points):
                self.by_frame.setdefault(point.frame, []).append((traj, pos))
        self.fps = fps
        self.tail = max(1, tail_frames)

    def color_for(self, traj: Trajectory):
        return COLOR_PASS if traj.is_pass else COLOR_OTHER

    def draw_at(self, canvas: np.ndarray, frame_no: int, draw_all: bool) -> None:
        entries = self.by_frame.get(frame_no, [])
        # Tail: polyline over the last `tail` points of each active trajectory.
        for traj, pos in entries:
            if not draw_all and not traj.is_pass:
                continue
            start = max(0, pos - self.tail)
            pts = traj.points[start:pos + 1]
            if len(pts) >= 2:
                coords = np.array([[int(p.x), int(p.y)] for p in pts], dtype=np.int32)
                cv2.polylines(canvas, [coords], False, self.color_for(traj), 2, cv2.LINE_AA)
            point = traj.points[pos]
            radius = int(max(point.radius, 4))
            cv2.circle(canvas, (int(point.x), int(point.y)), radius, COLOR_BALL, 2, cv2.LINE_AA)

    def draw_full_pass(self, canvas: np.ndarray, traj: Trajectory) -> None:
        coords = np.array([[int(p.x), int(p.y)] for p in traj.points], dtype=np.int32)
        cv2.polylines(canvas, [coords], False, COLOR_PASS, 2, cv2.LINE_AA)


def _draw_net(canvas: np.ndarray, net: NetGeometry, height: int) -> None:
    if not net.enabled:
        return
    y = int(net.top_y)
    x1, x2 = int(net.left_x), int(net.right_x)
    cv2.line(canvas, (x1, y), (x2, y), COLOR_NET, 2, cv2.LINE_AA)
    for antenna_x in (x1, x2):
        cv2.line(canvas, (antenna_x, y), (antenna_x, min(height - 1, y + 40)), COLOR_NET, 2)
        cv2.drawMarker(canvas, (antenna_x, y), COLOR_NET, cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(canvas, "NET TOP", (x1 + 8, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 1, cv2.LINE_AA)


def _draw_outliers(canvas: np.ndarray, trajectories: List[Trajectory]) -> None:
    for traj in trajectories:
        for frame, (x, y) in traj.removed_outliers:
            cv2.drawMarker(canvas, (int(x), int(y)), COLOR_OUTLIER,
                           cv2.MARKER_CROSS, 12, 2)


def _draw_hud(
    canvas: np.ndarray,
    frame_no: int,
    index: _FrameIndex,
    settings: Settings,
    trajectories: List[Trajectory],
) -> None:
    if not settings.show_per_frame_info:
        return
    h, w = canvas.shape[:2]
    active = index.by_frame.get(frame_no, [])
    lines = [f"frame {frame_no} | tracks active: {len(active)}"]
    for traj, _ in active:
        tag = f"#{traj.track_id} {'PASS' if traj.is_pass else 'flight'}: {traj.reason}"
        lines.append(tag[: w // 8])
    pass_events = [t.pass_event for t in trajectories if t.pass_event]
    for ev in pass_events:
        if ev.start_frame <= frame_no <= ev.end_frame:
            lines.append(
                f"PASS #{ev.track_id} -> {ev.side} edge  score={ev.score:.2f}  "
                f"apex +{ev.apex_above_net_px:.0f}px over net"
            )
    y = 24
    for line in lines[:8]:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 22


def render_video(
    video_path: str,
    output_path: Path,
    trajectories: List[Trajectory],
    settings: Settings,
    net: NetGeometry,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height)
    )

    index = _FrameIndex(trajectories, fps, settings.trajectory_tail_frames)
    draw_all = settings.draw_all_trajectories
    pass_trajs = [t for t in trajectories if t.is_pass]
    outlier_drawn = False

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame

        # Persistent full pass paths so the completed arc stays visible.
        for traj in pass_trajs:
            if traj.end_frame < frame_no <= traj.end_frame + int(fps * 1.5):
                index.draw_full_pass(canvas, traj)
        if draw_all:
            for traj in trajectories:
                if not traj.is_pass and traj.end_frame < frame_no <= traj.end_frame + int(fps * 0.7):
                    index.draw_full_pass(canvas, traj)

        index.draw_at(canvas, frame_no, draw_all)
        _draw_net(canvas, net, height)
        if not outlier_drawn and frame_no >= 1:
            _draw_outliers(canvas, trajectories)
            outlier_drawn = True
        _draw_hud(canvas, frame_no, index, settings, trajectories)

        writer.write(canvas)
        frame_no += 1

    cap.release()
    writer.release()
    LOG.info("Rendered %d frames -> %s", frame_no, output_path)
