"""Ball trajectory tracker ported from
https://github.com/asigatchov/fast-volleyball-tracking-inference (src/ball_tracker.py).

Greedy nearest-neighbour matching with constant-velocity prediction, track
creation/closing by distance and inactivity - the same principle as upstream,
simplified to a single-ball setting and typed points.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.models import TrackPoint

LOG = logging.getLogger(__name__)


class _Track:
    def __init__(self, track_id: int, point: TrackPoint, fps: float) -> None:
        self.track_id = track_id
        self.points: List[TrackPoint] = [point]
        self.start_frame = point.frame
        self.last_frame = point.frame
        self.fps = fps
        self.prediction: Tuple[float, float] = (point.x, point.y)

    def predict_at(self, frame_number: int) -> Tuple[float, float]:
        last = self.points[-1]
        horizon = max(frame_number - last.frame, 1)
        vx = self.prediction[0] - last.x
        vy = self.prediction[1] - last.y
        return (last.x + vx * horizon, last.y + vy * horizon)

    def update(self, point: TrackPoint) -> None:
        self.points.append(point)
        self.last_frame = point.frame
        if len(self.points) >= 2:
            prev = self.points[-2]
            dt = max(1, point.frame - prev.frame)
            dx = (point.x - prev.x) / dt
            dy = (point.y - prev.y) / dt
            self.prediction = (point.x + dx, point.y + dy)
        else:
            self.prediction = (point.x, point.y)


class BallTrajectoryTracker:
    """Builds continuous ball trajectories out of per-frame detections."""

    def __init__(
        self,
        max_distance: float = 200.0,
        max_disappeared: int = 40,
        frame_width: int = 1920,
        reference_width: int = 1920,
        fps: float = 30.0,
    ) -> None:
        # Upstream scales the matching radius to the video width relative to a
        # 1920 px reference; keep the same behaviour.
        self.max_distance = max_distance * (frame_width / reference_width)
        self.max_disappeared = max_disappeared
        self.fps = fps
        self.next_id = 0
        self.tracks: Dict[int, _Track] = {}
        self.closed: List[_Track] = []

    def update(
        self, frame_number: int, detection: Optional[Tuple[float, float]], radius: float = 0.0
    ) -> None:
        # 1. Close tracks that have been invisible for too long.
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if frame_number - track.last_frame > self.max_disappeared:
                self.closed.append(track)
                del self.tracks[track_id]

        if detection is not None:
            point = TrackPoint(frame=frame_number, x=float(detection[0]), y=float(detection[1]), radius=radius)
            best_id: Optional[int] = None
            best_dist = float("inf")
            for track_id, track in self.tracks.items():
                pred = track.predict_at(frame_number)
                dist = float(np.hypot(pred[0] - point.x, pred[1] - point.y))
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None and best_dist <= self.max_distance:
                self.tracks[best_id].update(point)
            else:
                new_track = _Track(self.next_id, point, self.fps)
                self.tracks[self.next_id] = new_track
                LOG.debug("Track %s opened at frame %s", self.next_id, frame_number)
                self.next_id += 1

    def finalize(self) -> List[List[TrackPoint]]:
        """Close all remaining tracks and return every trajectory's points."""
        for track in list(self.tracks.values()):
            self.closed.append(track)
        self.tracks.clear()
        return [t.points for t in sorted(self.closed, key=lambda t: t.start_frame)]
