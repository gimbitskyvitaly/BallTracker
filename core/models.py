"""Domain models shared across the service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point2D = Tuple[float, float]


@dataclass
class TrackPoint:
    """One ball observation in source-video pixel coordinates."""

    frame: int
    x: float
    y: float
    radius: float = 0.0


@dataclass
class NetGeometry:
    """Net geometry in *pixel* coordinates of the processed video frame."""

    top_y: float                 # y of the net top cable (average of antennas)
    left_x: float                # x of the left antenna
    right_x: float               # x of the right antenna
    clearance_px: float          # how far above the tape counts as "over net"
    enabled: bool = True

    @property
    def center_x(self) -> float:
        return (self.left_x + self.right_x) / 2.0

    @property
    def half_width(self) -> float:
        return max(1.0, abs(self.right_x - self.left_x) / 2.0)

    def top_y_at_x(self, x: float) -> float:
        """Linear interpolation of the top cable between the antennas."""
        dx = self.right_x - self.left_x
        if abs(dx) < 1e-6:
            return self.top_y
        t = min(max((x - self.left_x) / dx, 0.0), 1.0)
        return self.top_y + 0.0  # horizontal approximation for back-line view


@dataclass
class Trajectory:
    """A cleaned ball trajectory with pass-classification metadata."""

    track_id: int
    start_frame: int
    end_frame: int
    points: List[TrackPoint] = field(default_factory=list)
    removed_outliers: List[Tuple[int, Point2D]] = field(default_factory=list)
    is_pass: bool = False
    reason: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    pass_event: Optional["PassEvent"] = None

    @property
    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "is_pass": self.is_pass,
            "reason": self.reason,
            "points": [[p.frame, p.x, p.y, p.radius] for p in self.points],
            "removed_outliers": [
                {"frame": f, "x": pt[0], "y": pt[1]} for f, pt in self.removed_outliers
            ],
            "features": self.features,
        }


@dataclass
class PassEvent:
    """Detected setting pass: a hyperbolic over-net trajectory to an antenna."""

    track_id: int
    start_frame: int
    apex_frame: int
    end_frame: int
    side: str                       # "left" | "right"
    apex_above_net_px: float
    score: float
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "apex_frame": self.apex_frame,
            "end_frame": self.end_frame,
            "side": self.side,
            "apex_above_net_px": self.apex_above_net_px,
            "score": self.score,
            "features": self.features,
        }
