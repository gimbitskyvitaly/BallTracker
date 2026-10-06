"""Application settings loaded from environment variables (``.env``).

Every tunable of the pass-trajectory service lives in the ``.env`` file that
sits next to the project root (ball-time-ai style: configuration outside the
code).  Values may always be overridden by real environment variables, which
is handy for docker / CI runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    # Allow trailing inline comments: "CPU            # CPU | CUDA" -> "CPU"
    return raw.split("#", 1)[0].strip()


def _strip_comment(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = _strip_comment(raw).lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = _strip_comment(raw)
    if raw == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = _strip_comment(raw)
    if raw == "":
        return default
    return int(float(raw))


@dataclass(frozen=True)
class NetGeometryConfig:
    """Net geometry in *normalized* frame coordinates (x/W, y/H).

    A back-line camera sees the net as a horizontal band: its top cable runs
    between the left and right antennas.  The values are multiplied by the
    real video size at runtime, so the same ``.env`` works for 720p/1080p/4K.
    """

    enabled: bool           # NET_GEOMETRY_ENABLED - fall back to defaults if false
    top_y: float            # NET_TOP_Y_NORM   - normalized y of the top cable
    left_x: float           # NET_LEFT_X_NORM  - normalized x of the left antenna
    right_x: float          # NET_RIGHT_X_NORM - normalized x of the right antenna
    clearance_norm: float   # NET_CLEARANCE_NORM - extra height above tape


@dataclass(frozen=True)
class OutlierConfig:
    """Outlier rejection rule from the task statement.

    ``p_i`` is an outlier iff::

        ||p_i - p_{i-1}|| > eps  AND  ||p_i - p_{i+1}|| > eps
        AND  ||p_{i-1} - p_{i+1}|| < eps

    Such a point is a single-frame teleport: the ball moved far away for one
    frame and came back to where it was heading - typical for detection noise.
    A real fast attack/serve does *not* return to the neighbourhood of the
    previous point, so it survives the filter.
    """

    eps_ratio: float        # OUTLIER_EPS_RATIO - eps relative to frame width
    max_iterations: int     # OUTLIER_MAX_ITERATIONS - re-run passes until stable


@dataclass(frozen=True)
class PassConfig:
    """Hyperbola + over-net + towards-antenna criteria of a setting pass."""

    min_points: int                 # PASS_MIN_POINTS
    min_duration_sec: float         # PASS_MIN_DURATION_SEC
    apex_before_net_min_sec: float  # PASS_APEX_BEFORE_NET_MIN_SEC
    hyperbola_max_rmse_ratio: float # PASS_HYPERBOLA_MAX_RMSE_RATIO (of frame H)
    require_over_net: bool          # PASS_REQUIRE_OVER_NET
    near_antenna_margin_ratio: float  # PASS_NEAR_ANTENNA_MARGIN_RATIO (of W)
    max_center_offset_ratio: float    # PASS_MAX_CENTER_OFFSET_RATIO (of W)
    max_descent_after_apex: float     # PASS_MAX_DESCENT_AFTER_APEX_RATIO (of H)


@dataclass(frozen=True)
class Settings:
    # ---- input / output ----------------------------------------------------
    video_path: str
    output_dir: str
    model_path: str

    # ---- inference ---------------------------------------------------------
    confidence_threshold: float
    track_length: int
    device: str                     # CPUExecutionProvider / CUDAExecutionProvider

    # ---- tracking / track building ----------------------------------------
    fps: float                      # 0 -> take fps from the video container
    max_distance: float             # px at 1920 reference width
    max_disappeared: int            # frames without detection before closing
    min_track_points: int
    merge_gap_frames: int

    # ---- rendering -----------------------------------------------------------
    draw_all_trajectories: bool     # DRAW_ALL_TRAJECTORIES flag
    write_video: bool               # WRITE_OUTPUT_VIDEO
    trajectory_tail_frames: int
    show_per_frame_info: bool

    # ---- sub-configs ---------------------------------------------------------
    net: NetGeometryConfig = field(default_factory=lambda: NetGeometryConfig(True, 0.36, 0.21, 0.79, 0.01))
    outlier: OutlierConfig = field(default_factory=lambda: OutlierConfig(0.05, 5))
    pass_cfg: PassConfig = field(default_factory=lambda: PassConfig(6, 0.2, 0.1, 0.05, True, 0.12, 0.35, 0.8))

    # ------------------------------------------------------------------ paths
    @property
    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def resolved_model_path(self) -> Path:
        path = Path(self.model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env_file: Optional[str] = None) -> Settings:
    """Populate :class:`Settings` from ``.env`` (and process environment)."""
    dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)

    return Settings(
        video_path=_get_str("VIDEO_PATH", "examples/match.mp4"),
        output_dir=_get_str("OUTPUT_DIR", "data/output"),
        model_path=_get_str("MODEL_PATH", "models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx"),
        confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", 0.5),
        track_length=_get_int("TRACK_LENGTH", 30),
        device=_get_str("INFERENCE_DEVICE", "CPU"),
        fps=_get_float("FPS", 0.0),
        max_distance=_get_float("MAX_DISTANCE_PX", 200.0),
        max_disappeared=_get_int("MAX_DISAPPEARED_FRAMES", 40),
        min_track_points=_get_int("MIN_TRACK_POINTS", 5),
        merge_gap_frames=_get_int("MERGE_GAP_FRAMES", 40),
        draw_all_trajectories=_get_bool("DRAW_ALL_TRAJECTORIES", False),
        write_video=_get_bool("WRITE_OUTPUT_VIDEO", True),
        trajectory_tail_frames=_get_int("TRAJECTORY_TAIL_FRAMES", 45),
        show_per_frame_info=_get_bool("SHOW_PER_FRAME_INFO", True),
        net=NetGeometryConfig(
            enabled=_get_bool("NET_GEOMETRY_ENABLED", True),
            top_y=_get_float("NET_TOP_Y_NORM", 0.36),
            left_x=_get_float("NET_LEFT_X_NORM", 0.21),
            right_x=_get_float("NET_RIGHT_X_NORM", 0.79),
            clearance_norm=_get_float("NET_CLEARANCE_NORM", 0.01),
        ),
        outlier=OutlierConfig(
            eps_ratio=_get_float("OUTLIER_EPS_RATIO", 0.05),
            max_iterations=_get_int("OUTLIER_MAX_ITERATIONS", 5),
        ),
        pass_cfg=PassConfig(
            min_points=_get_int("PASS_MIN_POINTS", 6),
            min_duration_sec=_get_float("PASS_MIN_DURATION_SEC", 0.2),
            apex_before_net_min_sec=_get_float("PASS_APEX_BEFORE_NET_MIN_SEC", 0.1),
            hyperbola_max_rmse_ratio=_get_float("PASS_HYPERBOLA_MAX_RMSE_RATIO", 0.05),
            require_over_net=_get_bool("PASS_REQUIRE_OVER_NET", True),
            near_antenna_margin_ratio=_get_float("PASS_NEAR_ANTENNA_MARGIN_RATIO", 0.12),
            max_center_offset_ratio=_get_float("PASS_MAX_CENTER_OFFSET_RATIO", 0.35),
            max_descent_after_apex=_get_float("PASS_MAX_DESCENT_AFTER_APEX_RATIO", 0.8),
        ),
    )
