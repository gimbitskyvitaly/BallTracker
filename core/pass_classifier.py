"""Setting-pass classifier for ball trajectories.

A trajectory is classified as a *pass* (передача / setting pass) when all three
signatures from the task statement hold:

1. **Hyperbolic shape** - the free-flight arc fits an upside-down parabola
   ``y = c0 + c1*t + c2*t^2`` with ``c2 > 0`` (screen coordinates, y grows
   downwards) and the fit RMSE stays below a configurable fraction of the
   frame height.  A parabolic arc is the projection signature of a ballistic
   hyperbola-like flight; attacks/flat drives are too straight or descend too
   steeply to look like a gentle set.
2. **Above the net top cable** - the apex of the arc passes higher than the
   net tape (between the antennas) by at least ``NET_CLEARANCE_NORM``.
3. **Towards the left or right net edge** - after the apex the ball descends
   toward one of the antennas (a set goes to a hitter standing near the net
   edge), rather than down the middle of the court.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config.settings import PassConfig
from core.models import NetGeometry, PassEvent, Trajectory, TrackPoint


def _fit_parabola(points: List[TrackPoint]) -> Tuple[np.ndarray, float]:
    """Least-squares fit of y(t) = c0 + c1 t + c2 t^2, t in seconds.

    Time is normalised by the *mean frame spacing* (not the whole span), so
    ``t`` is expressed in frames relative to the first point; this keeps the
    fitted vertex position consistent for tracks with gaps.
    """
    t_raw = np.array([p.frame for p in points], dtype=np.float64)
    y = np.array([p.y for p in points], dtype=np.float64)
    if len(t_raw) > 1:
        dt = float(np.mean(np.diff(t_raw)))
    else:
        dt = 1.0
    if not np.isfinite(dt) or abs(dt) < 1e-9:
        dt = 1.0
    t = (t_raw - t_raw[0]) / dt  # t in "frames" units, starts at 0
    coeffs = np.polyfit(t, y, 2)
    residual = y - np.polyval(coeffs, t)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return coeffs, rmse


def _apex_of_arc(
    pts: List[TrackPoint], coeffs: np.ndarray, curvature: float
) -> Tuple[float, float, float]:
    """Return ``(t_norm, apex_frame, apex_y)`` of the flight arc.

    ``t_norm`` is the apex position on the normalised [0, 1] time axis of the
    trajectory.  A *fitted* parabola vertex is only trusted when it lies
    inside the time window; otherwise (e.g. a clipped track that starts right
    at the top of the arc) the observed highest point of the trajectory is
    used.
    """
    first, last = pts[0].frame, pts[-1].frame
    span = max(last - first, 1)
    ys = np.array([p.y for p in pts], dtype=np.float64)
    if curvature > 0 and len(pts) > 1:
        dt = float(np.mean(np.diff([p.frame for p in pts]))) or 1.0
        t_fit_frames = -coeffs[1] / (2.0 * curvature)  # in frame units
        t_norm = t_fit_frames * dt / span
        if 0.0 <= t_norm <= 1.0:
            return float(t_norm), first + t_norm * span, float(
                np.polyval(coeffs, t_fit_frames)
            )
    i = int(np.argmin(ys))  # screen y grows downwards -> min y is the apex
    t_obs = (pts[i].frame - first) / span
    return float(t_obs), float(pts[i].frame), float(ys[i])



def classify_pass(
    trajectory: Trajectory,
    fps: float,
    net: NetGeometry,
    cfg: PassConfig,
    frame_width: int,
    frame_height: int,
) -> None:
    """Fill ``is_pass`` / ``reason`` / ``features`` on *trajectory* in place."""
    pts = trajectory.points
    if len(pts) < cfg.min_points:
        trajectory.reason = f"too short ({len(pts)} < {cfg.min_points} points)"
        return

    duration_sec = trajectory.duration_frames / max(fps, 1e-6)
    if duration_sec < cfg.min_duration_sec:
        trajectory.reason = f"too fast ({duration_sec:.2f}s)"
        return

    coeffs, rmse = _fit_parabola(pts)
    curvature = float(coeffs[2])          # screen y grows downward: >0 means "upside-down" arc
    rmse_ratio = rmse / max(frame_height, 1)
    trajectory.features["parabola_rmse_ratio"] = round(rmse_ratio, 4)
    trajectory.features["curvature_px_per_norm_t2"] = round(curvature, 2)

    if curvature <= 0:
        trajectory.reason = "not hyperbolic (no apex / monotonic)"
        return
    if rmse_ratio > cfg.hyperbola_max_rmse_ratio:
        trajectory.reason = f"hyperbola fit poor (rmse={rmse_ratio:.3f}H)"
        return

    # The fitted vertex must lie strictly inside the time window and the arc
    # must be a real rise-and-fall (not a nearly straight segment): this is
    # what makes the shape "hyperbolic".  A monotonic climb/descent (attack,
    # serve flight captured between contacts) has its extremum at a boundary.
    span_frames = max(pts[-1].frame - pts[0].frame, 1)
    dt_mean = float(np.mean(np.diff([p.frame for p in pts]))) if len(pts) > 1 else 1.0
    if not np.isfinite(dt_mean) or abs(dt_mean) < 1e-9:
        dt_mean = 1.0
    t_fit_norm = (-coeffs[1] / (2.0 * curvature)) * dt_mean / span_frames
    if not (0.05 <= t_fit_norm <= 0.95):
        trajectory.reason = "not hyperbolic (apex at trajectory boundary)"
        return
    y_first, y_last = float(pts[0].y), float(pts[-1].y)
    y_vertex = float(np.polyval(coeffs, t_fit_norm * span_frames / dt_mean))
    lift = max(y_first, y_last) - y_vertex
    if lift < 0.02 * max(frame_height, 1) or lift <= 0.0:
        trajectory.reason = "not hyperbolic (arc too flat / monotonic)"
        return

    # Apex of the flight arc: fitted parabola vertex when it lies inside the
    # window, otherwise the observed highest point (clipped tracks).
    t_apex, apex_frame, apex_y = _apex_of_arc(pts, coeffs, curvature)
    apex_x = float(np.interp(t_apex, np.linspace(0, 1, len(pts)), [p.x for p in pts]))
    trajectory.features["apex_frame"] = round(apex_frame, 2)
    trajectory.features["apex_x"] = round(apex_x, 1)
    trajectory.features["apex_y"] = round(apex_y, 1)

    # --- criterion 2: over the net cable -----------------------------------
    net_top_at_apex = net.top_y_at_x(apex_x)
    above_net_px = net_top_at_apex - apex_y  # positive => higher than the tape
    trajectory.features["above_net_px"] = round(above_net_px, 1)
    if cfg.require_over_net and above_net_px < net.clearance_px:
        trajectory.reason = "does not go over the net"
        return

    # The apex must sit between start and end - a set rises then falls.
    rise_span = t_apex
    fall_span = 1.0 - t_apex
    if rise_span <= 0.05 or fall_span <= 0.05:
        trajectory.reason = "apex at trajectory boundary"
        return
    apex_before_net_sec = rise_span * duration_sec
    if apex_before_net_sec < cfg.apex_before_net_min_sec:
        trajectory.reason = "apex too close to release"
        return

    # --- criterion 3: descending towards a left/right net edge --------------
    end_pt = pts[-1]
    margin = cfg.near_antenna_margin_ratio * frame_width
    near_left = abs(end_pt.x - net.left_x) <= margin
    near_right = abs(end_pt.x - net.right_x) <= margin
    center_offset = abs(end_pt.x - net.center_x) / max(frame_width, 1)
    trajectory.features["end_x"] = round(end_pt.x, 1)
    trajectory.features["center_offset_ratio"] = round(center_offset, 3)

    if not (near_left or near_right):
        trajectory.reason = "end point not near a net edge"
        return
    if center_offset > cfg.max_center_offset_ratio:
        trajectory.reason = "end point outside the court body"
        return

    # Descent sanity: after the apex the ball must come down, but a set lands
    # softly - a plunge deeper than the configured fraction of frame height is
    # more likely an attack finishing blow.
    descent = end_pt.y - apex_y
    max_descent = cfg.max_descent_after_apex * frame_height
    trajectory.features["descent_after_apex_px"] = round(descent, 1)
    if descent > max_descent:
        trajectory.reason = "steep descent (attack/serve)"
        return

    side = "left" if near_left else "right"
    score = float(
        np.clip(1.0 - rmse_ratio / max(cfg.hyperbola_max_rmse_ratio, 1e-6), 0.0, 1.0)
    )
    trajectory.is_pass = True
    trajectory.reason = f"pass to the {side} net edge"
    trajectory.features["side"] = side
    trajectory.features["score"] = round(score, 3)

    event = PassEvent(
        track_id=trajectory.track_id,
        start_frame=trajectory.start_frame,
        apex_frame=int(round(apex_frame)),
        end_frame=trajectory.end_frame,
        side=side,
        apex_above_net_px=round(above_net_px, 1),
        score=round(score, 3),
        features=dict(trajectory.features),
    )
    trajectory.pass_event = event
