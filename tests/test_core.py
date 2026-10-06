"""Unit tests for the pass-trajectory service core logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.settings import PassConfig, load_settings
from core.models import NetGeometry, TrackPoint, Trajectory
from core.outliers import find_outlier_indices, remove_outliers
from core.pass_classifier import classify_pass


def _pts(coords):
    """Accept both ``(frame, (x, y))`` and ``(frame, x, y)`` tuples."""
    out = []
    for c in coords:
        if len(c) == 3:
            f, x, y = c
        else:
            (f, (x, y)) = c
        out.append(TrackPoint(frame=f, x=x, y=y))
    return out


# ------------------------------------------------------------------ outliers
def test_single_frame_teleport_is_removed():
    pts = _pts([(0, 100, 100), (1, 105, 102), (2, 900, 700), (3, 112, 108), (4, 118, 110)])
    bad = find_outlier_indices(pts, eps=64.0)
    assert bad == [2], bad


def test_fast_flight_that_does_not_return_is_kept():
    # attack/serve: ball moves fast and *stays* at the new place
    pts = _pts([(0, 100, 100), (1, 150, 120), (2, 900, 650), (3, 960, 700)])
    bad = find_outlier_indices(pts, eps=64.0)
    assert bad == []


def test_consecutive_outliers_removed_iteratively():
    pts = _pts(
        [(0, 100, 100), (1, 105, 102), (2, 900, 700), (3, 950, 720),
         (4, 112, 108), (5, 118, 110)]
    )
    clean, removed = remove_outliers(pts, eps=64.0, max_iterations=5)
    frames = [p.frame for p in clean]
    assert frames == [0, 1, 4, 5], frames
    assert sorted(f for f, _, _ in removed) == [2, 3]


def test_two_adjacent_teleports_are_not_flagged():
    # ||p_{i-1} - p_{i+1}|| is NOT < eps here -> rule must not fire
    pts = _pts([(0, 100, 100), (1, 900, 700), (2, 950, 720), (3, 112, 108)])
    bad = find_outlier_indices(pts, eps=64.0)
    assert bad == []


# ---------------------------------------------------------------------- pass
NET = NetGeometry(top_y=259.2, left_x=268.8, right_x=1011.2, clearance_px=7.2)
CFG = PassConfig(
    min_points=6,
    min_duration_sec=0.2,
    apex_before_net_min_sec=0.1,
    hyperbola_max_rmse_ratio=0.05,
    require_over_net=True,
    near_antenna_margin_ratio=0.12,
    max_center_offset_ratio=0.35,
    max_descent_after_apex=0.8,
)
W, H, FPS = 1280, 720, 30.0


def _arc(start_frame, p0, p1, lift, duration):
    coords = []
    for k in range(duration):
        t = k / (duration - 1)
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t - 4 * lift * t * (1 - t)
        coords.append((start_frame + k, x, y))
    traj = Trajectory(
        track_id=0, start_frame=start_frame, end_frame=start_frame + duration - 1,
        points=_pts([(f, x, y) for f, x, y in coords]),
    )
    return traj


def test_hyperbolic_over_net_to_right_edge_is_a_pass():
    traj = _arc(0, (384, 612), (985, 446), 396, 45)
    classify_pass(traj, FPS, NET, CFG, W, H)
    assert traj.is_pass, traj.reason
    assert traj.features["side"] == "right"
    assert traj.pass_event is not None


def test_same_arc_ending_mid_court_is_not_a_pass():
    traj = _arc(0, (384, 612), (640, 446), 396, 45)
    classify_pass(traj, FPS, NET, CFG, W, H)
    assert not traj.is_pass
    assert "net edge" in traj.reason


def test_flat_fast_attack_is_not_a_pass():
    traj = _arc(0, (128, 216), (1152, 576), 14, 12)
    classify_pass(traj, FPS, NET, CFG, W, H)
    assert not traj.is_pass


def test_under_net_arc_is_not_a_pass():
    traj = _arc(0, (384, 612), (985, 500), 60, 45)  # apex below the cable
    classify_pass(traj, FPS, NET, CFG, W, H)
    assert not traj.is_pass
    assert "over the net" in traj.reason


def test_monotonic_descent_is_not_hyperbolic():
    pts = _pts([(f, 300 + 10 * f, 200 + 8 * f) for f in range(40)])
    traj = Trajectory(track_id=0, start_frame=0, end_frame=39, points=pts)
    classify_pass(traj, FPS, NET, CFG, W, H)
    assert not traj.is_pass
    assert "hyperbolic" in traj.reason


# -------------------------------------------------------------------- config
def test_env_file_present_and_loaded():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    assert env_path.exists(), ".env must be shipped with the service"
    settings = load_settings(str(env_path))
    assert isinstance(settings.draw_all_trajectories, bool)
    assert settings.outlier.eps_ratio > 0
    assert 0 < settings.net.top_y < 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
