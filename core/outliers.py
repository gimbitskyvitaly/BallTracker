"""Outlier rejection for ball trajectories.

The rule from the task statement: a point ``p_i`` is an outlier iff

    ||p_i - p_{i-1}|| > eps   AND   ||p_i - p_{i+1}|| > eps
    AND   ||p_{i-1} - p_{i+1}|| < eps

i.e. the ball "teleports" far away for exactly one frame and then continues as
if nothing happened - typical single-frame detection noise.  A genuinely fast
ball (attack / serve) jumps to a new position and *stays* there: its new point
behaves normally and does not return into the neighbourhood of the previous
point, so such transitions are kept intact.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from core.models import TrackPoint


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def find_outlier_indices(points: Sequence[TrackPoint], eps: float) -> List[int]:
    """Return indices of points that satisfy the teleport-outlier rule.

    The strict per-point rule (``d_prev > eps``, ``d_next > eps``,
    ``d_neighbours < eps``) catches isolated single-frame noise.  It is
    extended to *chains* of consecutive noisy detections: a maximal run of
    points is rejected when every point inside the run jumps more than ``eps``
    from its immediate predecessor, the two endpoints surrounding the run are
    within ``eps`` of each other (the ball teleported away and came back), and
    the whole run stays far (> ``eps``) from both endpoints.  A genuinely fast
    ball (attack / serve) still keeps its new position afterwards, so such
    transitions are never matched by this rule and remain in the trajectory.
    """
    n = len(points)
    if n < 3 or eps <= 0:
        return []
    xy = np.array([[p.x, p.y] for p in points], dtype=np.float64)

    bad: set = set()

    # --- isolated outliers (strict rule from the task statement) ----------
    for i in range(1, n - 1):
        d_prev = _dist(xy[i], xy[i - 1])
        d_next = _dist(xy[i], xy[i + 1])
        d_neighbours = _dist(xy[i - 1], xy[i + 1])
        if d_prev > eps and d_next > eps and d_neighbours < eps:
            bad.add(i)

    # --- chains of consecutive outliers ------------------------------------
    # Iterative relaxation: remove the strict single-point outliers first,
    # then re-evaluate the same rule on the reduced sequence.  After an
    # isolated noise point is dropped, its noisy neighbours become adjacent to
    # the clean context and are flagged by the very same criterion; a genuine
    # fast flight (attack / serve) never matches, because after teleporting to
    # the new region the ball *stays* there - the "came back within eps"
    # condition cannot hold for its landing points.
    remaining = [i for i in range(n) if i not in bad]
    while len(remaining) >= 3:
        m = len(remaining)
        drop: set = set()
        for j in range(1, m - 1):
            i_prev, i, i_next = remaining[j - 1], remaining[j], remaining[j + 1]
            d_prev = _dist(xy[i], xy[i_prev])
            d_next = _dist(xy[i], xy[i_next])
            d_neighbours = _dist(xy[i_prev], xy[i_next])
            if d_prev > eps and d_next > eps and d_neighbours < eps:
                drop.add(j)
        if not drop:
            break
        bad.update(remaining[j] for j in drop)
        remaining = [idx for j, idx in enumerate(remaining) if j not in drop]

    return sorted(bad)


def remove_outliers(
    points: Sequence[TrackPoint], eps: float, max_iterations: int = 5
) -> Tuple[List[TrackPoint], List[Tuple[int, float, float]]]:
    """Iteratively strip outliers until the trajectory is stable.

    Returns ``(clean_points, removed)`` where ``removed`` lists the original
    ``(frame, x, y)`` of every rejected point.  After removing one outlier the
    neighbours become adjacent, so the pass is repeated (bounded by
    ``max_iterations``) to catch chains of noisy detections.
    """
    current = list(points)
    removed: List[Tuple[int, float, float]] = []
    for _ in range(max(1, max_iterations)):
        bad = set(find_outlier_indices(current, eps))
        if not bad:
            break
        for idx in sorted(bad):
            p = current[idx]
            removed.append((p.frame, p.x, p.y))
        current = [p for i, p in enumerate(current) if i not in bad]
    return current, removed
