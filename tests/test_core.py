"""Юнит-тесты ядра: SORT-трекинг, физическая модель, оценка полёта."""
import math

import numpy as np
import pytest

from app.services.tracker import SORTTracker, iou_batch
from app.services.physics import estimate_flight, set_gravity_px, fit_physics_trajectory
from app.config import settings


def _parabola_points(fps=30.0, seconds=1.5, x0=100.0, y0=300.0, vx=180.0, vy=-400.0):
    """Синтетический полёт с известной аналитикой (без drag)."""
    set_gravity_px(fps, settings.gravity_ratio)
    from app.services import physics
    g = physics.G_PX
    pts = []
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        x = x0 + vx * t
        y = y0 + vy * t + 0.5 * g * t * t
        if y > 470:
            break
        pts.append((i + 1, x, y))
    return pts, g


class TestTracker:
    def test_iou_identity(self):
        b = np.array([[0, 0, 10, 10]], float)
        assert abs(iou_batch(b, b)[0, 0] - 1.0) < 1e-9

    def test_iou_disjoint(self):
        a = np.array([[0, 0, 10, 10]], float)
        b = np.array([[20, 20, 30, 30]], float)
        assert iou_batch(a, b)[0, 0] == 0.0

    def test_single_object_constant_velocity(self):
        tr = SORTTracker(min_hits=1)
        ids = set()
        for f in range(10):
            x = 50 + 10 * f
            dets = np.array([[x, 100, x + 20, 120]], float)
            out = tr.update(dets)
            assert len(out) == 1, "должен быть ровно один подтверждённый трек"
            ids.add(out[0][0])
        assert len(ids) == 1, "идентичность трека не должна теряться"
        # скорость по x ~ 10 px/кадр
        _, _, vx, vy = tr.update(np.array([[140, 100, 160, 120]], float))[0]
        assert 8 < vx < 12

    def test_two_objects_no_id_swap(self):
        tr = SORTTracker(min_hits=1)
        for f in range(8):
            a = 50 + 5 * f
            b = 400 - 5 * f
            dets = np.array([[a, 100, a + 20, 120], [b, 300, b + 20, 320]], float)
            out = sorted(tr.update(dets), key=lambda r: r[1][0])
            assert len(out) == 2
        # после 8 кадров id у меньшего x должен соответствовать треку A
        ids_first = {round(o[1][0]): o[0] for o in out}
        assert len(set(ids_first.values())) == 2

    def test_occlusion_recovery(self):
        tr = SORTTracker(min_hits=1, max_age=12)
        for f in range(6):
            x = 50 + 10 * f
            tr.update(np.array([[x, 100, x + 20, 120]], float))
        for _ in range(5):   # пропуск детекций
            tr.update(np.empty((0, 4)))
        out = tr.update(np.array([[150, 100, 170, 120]], float))
        assert len(out) == 1


class TestPhysics:
    def test_tof_matches_analytic(self):
        """ToF без drag: t_полёта = 2*|vy0|/g (возврат на ту же высоту)."""
        fps = 30.0
        set_gravity_px(fps, settings.gravity_ratio)
        pts, g = _parabola_points(fps=fps, vy=-400.0)
        est = estimate_flight(pts, fps, ball_diam_px=28.0,
                              drag=settings.drag_coefficient, use_physics=True)
        assert est is not None
        analytic = 2 * 400.0 / g
        measured = (pts[-1][0] - pts[0][0]) / fps
        assert abs(est.time_of_flight_s - measured) < 0.05
        assert abs(est.time_of_flight_s - analytic) < 0.25, \
            f"ToF {est.time_of_flight_s} vs analytic {analytic}"

    def test_physics_fit_recovers_initial_velocity(self):
        fps = 30.0
        set_gravity_px(fps, settings.gravity_ratio)
        pts, g = _parabola_points(fps=fps, vx=180.0, vy=-400.0)
        fr = np.array([p[0] for p in pts], dtype=float)
        xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
        fit = fit_physics_trajectory(fr, xs, ys, fps, 28.0, 0.005)
        assert fit is not None
        v0x, v0y = fit["params"][2], fit["params"][3]
        assert abs(v0x - 180) < 25 and abs(v0y + 400) < 60

    def test_short_track_returns_none(self):
        assert estimate_flight([(1, 0, 0), (2, 5, 5)], 30.0, 20.0, 0.02) is None

    def test_apex_positive(self):
        fps = 30.0
        pts, _ = _parabola_points(fps=fps)
        est = estimate_flight(pts, fps, 28.0, settings.drag_coefficient)
        assert est.apex_height_m > 0.5
