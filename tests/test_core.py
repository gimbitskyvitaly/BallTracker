"""Юнит-тесты ядра: физическая модель, детекция пасов (параболы к сетке), рендер.

Сценарии «пас vs не пас» собираются синтетически:
  * пас     — баллистический полёт ПО ГОРИЗОНТАЛИ К СЕТКЕ (центр кадра);
  * не пас  — тот же полёт ОТ сетки (атака/отбой) или движение вместе с
              игроком (ведение мяча — не парабола свободного падения).
"""
import math

import numpy as np
import pytest

from app.services.physics import estimate_flight, set_gravity_px, fit_physics_trajectory
from app.services.pass_detector import (detect_passes, split_rallies,
                                        find_parabolic_segments, classify_direction)
from app.config import settings

FPS = 30.0
W, H = 640, 480
G_PX_S2 = FPS * FPS * 0.55          # эмпирическая гравитация для синтетики


def _flight_points(f0, x0, y0, vx, vy0, n, jitter=0.0, rng=None):
    """Точки баллистического полёта: x=x0+vx*t, y=y0+vy0*t+0.5*g*t^2 (y вниз)."""
    pts = []
    for i in range(n):
        t = (f0 + i - f0) / FPS
        x = x0 + vx * t + (rng.normal(0, jitter) if jitter and rng else 0)
        y = y0 + vy0 * t + 0.5 * G_PX_S2 * t * t \
            + (rng.normal(0, jitter) if jitter and rng else 0)
        pts.append((f0 + i, float(x), float(y)))
    return pts


def _params(**kw):
    base = dict(rally_gap_frames=settings.rally_gap_frames,
                par_min_frames=settings.par_min_frames,
                par_max_frames=settings.par_max_frames,
                par_max_rmse_frac=settings.par_max_rmse_frac,
                gravity_px_s2=G_PX_S2, grav_tol_rel=settings.grav_tol_rel,
                min_flight_frames=settings.min_flight_frames,
                min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
    base.update(kw)
    return base


class TestPhysics:
    def _parabola(self, seconds=1.5, x0=100.0, y0=300.0, vx=180.0, vy=-400.0):
        g = G_PX_S2
        pts = []
        n = int(seconds * FPS)
        for i in range(n):
            t = i / FPS
            x = x0 + vx * t
            y = y0 + vy * t + 0.5 * g * t * t
            if y > 470:
                break
            pts.append((i + 1, x, y))
        return pts

    def test_tof_matches_analytic(self):
        pts = self._parabola(vy=-400.0)
        est = estimate_flight(pts, FPS, ball_diam_px=28.0,
                              drag=settings.drag_coefficient, use_physics=True)
        assert est is not None
        analytic = 2 * 400.0 / G_PX_S2
        measured = (pts[-1][0] - pts[0][0]) / FPS
        assert abs(est.time_of_flight_s - measured) < 0.05
        assert abs(est.time_of_flight_s - analytic) < 0.25

    def test_physics_fit_recovers_initial_velocity(self):
        pts = self._parabola(vx=180.0, vy=-400.0)
        fr = np.array([p[0] for p in pts], float)
        xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
        fit = fit_physics_trajectory(fr, xs, ys, FPS, 28.0, 0.005)
        assert fit is not None
        v0x, v0y = fit["params"][2], fit["params"][3]
        assert abs(v0x - 180) < 25 and abs(v0y + 400) < 60

    def test_short_track_returns_none(self):
        assert estimate_flight([(1, 0, 0), (2, 5, 5)], FPS, 20.0, 0.02) is None

    def test_apex_positive(self):
        pts = self._parabola()
        est = estimate_flight(pts, FPS, 28.0, settings.drag_coefficient)
        assert est.apex_height_m > 0.5


class TestRallySegmentation:
    def test_split_by_visibility_gaps(self):
        pts = [(1, 10, 10), (2, 12, 11), (3, 14, 12),      # розыгрыш 1
               (30, 100, 100), (31, 105, 99)]              # розыгрыш 2 (разрыв 27)
        rallies = split_rallies(pts, gap_frames=15)
        assert len(rallies) == 2
        assert [r[0][0] for r in rallies] == [1, 30]

    def test_empty_input(self):
        assert split_rallies([], 15) == []


class TestParabolicSegments:
    def test_finds_clean_ballistic_arc(self):
        pts = _flight_points(1, 500, 300, vx=-260, vy0=-350, n=25)
        segs = find_parabolic_segments(pts, FPS, gap_break=15,
                                       frame_small=min(W, H), frame_width=W,
                                       min_frames=settings.par_min_frames,
                                       max_frames=settings.par_max_frames,
                                       max_rmse_frac=settings.par_max_rmse_frac,
                                       gravity_px_s2=G_PX_S2,
                                       grav_tol_rel=settings.grav_tol_rel,
                                       min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
        assert len(segs) == 1
        s = segs[0]
        assert s.start_frame == 1 and s.end_frame >= 20
        assert s.vx_px_s < 0                       # летит влево
        assert s.direction == "left"               # и именно К сетке (старт справа)

    def test_jittered_arc_still_found(self):
        rng = np.random.default_rng(3)
        pts = _flight_points(1, 500, 300, vx=-260, vy0=-350, n=25,
                             jitter=2.0, rng=rng)
        segs = find_parabolic_segments(pts, FPS, gap_break=15,
                                       frame_small=min(W, H), frame_width=W,
                                       min_frames=settings.par_min_frames,
                                       max_frames=settings.par_max_frames,
                                       max_rmse_frac=settings.par_max_rmse_frac,
                                       gravity_px_s2=G_PX_S2,
                                       grav_tol_rel=settings.grav_tol_rel,
                                       min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
        assert len(segs) >= 1

    def test_carried_ball_is_not_parabolic(self):
        """Мяч, движущийся вместе с игроком (равномерно, без g) — не пас."""
        pts = [(1 + i, 100.0 + 3 * i, 300.0 + 1.5 * i) for i in range(30)]
        segs = find_parabolic_segments(pts, FPS, gap_break=15,
                                       frame_small=min(W, H), frame_width=W,
                                       min_frames=settings.par_min_frames,
                                       max_frames=settings.par_max_frames,
                                       max_rmse_frac=settings.par_max_rmse_frac,
                                       gravity_px_s2=G_PX_S2,
                                       grav_tol_rel=settings.grav_tol_rel,
                                       min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
        assert segs == []


class TestDirectionClassification:
    def test_left_to_net(self):
        pts = [(1, 500, 300), (10, 400, 250), (20, 340, 300)]
        d, ratio = classify_direction(pts, W)
        assert d == "left" and ratio < 0.85

    def test_right_to_net(self):
        pts = [(1, 100, 300), (10, 200, 250), (20, 280, 300)]
        d, _ = classify_direction(pts, W)
        assert d == "right"

    def test_away_from_net(self):
        pts = [(1, 340, 300), (10, 450, 250), (20, 560, 300)]
        d, _ = classify_direction(pts, W)
        assert d == "away"


class TestDetectPasses:
    def test_pass_vs_nonpass_in_one_rally(self):
        """В одном розыгрыше: пас К сетке детектится, полёт ОТ сетки — нет."""
        pass_pts = _flight_points(1, 520, 320, vx=-280, vy0=-320, n=26)
        rallies, passes = detect_passes(pass_pts, FPS, W, H, **_params())
        assert len(rallies) == 1
        assert len(passes) == 1
        p = passes[0]
        assert p.direction == "left"
        assert p.segment.to_net_ratio < 0.85

        attack_pts = _flight_points(1, 330, 220, vx=300, vy0=-150, n=26)
        _, passes2 = detect_passes(attack_pts, FPS, W, H, **_params())
        assert passes2 == [], "полёт от сетки (атака) пасом не считается"

    def test_two_passes_both_directions(self):
        a = _flight_points(1, 520, 320, vx=-280, vy0=-320, n=26)       # влево к сетке
        b = _flight_points(60, 120, 320, vx=280, vy0=-320, n=26)       # вправо к сетке
        rallies, passes = detect_passes(a + b, FPS, W, H, **_params())
        assert len(passes) == 2
        assert {p.direction for p in passes} == {"left", "right"}
        assert passes[0].release_frame < passes[1].release_frame

    def test_rally_break_on_long_invisibility(self):
        a = _flight_points(1, 520, 320, vx=-280, vy0=-320, n=26)
        b = _flight_points(120, 120, 320, vx=280, vy0=-320, n=26)   # разрыв ~68 кадров
        rallies, passes = detect_passes(a + b, FPS, W, H, **_params())
        assert len(rallies) == 2
        assert len(passes) == 2
        assert rallies[0].end_frame < rallies[1].start_frame

    def test_noise_only_track_yields_nothing(self):
        rng = np.random.default_rng(11)
        pts = [(1 + i, float(rng.uniform(0, W)), float(rng.uniform(0, H)))
               for i in range(200)]
        _, passes = detect_passes(pts, FPS, W, H, **_params())
        assert passes == []


class TestRenderer:
    """Отрисовка траекторий пасов поверх видео."""

    def _make_analysis(self):
        from app.services.pipeline import VideoAnalysis, PassEvent
        from app.services.pass_detector import ParabolicSegment
        from app.services.physics import FlightEstimate
        pts = _flight_points(31, 520, 320, vx=-280, vy0=-320, n=40)
        traj = [{"t": round(i / FPS, 3), "x_m": x * 0.01, "y_m": y * 0.01,
                 "x_px": x, "y_px": y} for i, (f, x, y) in enumerate(pts)]
        fl = FlightEstimate(time_of_flight_s=1.3, release_frame=31, catch_frame=70,
                            apex_height_m=1.5, distance_m=9.0, initial_speed_mps=6.0,
                            peak_speed_mps=7.0, fit_rmse_px=1.0, trajectory=traj,
                            method="tracked_direct")
        an = VideoAnalysis(fps=FPS, width=W, height=H, n_frames=100)
        an.ball_track_points = [{"frame": f, "x": round(x, 1), "y": round(y, 1)}
                                for f, x, y in pts]
        seg = ParabolicSegment(start_frame=31, end_frame=70, points=pts,
                               vx_px_s=-280, vy0_px_s=-320, ay_px_s2=G_PX_S2,
                               rmse_px=1.0, apex_y_px=150, direction="left",
                               to_net_ratio=0.4)
        an.passes = [PassEvent(release_frame=31, catch_frame=70, direction="left",
                               to_net_ratio=0.4, rmse_px=1.0, points_px=pts,
                               flight=fl, rally_index=0)]
        return an, pts

    def test_draw_flight_overlays_pixels(self):
        from app.services.renderer import FlightOverlay, draw_flight
        an, pts = self._make_analysis()
        frame = np.full((H, W, 3), 60, np.uint8)
        before = int((frame != 60).sum())
        ov = FlightOverlay(0, 31, 70, pts, {"time_of_flight_s": 1.3},
                           direction="left")
        draw_flight(frame, ov, 6, 25, fid=50)
        colored = int((frame != 60).sum())
        assert colored - before > 500, "оверлей должен закрасить заметную область"
        assert (frame[:, :, 2] > 100).sum() > 100

    def test_label_contains_direction(self):
        from app.services.renderer import FlightOverlay
        pts = [(1, 10.0, 10.0), (2, 20.0, 15.0)]
        assert "LEFT" in FlightOverlay(0, 1, 2, pts, direction="left").label
        assert "RIGHT" in FlightOverlay(1, 1, 2, pts, direction="right").label

    def test_render_produces_video_with_trails(self, tmp_path):
        import cv2, os
        from app.services.renderer import render_tracked_video
        src = str(tmp_path / "in.mp4"); dst = str(tmp_path / "out.mp4")
        rng = np.random.default_rng(7)
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        assert vw.isOpened()
        for i in range(140):
            base = np.full((H, W, 3), 60, np.uint8)
            noise = rng.integers(0, 25, (H, W, 1), dtype=np.uint8)
            vw.write(np.clip(base + noise, 0, 255).astype(np.uint8))
        vw.release()
        an, pts = self._make_analysis()
        out = render_tracked_video(src, dst, an, tail_hold_frames=24)
        assert os.path.exists(out) and os.path.getsize(out) > 0

        ca, cb = cv2.VideoCapture(src), cv2.VideoCapture(dst)
        n = 0; flight_colored = 0; clean_ok = True
        while True:
            oka, fa = ca.read(); okb, fb = cb.read()
            if not (oka and okb):
                break
            n += 1
            diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
            strong = int((diff > 40).sum())
            if 40 <= n <= 70 and strong > 200:
                flight_colored += 1
            if n >= 100 and strong > 200:      # эхо держится 24 кадра после приёмки
                clean_ok = False
        ca.release(); cb.release()
        assert n == 140, "длина результата должна совпадать с исходником"
        assert flight_colored >= 20, "в кадрах полёта должен формироваться след"
        assert clean_ok, "спустя хвост удержания кадры должны быть чистыми"

    def test_point_at_time_interpolation(self):
        from app.services.renderer import _point_at_time
        pts = [(1, 0.0, 0.0), (3, 20.0, 10.0)]
        assert _point_at_time(pts, 2.0) == (10.0, 5.0)
        assert _point_at_time(pts, 0.5) is None
        assert _point_at_time(pts, 3.0) == (20.0, 10.0)

    # ---------------- флаг render_mode: "passes" | "full" -------------------
    def _make_full_analysis(self):
        """Анализ с длинным треком (ведение + пас «к сетке» + полёт «от сетки»)
        и готовым fit_diagnostics для режима full."""
        from app.services.pipeline import VideoAnalysis
        from app.services.pass_detector import raw_fit_diagnostics
        an, _ = self._make_analysis()
        carry = [(f, 80.0 + 0.4 * f, 380.0 + 6 * np.sin(f / 4)) for f in range(1, 31)]
        away = _flight_points(71, 120, 250, vx=+240, vy0=-260, n=30)
        track = sorted(carry + [(p["frame"], p["x"], p["y"])
                                for p in an.ball_track_points] + away)
        an.ball_track_points = [{"frame": int(f), "x": float(x), "y": float(y)}
                                for f, x, y in track]
        an.n_frames = 140
        fits = raw_fit_diagnostics([(p["frame"], p["x"], p["y"]) for p in an.ball_track_points],
                                   FPS, min_frames=settings.par_min_frames,
                                   max_frames=settings.par_max_frames,
                                   gap_break=settings.rally_gap_frames,
                                   gravity_px_s2=G_PX_S2,
                                   grav_tol_rel=settings.grav_tol_rel,
                                   max_rmse_frac=settings.par_max_rmse_frac,
                                   frame_small=min(W, H), frame_width=W,
                                   min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
        for i, ft in enumerate(fits):
            ft["fit_index"] = i
        an.fit_diagnostics = {"render_mode": "full", "total_raw_fits": len(fits),
                              "total_would_pass": sum(f["would_pass_filters"] for f in fits),
                              "coverage_px": {"track_frames": len(track),
                                              "ballistic_frames": 0, "coverage": 0.5},
                              "rallies": [], "fits": fits}
        return an

    def test_raw_fit_diagnostics_no_filters(self):
        """Фиты без фильтров находятся даже там, где фильтры пасов их отсекают
        (проверка гипотезы: баллистический участок есть, но не проходит пороги)."""
        from app.services.pass_detector import raw_fit_diagnostics
        # дуга с завышенным ускорением (завалит gravity-фильтр) — в обычном
        # поиске сегментов её нет, в диагностике — есть и помечена failed_checks
        bad_g = [(f, 100.0 + 200 * (f - 1) / FPS,
                  300.0 - 300 * (f - 1) / FPS + 0.5 * G_PX_S2 * 3.0 * ((f - 1) / FPS) ** 2)
                 for f in range(1, 26)]
        segs = find_parabolic_segments(bad_g, FPS,
                                       min_frames=settings.par_min_frames,
                                       max_frames=settings.par_max_frames,
                                       max_rmse_frac=settings.par_max_rmse_frac,
                                       gravity_px_s2=G_PX_S2,
                                       grav_tol_rel=settings.grav_tol_rel,
                                       gap_break=15,
                                       min_horizontal_disp_frac=settings.min_horizontal_disp_frac,
                                       frame_small=min(W, H), frame_width=W)
        diag = raw_fit_diagnostics(bad_g, FPS,
                                   min_frames=settings.par_min_frames,
                                   max_frames=settings.par_max_frames,
                                   gap_break=15, gravity_px_s2=G_PX_S2,
                                   grav_tol_rel=settings.grav_tol_rel,
                                   max_rmse_frac=settings.par_max_rmse_frac,
                                   frame_small=min(W, H), frame_width=W,
                                   min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
        assert segs == [], "участок с ay≈3g не должен проходить гравитационный фильтр"
        assert len(diag) >= 1, "без фильтров кандидат находиться обязан"
        assert any("gravity" in d["failed_checks"] for d in diag)

    def test_render_mode_full_draws_whole_track(self, tmp_path):
        """В режиме full траектория рисуется ВО ВСЕХ кадрах с мячом (до поиска
        параболических участков), а не только в окнах пасов."""
        import cv2, os
        from app.services.renderer import render_tracked_video
        src = str(tmp_path / "in.mp4"); dst = str(tmp_path / "out_full.mp4")
        rng = np.random.default_rng(7)
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i in range(140):
            base = np.full((H, W, 3), 60, np.uint8)
            noise = rng.integers(0, 25, (H, W, 1), dtype=np.uint8)
            vw.write(np.clip(base + noise, 0, 255).astype(np.uint8))
        vw.release()
        an = self._make_full_analysis()
        out = render_tracked_video(src, dst, an, mode="full")
        assert os.path.exists(out) and os.path.getsize(out) > 0

        ca, cb = cv2.VideoCapture(src), cv2.VideoCapture(dst)
        marked_frames = 0; total = 0
        while True:
            oka, fa = ca.read(); okb, fb = cb.read()
            if not (oka and okb):
                break
            total += 1
            diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
            if int((diff > 40).sum()) > 100:
                marked_frames += 1
        ca.release(); cb.release()
        assert total == 140
        # режим passes красит ~40 кадров полёта; full — весь трек (1..100) + HUD всегда
        assert marked_frames > 90, ("full-режим должен рисовать полную траекторию, "
                                    f"закрашено кадров: {marked_frames}")

    def test_render_mode_passes_only_pass_windows(self, tmp_path):
        """Режим passes (по умолчанию) остаётся прежним: чистые кадры вне пасов."""
        import cv2
        from app.services.renderer import render_tracked_video
        src = str(tmp_path / "in.mp4"); dst = str(tmp_path / "out_passes.mp4")
        rng = np.random.default_rng(7)
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i in range(140):
            base = np.full((H, W, 3), 60, np.uint8)
            noise = rng.integers(0, 25, (H, W, 1), dtype=np.uint8)
            vw.write(np.clip(base + noise, 0, 255).astype(np.uint8))
        vw.release()
        an = self._make_full_analysis()
        render_tracked_video(src, dst, an, mode="passes", tail_hold_frames=24)
        ca, cb = cv2.VideoCapture(src), cv2.VideoCapture(dst)
        clean_outside = True
        n = 0
        while True:
            oka, fa = ca.read(); okb, fb = cb.read()
            if not (oka and okb):
                break
            n += 1
            if n < 31 or n > 95:      # вне паса и хвоста удержания
                diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
                if int((diff > 40).sum()) > 200:
                    clean_outside = False
        ca.release(); cb.release()
        assert clean_outside, "в режиме passes кадры вне пасов должны быть чистыми"

    def test_render_mode_setting_default_passes(self):
        from app.config import settings as s
        assert s.render_mode in ("passes", "full")


class TestPipelineTrackBuilding:
    def test_interp_track_closes_small_gaps(self):
        from app.services.pipeline import _interp_track
        raw = [(1, 0.0, 0.0), (2, 10.0, 5.0), (8, 70.0, 35.0), (9, 80.0, 40.0)]
        trk = _interp_track(raw, gap_max=15)
        frames = [p[0] for p in trk]
        assert frames == list(range(1, 10)), "разрыв <= gap_max интерполируется"
        # интерполированные точки лежат на прямой между наблюдениями
        d = {f: (x, y) for f, x, y in trk}
        assert abs(d[5][0] - 40.0) < 1e-6

    def test_interp_track_keeps_large_gaps(self):
        from app.services.pipeline import _interp_track
        raw = [(1, 0.0, 0.0), (30, 100.0, 50.0)]
        trk = _interp_track(raw, gap_max=15)
        assert [p[0] for p in trk] == [1, 30], "долгий разрыв — граница розыгрыша"


class TestTrajectoryFilter:
    """Скоростной шлюз ложных детекций + разрыв трека по макс. скорости + сетка."""

    GATE = 3.0
    VMIN = 16.0
    SMAX = 120.0

    def _clean(self, pts):
        from app.services.trajectory_filter import remove_outliers_velocity
        return remove_outliers_velocity(pts, gate_mult=self.GATE,
                                        v_min_px=self.VMIN,
                                        speed_max_px_f=self.SMAX)

    def test_isolated_teleport_removed(self):
        pts = [(i, float(i * 5), 100.0) for i in range(10)]
        pts.insert(5, (5, 400.0, 400.0))            # ложная точка-телепорт
        kept, dropped = self._clean(pts)
        assert len(dropped) == 1 and len(kept) == 10
        assert dropped[0][1] == 400.0

    def test_alternating_hotspot_chain_removed(self):
        """Главный регресс на реальном видео: ложные heatmap-срабатывания
        приходят ЧЕРЕДУЮЩИМИСЯ hotspot'ами (по 2+ кадра в одной точке).
        Старое eps-правило («обе соседки рядом») их НЕ видело; скоростной
        шлюз обязан вычистить всю цепочку, сохранив полёт мяча."""
        fly = [(f, 100.0 + 8.0 * f, 300.0 - 2.0 * f) for f in range(0, 20)]
        hot1 = [(2, 1700.0, 600.0), (3, 1701.0, 601.0)]     # static hotspot A
        hot2 = [(9, 60.0, 950.0), (10, 61.0, 949.0)]        # static hotspot B
        pts = sorted(fly + hot1 + hot2, key=lambda p: p[0])
        kept, dropped = self._clean(pts)
        removed_frames = {p[0] for p in dropped}
        assert {2, 3, 9, 10} <= removed_frames, "цепочка hotspot'ов удалена"
        # весь честный полёт сохранён
        assert all(any(k[0] == f for k in kept) for f in range(0, 20))

    def test_fast_flight_not_removed_but_split(self):
        """Честный быстрый перелёт (атака/подача): точки НЕ выбрасываются
        (шлюз пропорционален скорости); трек рвётся только там, где шаг
        физически невозможен (dist/dt > speed_max) — граница розыгрышей."""
        left = [(i, float(i * 5), 100.0) for i in range(5)]           # x 0..20
        right = [(100 + i, 900.0 + i * 5, 300.0) for i in range(5)]  # после прыжка
        pts = left + right
        kept, dropped = self._clean(pts)
        assert dropped == []                        # ничего не «ложное»
        from app.services.trajectory_filter import split_track_by_jumps
        head, jumped = split_track_by_jumps(kept, self.SMAX)
        assert jumped == 1 and head == left         # разрез ровно в месте прыжка

    def test_no_micro_splits_on_dense_fast_flight(self):
        """Регресс flights=[]: раньше разрез «по расстоянию > eps» рвал плотный
        быстрый полёт (шаг 40 px/кадр при eps=10) на микро-осколки. Теперь
        порог — скорость px/кадр: 40 < 120 → трек остаётся цельным."""
        from app.services.trajectory_filter import split_track_by_jumps
        fast = [(i, float(i * 40), 200.0 + i * 10) for i in range(10)]
        segs, rest = [], fast
        while True:
            head, jumped = split_track_by_jumps(rest, self.SMAX)
            segs.append(head)
            if not jumped:
                break
            rest = rest[len(head):]
        assert len(segs) == 1 and segs[0] == fast

    def test_net_level_manual_and_fixed_default(self):
        from app.services.trajectory_filter import estimate_net_level
        mixed = [(i, 50.0, y) for i, y in enumerate([100] * 30 + [400] * 30)]
        net = estimate_net_level(480, mixed, net_height_m=2.43, manual_y_px=216)
        assert net is not None and net.source == "manual" and net.y_px == 216
        net = estimate_net_level(480, mixed, net_height_m=2.43, manual_y_px=0,
                                 default_frac=0.65)
        assert net is not None and abs(net.y_px - 0.65 * 480) < 1e-6

    def test_net_level_none_when_track_all_above(self):
        """Весь трек выше типового уровня («сетку не видно») → None, лимит не
        применяется (на некоторых видео сетки может не быть)."""
        from app.services.trajectory_filter import estimate_net_level
        high = [(i, 50.0, 100.0) for i in range(50)]
        assert estimate_net_level(480, high, net_height_m=2.43) is None

    def test_clip_above_net(self):
        from app.services.trajectory_filter import (NetLevel,
                                                    clip_track_above_net)
        net = NetLevel(y_px=200.0, source="manual", scale_px_per_m=80.0,
                       net_height_m=2.43)
        track = [(0, 10.0, 150.0), (1, 20.0, 250.0), (2, 30.0, 120.0)]
        kept, stats = clip_track_above_net(track, net)
        assert [p[0] for p in kept] == [0, 2]
        assert stats["removed_below"] == 1 and stats["applied"]
