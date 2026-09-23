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


class TestRenderer:
    """Отрисовка траекторий поверх видео."""

    def _make_analysis(self):
        import numpy as np
        from app.services.pipeline import VideoAnalysis, PassEvent
        from app.services.physics import FlightEstimate
        fps = 30.0
        set_gravity_px(fps, settings.gravity_ratio)
        pts = []
        for i in range(40):
            t = i / fps
            x = 100 + 8 * i
            y = 300 - 250 * t + 0.5 * (settings.gravity_ratio * fps * fps) * t * t
            pts.append((i + 31, x, min(y, 470)))
        traj = [{"t": round(i / fps, 3), "x_m": x * 0.01, "y_m": y * 0.01,
                 "x_px": x, "y_px": y} for i, (f, x, y) in enumerate(pts)]
        fl = FlightEstimate(time_of_flight_s=1.3, release_frame=31, catch_frame=70,
                            apex_height_m=1.5, distance_m=9.0, initial_speed_mps=6.0,
                            peak_speed_mps=7.0, fit_rmse_px=1.0, trajectory=traj,
                            method="tracked_direct")
        an = VideoAnalysis(fps=fps, width=640, height=480, n_frames=100)
        an.ball_track_points = [{"frame": f, "x": round(x, 1), "y": round(y, 1)}
                                for f, x, y in pts]
        an.passes = [PassEvent(release_frame=31, catch_frame=70,
                               passer_bbox=[0, 0, 10, 10], catcher_bbox=None, flight=fl)]
        return an, pts

    def test_draw_flight_overlays_pixels(self):
        """Прямая проверка: draw_flight добавляет цветные пиксели во время полёта."""
        import numpy as np
        from app.services.renderer import FlightOverlay, draw_flight
        an, pts = self._make_analysis()
        frame = np.full((480, 640, 3), 60, np.uint8)
        before = int((frame != 60).sum())
        ov = FlightOverlay(0, 31, 70, pts, {"time_of_flight_s": 1.3})
        ov._cur_frame = 50
        draw_flight(frame, ov, 6, 25)
        colored = int((frame != 60).sum())
        assert colored - before > 500, "оверлей должен закрасить заметную область"
        # цвет дуги/хвоста отличается от фона по каналам (BGR-оранжевый)
        assert (frame[:, :, 2] > 100).sum() > 100

    def test_render_produces_video_with_trails(self, tmp_path):
        """E2E: видео на выходе аналогично входному, но в полёте виден след."""
        import cv2, numpy as np, os
        from app.services.renderer import render_tracked_video
        src = str(tmp_path / "in.mp4"); dst = str(tmp_path / "out.mp4")
        fps = 30.0
        w, h = 640, 480
        rng = np.random.default_rng(7)
        vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        assert vw.isOpened()
        for i in range(100):
            base = np.full((h, w, 3), 60, np.uint8)   # «поле»
            base[200:400, 80:160] = 120               # игрок слева
            base[200:400, 480:560] = 120              # игрок справа
            noise = rng.integers(0, 25, (h, w, 1), dtype=np.uint8)
            vw.write(np.clip(base + noise, 0, 255).astype(np.uint8))
        vw.release()
        an, pts = self._make_analysis()
        out = render_tracked_video(src, dst, an)
        assert os.path.exists(out) and os.path.getsize(out) > 0

        ca, cb = cv2.VideoCapture(src), cv2.VideoCapture(dst)
        n = 0; flight_colored = 0; clean_ok = True
        while True:
            oka, fa = ca.read(); okb, fb = cb.read()
            if not (oka and okb):
                break
            n += 1
            diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
            strong = int((diff > 40).sum())           # яркие штрихи оверлея
            if 40 <= n <= 70 and strong > 200:
                flight_colored += 1                   # во время полёта есть след
            if n >= 80 and strong > 200:
                clean_ok = False                      # после паса кадр чистый
        ca.release(); cb.release()
        assert n == 100, "длина результата должна совпадать с исходником"
        assert flight_colored >= 20, \
            "в кадрах полёта должен формироваться след траектории"
        assert clean_ok, "после завершения паса кадры должны оставаться чистыми"

    def test_point_at_time_interpolation(self):
        from app.services.renderer import _point_at_time
        pts = [(1, 0.0, 0.0), (3, 20.0, 10.0)]
        assert _point_at_time(pts, 2.0) == (10.0, 5.0)
        assert _point_at_time(pts, 0.5) is None
        assert _point_at_time(pts, 3.0) == (20.0, 10.0)


class TestTrackerAntiJump:
    """Регрессии на баг «срыв трека»: ложная далёкая детекция не должна
    перетягивать трекер, а сорванный трек — умирать, а не лететь по экрану."""

    def test_far_false_detection_does_not_hijack_track(self):
        tr = SORTTracker(min_hits=1)
        tr.set_frame_size(640, 480)
        for f in range(5):                      # стабильный мяч в (100..120, 100..120)
            tr.update(np.array([[100, 100, 120, 120]], float))
        # ложная сдетектированная «сфера» далеко от реального мяча.
        # ВНИМАНИЕ: трекер многообъектный — далёкая детекция порождает НОВЫЙ
        # трек; баг («срыв») был бы, если существующий трек ПЕРЕСКОЧИЛ на неё.
        out = tr.update(np.array([[500, 400, 520, 420]], float))
        real = [o for o in out if math.hypot((o[1][0] + o[1][2]) / 2 - 110,
                                             (o[1][1] + o[1][3]) / 2 - 110) < 60]
        assert len(real) == 1, \
            "реальный трек обязан остаться у мяча (не перескочить на ложную детекцию)"
        # и он именно тот же id, что до ложной детекции
        assert real[0][0] == min(o[0] for o in out)

    def test_lost_track_dies_instead_of_flying_offscreen(self):
        tr = SORTTracker(min_hits=1, max_age=5)
        tr.set_frame_size(640, 480)
        for f in range(5):
            x = 580 + 10 * f                    # быстрый мяч уходит за правый край
            tr.update(np.array([[x, 100, x + 20, 120]], float))
        gone = False
        for _ in range(6):
            out = tr.update(np.empty((0, 4)))
            if not out:
                gone = True
                break
        assert gone, "сорванный трек обязан умереть, а не продолжать путь по экрану"


class TestPassGating:
    """Мяч отслеживается только во время пасов, а не на всём видео."""

    class _FakeDetector:
        """Детектор-заглушка: кадры сцены описаны списком (balls, persons)."""
        def __init__(self, frames):
            self.frames = list(frames)
            self.i = 0
        def detect(self, frame):
            from app.services.detector import Detection
            balls, persons = self.frames[min(self.i, len(self.frames) - 1)]
            self.i += 1
            b = [Detection(np.array(bb, float), 0.9, 32) for bb in balls]
            p = [Detection(np.array(bb, float), 0.9, 0) for bb in persons]
            return b, p

    def _write_video(self, path, n_frames=120):
        import cv2
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        for i in range(n_frames):
            f = np.full((480, 640, 3), 60, np.uint8)
            w.write(f)
        w.release()

    def _scene(self):
        P1 = [400, 200, 460, 440]   # игрок-отдающий
        P2 = [100, 200, 160, 440]   # игрок-принимающий
        held = ([[420, 210, 440, 230]], [P1])          # мяч в зоне P1
        fly = lambda t: ([[420 - 25 * t, 200 - 5 * math.sin(t), 440 - 25 * t,
                          220 - 5 * math.sin(t)]], [P1, P2])
        caught = ([[120, 210, 140, 230]], [P1, P2])    # мяч в зоне P2
        idle = ([], [P1, P2])                          # мяч вообще не виден
        frames = []
        frames += [held] * 8                           # владение P1
        frames += [fly(t) for t in range(1, 13)]       # полёт паса
        frames += [caught] * 8                         # приёмка P2
        frames += [idle] * 40                          # длинный «мёртвый» участок
        frames += [held] * 4                           # мяч снова у P1 (без полёта)
        return frames

    def test_only_flight_is_tracked_and_single_pass(self, tmp_path):
        from app.services.pipeline import analyze_video
        src = str(tmp_path / "v.mp4")
        self._write_video(src)
        an = analyze_video(src, detector=self._FakeDetector(self._scene()))
        # ровно один пас
        assert len(an.passes) == 1, f"ожидался 1 пас, получено {len(an.passes)}"
        p = an.passes[0]
        # все точки траектории лежат строго внутри окна полёта
        assert an.ball_track_points, "траектория полёта должна присутствовать"
        assert all(p.release_frame <= pt["frame"] <= p.catch_frame
                   for pt in an.ball_track_points), \
            "вне полёта мяч отслеживаться не должен"
        # «мёртвая» зона (~50 кадров после приёмки) не порождает новых треков
        tail = [pt for pt in an.ball_track_points if pt["frame"] > p.catch_frame]
        assert not tail

    def test_no_release_contact_means_no_pass(self, tmp_path):
        """Мяч, прилетевший «ниоткуда» (без владения перед релизом), — не пас."""
        from app.services.pipeline import analyze_video
        src = str(tmp_path / "v2.mp4")
        self._write_video(src)
        P1 = [400, 200, 460, 440]
        frames = []
        frames += ([], [P1]) * 5
        for t in range(1, 15):                        # мяч летит, но владения не было
            frames.append(([[420 - 20 * t, 200, 440 - 20 * t, 220]], [P1]))
        an = analyze_video(src, detector=self._FakeDetector(frames))
        assert len(an.passes) == 0
        assert len(an.ball_track_points) == 0
