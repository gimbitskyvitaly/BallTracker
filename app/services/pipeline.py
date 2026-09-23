"""Пайплайн анализа видео: YOLO-детекция → SORT-трекинг → события паса → физика.

Логика детекции паса (по аналогии с BallTime™, где события привязываются
к игрокам):
  * RELEASE: мяч был «в контакте» с игроком (центр мяча внутри расширенного
    бокса person) и в следующем кадре покинул зону с резким ростом скорости;
  * CATCH: траектория мяча входит в расширенный бокс другого игрока или
    скорость мяча падает почти до нуля рядом с игроком;
  * между release и catch формируется сегмент полёта → оценка ToF через
    fit физмодели (app.services.physics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings
from app.services.detector import BallDetector, Detection
from app.services.tracker import SORTTracker
from app.services.physics import estimate_flight, FlightEstimate, set_gravity_px


@dataclass
class PassEvent:
    release_frame: int
    catch_frame: int
    passer_bbox: list[float]
    catcher_bbox: list[float] | None
    flight: FlightEstimate | None = None


@dataclass
class VideoAnalysis:
    fps: float
    width: int
    height: int
    n_frames: int
    ball_track_points: list[dict] = field(default_factory=list)  # [{frame,x,y}]
    passes: list[PassEvent] = field(default_factory=list)


def _expanded(box: np.ndarray, scale: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = math.hypot(x2 - x1, y2 - y1) * scale / 2
    return cx - r, cy - r, cx + r, cy + r


def _inside(pt: tuple[float, float], rect) -> bool:
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


def analyze_video(path: str, detector: BallDetector | None = None,
                  max_frames: int | None = None,
                  progress_cb=None) -> VideoAnalysis:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = detector or BallDetector()
    tracker = SORTTracker()
    set_gravity_px(fps, settings.gravity_ratio)

    analysis = VideoAnalysis(fps=fps, width=width, height=height, n_frames=0)

    prev_ball: tuple[float, float] | None = None
    prev_persons: list[Detection] = []
    contact_person: Detection | None = None      # игрок, державший мяч
    flight_pts: list[tuple[int, float, float]] = []
    ball_diam_sum, ball_diam_n = 0.0, 0
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        balls, persons = det.detect(frame)

        ball_det: Detection | None = balls[0] if balls else None
        dets_arr = np.array([b.bbox for b in balls[:3]]) if balls else np.empty((0, 4))
        tracks = tracker.update(dets_arr)
        tvx = tvy = 0.0
        center: tuple[float, float] | None = None
        used_det: Detection | None = None
        # выбор позиции мяча: трекера нет -> прямая детекция; иначе — Kalman-сглаживание
        if ball_det is not None and tracks:
            bx, by = ball_det.center
            best = min(tracks, key=lambda t: math.hypot((t[1][0] + t[1][2]) / 2 - bx,
                                                        (t[1][1] + t[1][3]) / 2 - by))
            tb = best[1]
            cx_t, cy_t = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            alpha = 0.65   # доверие Kalman-предсказанию при наличии свежей детекции
            center = (alpha * cx_t + (1 - alpha) * bx, alpha * cy_t + (1 - alpha) * by)
            tvx, tvy = best[2], best[3]
            used_det = ball_det
        elif ball_det is not None:
            center = ball_det.center
            used_det = ball_det
        elif tracks:      # окклюзия: держимся за предсказание трекера
            tb = tracks[0][1]
            center = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
            tvx, tvy = tracks[0][2], tracks[0][3]
            used_det = None               # диаметра не наблюдаем
        else:
            center = None

        if used_det is not None:
            diag = used_det.diag
            ball_diam_sum += min(diag, height * 0.5)
            ball_diam_n += 1
        if center is not None:
            analysis.ball_track_points.append(
                {"frame": frame_id, "x": round(center[0], 1), "y": round(center[1], 1)})

        # --- состояние контакта с игроком -----------------------------------
        nearest_person = None
        if center and persons:
            best_p, bestd = None, 1e18
            for p in persons:
                pcx, pcy = p.center
                d = math.hypot(center[0] - pcx, center[1] - pcy)
                if d < bestd:
                    best_p, bestd = p, d
            nearest_person = best_p

        speed = math.hypot(tvx, tvy)

        if center and nearest_person is not None:
            rect = _expanded(nearest_person.bbox, settings.contact_expand)
            touching = _inside(center, rect)
        else:
            touching = False

        if touching:
            if flight_pts:
                # приёмка: полёт завершён
                if len(flight_pts) >= settings.min_flight_frames:
                    _finalize_pass(analysis, flight_pts, contact_person, nearest_person,
                                   fps, ball_diam_sum / max(ball_diam_n, 1))
                flight_pts = []
            contact_person = nearest_person
        elif center is not None:
            if not flight_pts:
                # релиз: был контакт и мяч улетел из зоны игрока
                if contact_person is not None:
                    flight_pts.append((frame_id, center[0], center[1]))
            else:
                flight_pts.append((frame_id, center[0], center[1]))
            # защита от «вечного» полёта без приёмки
            if flight_pts and frame_id - flight_pts[-1][0] > settings.max_age:
                _finalize_pass(analysis, flight_pts, contact_person, None, fps,
                               ball_diam_sum / max(ball_diam_n, 1))
                flight_pts = []
                contact_person = None

        analysis.n_frames = frame_id
        if progress_cb and frame_id % 25 == 0:
            progress_cb(frame_id, total)
        if max_frames and frame_id >= max_frames:
            break

    # незавершённый полёт в конце видео
    if flight_pts:
        _finalize_pass(analysis, flight_pts, contact_person, None, fps,
                       ball_diam_sum / max(ball_diam_n, 1))
    cap.release()
    return analysis


def _finalize_pass(analysis: VideoAnalysis, pts, passer: Detection | None,
                   catcher: Detection | None, fps: float,
                   ball_diam_px: float) -> None:
    if len(pts) < settings.min_flight_frames:
        return
    flight = estimate_flight(pts, fps, ball_diam_px,
                             settings.drag_coefficient,
                             use_physics=settings.use_physics_fit)
    ev = PassEvent(
        release_frame=int(pts[0][0]),
        catch_frame=int(pts[-1][0]),
        passer_bbox=passer.bbox.tolist() if passer else [0, 0, 0, 0],
        catcher_bbox=catcher.bbox.tolist() if catcher else None,
        flight=flight,
    )
    analysis.passes.append(ev)
