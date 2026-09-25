"""Пайплайн анализа видео: детекция → SORT-трекинг мяча + игроков → события.

Архитектура (по аналогии с BallTime™ / готовыми sports-analytics-решениями,
где события привязываются к игрокам, а не только к геометрии мяча):

  * Игроки ведутся отдельным SORT-трекером (person tracker): стабильный
    track id игрока — основа для «пас от одного игрока другому». Раньше
    игроки матчились по «квантованному bbox» (_person_key), что ломалось
    при дрожании/перемещении боксов;
  * CONTACT: центр мяча попал в зону вокруг бокса игрока (бокс наружу на
    contact_expand * диагональ); владение засчитывается после contact_streak
    подряд кадров контакта — «мяч отлетает от игрока»;
  * RELEASE: был ПОДТВЕРЖДЁННЫЙ контакт (владение) и мяч покинул зону
    игрока (не позже release_max_lag кадров после последнего касания).
    Мягкие «полёты ниоткуда» (без владения) НЕ порождают сегментов: на
    реальных видео они обрывали настоящий пас на первых кадрах и давали
    слишком короткие сегменты, которые классификатор отвергал, — итог:
    «пас вообще не детектится» (баг). Порог BT_REQUIRE_CONTACT=0 оставляет
    мягкий режим как аварийный fallback;
  * CATCH: траектория входит в расширенный бокс ДРУГОГО игрока (другой
    person-track);
  * между release и catch формируется сегмент полёта, который затем
    КЛАССИФИЦИРУЕТСЯ (_is_pass): пас = мяч отлетел от игрока и пролетёл
    выражено по параболе влево/вправо к другому игроку. Подача
    (вертикальный удар), приём (медленный контакт), атака (монотонное
    падение / возврат тому же игроку) — пасами НЕ считаются (баг
    «4 паса за розыгрыш»).
  * Траектория мяча пишется в ball_track_points ВСЕГДА, когда мяч виден
    (а не только внутри признанных полётов): иначе при строгой
    классификации на видео не остаётся ни одной отметки (баг «нет
    траекторий»), а API/render теряют данные для показа.
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
    passer_id: int | None = None
    catcher_id: int | None = None


@dataclass
class VideoAnalysis:
    fps: float
    width: int
    height: int
    n_frames: int
    ball_track_points: list[dict] = field(default_factory=list)  # [{frame,x,y}]
    passes: list[PassEvent] = field(default_factory=list)
    ball_radius_px: float = 6.0


def _expanded(box: np.ndarray, scale: float) -> tuple[float, float, float, float]:
    """Зона контакта/приёмки: бокс игрока, расширенный НАРУЖНУ на
    scale * диагональ бокса с каждой стороны."""
    x1, y1, x2, y2 = box
    pad = math.hypot(x2 - x1, y2 - y1) * scale
    return x1 - pad, y1 - pad, x2 + pad, y2 + pad


def _inside(pt: tuple[float, float], rect) -> bool:
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


def _is_pass(pts: list[tuple[int, float, float]], passer: Detection | None,
             catcher: Detection | None, width: int, height: int,
             fps: float) -> bool:
    """Классификация сегмента полёта: пас или подача/приём/атака.

    Пас = мяч ОТЛЕТЕЛ ОТ ИГРОКА и пролетел выражено по параболе влево или
    вправо к ДРУГОМУ игроку. Признаки (эвристики в духе готовых решений —
    спортивной аналитики SportsCode/Second Spectrum и research-подходов
    «ball strike detection»):
      1. гор. пролёт |dx| >= pass_min_dx_frac * W (выраженно влево/вправо);
         вертикальные удары (подача вверх, свеча) отсекаются;
      2. скорость релиза >= pass_min_speed_px_f px/кадр (приём сверху —
         медленное движение, не пас);
      3. вершина внутри полёта >= pass_min_apex_px (параболичность;
         монотонное падение — это сброс/атака по нисходящей без дуги);
      4. приёмка другим игроком: bbox отдающего != bbox принимающего
         (подброс себе / та же рука — не пас); если приёмки нет вообще
         (конец видео/окклюзия) — считаем передачей только длинный
         горизонтальный полёт без возврата к отдающему.
    """
    if len(pts) < settings.min_flight_frames:
        return False
    xs = np.array([p[1] for p in pts], dtype=float)
    ys = np.array([p[2] for p in pts], dtype=float)
    span = abs(xs[-1] - xs[0])
    # максимальный горизонтальный пролёт внутри сегмента (устойчив к шуму
    # последних точек Kalman-сглаживания)
    dx = float(np.max(xs) - np.min(xs))
    dx = max(dx, span)
    if dx < settings.pass_min_dx_frac * width:
        return False                                   # нет выраженного полёта влево/вправо
    # скорость релиза: средняя по первым ~3 наблюдениям (мяч только что
    # отлетел от игрока); медленные движения — это приём, а не передача
    k = min(3, len(pts) - 1)
    dt = max(pts[k][0] - pts[0][0], 1)
    v_release_per_f = math.hypot(xs[k] - xs[0], ys[k] - ys[0]) / dt
    if v_release_per_f < settings.pass_min_speed_px_f:
        return False
    # выраженная дуга: вершина (минимум y, ось вниз) строго внутри полёта и
    # заметно выше обоих концов; монотонное падение/подъём (атака вниз,
    # свеча вверх без приёма другим игроком) — не пас
    i_apex = int(np.argmin(ys))
    if not (0 < i_apex < len(pts) - 1):
        return False
    apex_lift = min(ys[0], ys[-1]) - float(ys[i_apex])
    if apex_lift < settings.pass_min_apex_px:
        return False                                   # плоский дрейф без параболичности
    if catcher is None:
        # Приёмка не зафиксирована. Это НОРМАЛЬНО для реальных видео: COCO-
        # детектор людей не стабилен (игроки слиты/в кадре только ноги), поэтому
        # требовать «catch другим игроком» как обязательное условие — значит
        # потерять все реальные пасы (баг «пас вообще не детектится»).
        # Без приёмки подтверждаем пас физикой полёта: дуга уже проверена выше,
        # теперь требуем достаточно длинный горизонтальный пролёт
        # (pass_no_catch_min_dx_frac), чтобы короткие отскоки/шум трека
        # не плодили фантомные события.
        return dx >= settings.pass_no_catch_min_dx_frac * width
    if passer is not None:
        same_center = math.hypot(passer.center[0] - catcher.center[0],
                                 passer.center[1] - catcher.center[1]) \
            < 0.05 * math.hypot(width, height)
        overlap = _inside(passer.center, catcher.bbox) and _inside(catcher.center,
                                                                   passer.bbox)
        if same_center or overlap:
            # «туда-обратно» к тому же игроку: подброс/доведение — не пас
            return not settings.pass_reject_self_return
    return True


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
    # Отдельный SORT-трекер игроков: даёт СТАБИЛЬНЫЕ track id людей, по ним
    # определяем «тот же игрок / другой игрок» (passer != catcher). Раньше
    # игроки матчились по квантованному bbox — на реальных видео (дрожание,
    # перемещение, частичные окклюзии) идентификаторы прыгали и владение
    # не подтверждалось никогда — «пас вообще не детектится».
    ptracker = SORTTracker(max_age=settings.person_max_age, min_hits=1)
    tracker.set_frame_size(width, height)   # физический gate от срывов за кадр
    ptracker.set_frame_size(width, height)
    set_gravity_px(fps, settings.gravity_ratio)   # только для fit'а метрик (px/s^2)
    # Kalman-модель постоянного ускорения по вертикали: эмпирическое g в
    # px/frame^2 (НЕ gravity_ratio*fps^2 — это размерность px/s^2 для фита;
    # подстановка её в трекер разгоняла треки по вертикали — «catch на
    # потолке», а при другом fps рвала полёт на осколки).
    # Предел скорости трекера — из физического gate кадра.
    v_max = settings.max_jump_frac * min(width, height)
    tracker.set_physics(settings.kalman_gravity_px_f2, max_speed_px_per_f=v_max)

    analysis = VideoAnalysis(fps=fps, width=width, height=height, n_frames=0)

    prev_ball: tuple[float, float] | None = None
    contact_person: Detection | None = None      # игрок, державший мяч
    contact_pid: int | None = None               # track id владеющего игрока
    last_contact_frame: int = 0                  # последний кадр подтверждения владения
    flight_pts: list[tuple[int, float, float]] = []
    ball_diam_sum, ball_diam_n = 0.0, 0
    frame_id = 0
    contact_streak = 0        # подряд идущие кадры «мяч в зоне игрока»
    no_person_frames = 0      # подряд идущие кадры без людей в кадре

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        balls, persons = det.detect(frame)

        ball_det: Detection | None = balls[0] if balls else None
        dets_arr = np.array([b.bbox for b in balls[:3]]) if balls else np.empty((0, 4))
        tracks = tracker.update(dets_arr)
        # --- трекинг игроков: [{track_id: (bbox, center)}] -------------------
        pdets_arr = np.array([p.bbox for p in persons]) if persons else np.empty((0, 4))
        ptracks = ptracker.update(pdets_arr)
        person_tracks = {int(t[0]): t[1] for t in ptracks}
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
            # физический gate на уровне пайплайна: если Kalman-предсказание
            # уехало далеко от свежей детекции — это сорванный трек, не
            # подмешиваем его (иначе траектория «тянется» за призраком).
            jump = math.hypot(cx_t - bx, cy_t - by)
            max_jump = settings.max_jump_frac * min(width, height)
            if jump <= max_jump:
                alpha = 0.65   # доверие Kalman-предсказанию при наличии свежей детекции
                center = (alpha * cx_t + (1 - alpha) * bx, alpha * cy_t + (1 - alpha) * by)
                tvx, tvy = best[2], best[3]
            else:
                center = (bx, by)     # доверяем только детекции
            used_det = ball_det
        elif ball_det is not None:
            center = ball_det.center
            used_det = ball_det
        elif tracks:      # окклюзия: держимся за предсказание трекера
            tb = tracks[0][1]
            cx_t, cy_t = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            # то же ограничение на extrapolation: без наблюдений долго лететь
            # по инерции нельзя (max_age мал, но и запасной контроль не помеха)
            max_jump = settings.max_jump_frac * min(width, height) * settings.max_age
            if prev_ball is not None and math.hypot(cx_t - prev_ball[0],
                                                    cy_t - prev_ball[1]) > max_jump:
                center = None
            else:
                center = (cx_t, cy_t)
                tvx, tvy = tracks[0][2], tracks[0][3]
            used_det = None               # диаметра не наблюдаем
        else:
            center = None

        if center is not None:
            prev_ball = center
        else:
            prev_ball = None

        # --- состояние контакта с игроком ------------------------------------
        # ближайший игрок считаем по СВЕЖИМ трекам (детекция или Kalman-
        # предсказание трекера игроков): детекции YOLO/CV нестабильны покадрово
        nearest_person = None
        nearest_pid = None
        if center and person_tracks:
            best_p, bestd, best_id = None, 1e18, None
            for pid, tb in person_tracks.items():
                pcx, pcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                d = math.hypot(center[0] - pcx, center[1] - pcy)
                if d < bestd:
                    best_p, bestd, best_id = np.asarray(tb, float), d, pid
            if best_p is not None:
                nearest_person = Detection(best_p, 0.9, 0)
                nearest_pid = best_id

        if center and nearest_person is not None:
            rect = _expanded(nearest_person.bbox, settings.contact_expand)
            touching = _inside(center, rect)
        else:
            touching = False

        # владение засчитываем только после непрерывной серии контактов —
        # защита от одиночных ложных срабатываний («мяч» рядом с игроком)
        if touching:
            contact_streak += 1
        else:
            contact_streak = 0
        has_contact = contact_streak >= settings.contact_streak

        # эпизод без людей вообще (повтор/тайм-аут/пустой кадр) — сброс состояния
        if not persons and not person_tracks:
            no_person_frames += 1
            if no_person_frames >= settings.no_person_reset:
                contact_person = None
                contact_pid = None
                contact_streak = 0
                if flight_pts:
                    _finalize_segment(analysis, flight_pts, contact_person, None,
                                      contact_pid, None, fps, width, height)
                    flight_pts = []
        else:
            no_person_frames = 0

        if has_contact:
            if flight_pts:
                # приёмка: полёт завершён (пас или другое событие — решит
                # классификатор _is_pass)
                _finalize_segment(analysis, flight_pts, contact_person,
                                  nearest_person, contact_pid, nearest_pid,
                                  fps, width, height)
                flight_pts = []
            # фиксируем владение: новый контакт всегда принадлежит текущему
            # ближайшему игроку (если серия идёт у того же — ничего не меняется)
            contact_person = nearest_person
            contact_pid = nearest_pid
            last_contact_frame = frame_id
        elif center is not None:
            if not flight_pts:
                # Релиз начинается ТОЛЬКО после подтверждённого владения
                # (строгий режим по умолчанию). Мягкий режим
                # (BT_REQUIRE_CONTACT=0) допускает «полёт ниоткуда», если в
                # кадре есть люди, — аварийный fallback для видео, где
                # контактная зона не настраивается.
                fresh = contact_person is not None \
                    and (frame_id - last_contact_frame) <= settings.release_max_lag
                soft = (not settings.require_release_contact) and persons
                if fresh or soft:
                    flight_pts.append((frame_id, center[0], center[1]))
            else:
                flight_pts.append((frame_id, center[0], center[1]))
            # защита от «вечного» полёта без приёмки
            if flight_pts and frame_id - flight_pts[-1][0] > settings.max_age:
                _finalize_segment(analysis, flight_pts, contact_person, None,
                                  contact_pid, None, fps, width, height)
                flight_pts = []
                contact_person = None
                contact_pid = None

        if used_det is not None:
            diag = used_det.diag
            ball_diam_sum += min(diag, height * 0.5)
            ball_diam_n += 1
        if center is not None:
            # Траектория пишется ВСЕГДА, когда мяч виден: и во время полёта,
            # и при владении (маркер мяча на видео обязателен — баг «никаких
            # отметок на видео нет»). Полупрозрачная отрисовка вне полётов
            # делается в renderer по признаку «точка внутри окна паса».
            analysis.ball_track_points.append(
                {"frame": frame_id, "x": round(center[0], 1), "y": round(center[1], 1)})
        if progress_cb and frame_id % 25 == 0:
            progress_cb(frame_id, total)
        if max_frames and frame_id >= max_frames:
            break

    # незавершённый полёт в конце видео
    if flight_pts:
        _finalize_segment(analysis, flight_pts, contact_person, None,
                          contact_pid, None, fps, width, height)
    cap.release()
    if ball_diam_n:
        analysis.ball_radius_px = max(4.0, ball_diam_sum / ball_diam_n / 2.0)
    analysis.n_frames = frame_id
    return analysis


def _finalize_segment(analysis: VideoAnalysis, pts, passer: Detection | None,
                      catcher: Detection | None, passer_id, catcher_id,
                      fps: float, width: int, height: int) -> None:
    """Завершение сегмента полёта: классификация (пас / подача / приём /
    атака) и, если это пас — оценка физики и регистрация события."""
    try:
        if not _is_pass(pts, passer, catcher, width, height, fps):
            return
    except Exception:  # noqa: BLE001 — классификатор не должен валить анализ
        return
    _finalize_pass(analysis, pts, passer, catcher, fps,
                   ball_diam_sum_px(analysis), passer_id, catcher_id)


def ball_diam_sum_px(analysis: VideoAnalysis) -> float:
    return analysis.ball_radius_px * 2.0


def _finalize_pass(analysis: VideoAnalysis, pts, passer: Detection | None,
                   catcher: Detection | None, fps: float,
                   ball_diam_px: float, passer_id=None, catcher_id=None) -> None:
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
        passer_id=passer_id,
        catcher_id=catcher_id,
    )
    analysis.passes.append(ev)
