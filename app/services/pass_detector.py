"""Сегментация розыгрышей и детекция пасов по траектории мяча.

Задача: «в каждом розыгрыше (время, когда мяч находится на одной стороне
площадки) найти параболические участки в левую или правую часть сетки».

Алгоритм (все пороги — доли размера кадра, поэтому работают на любом
разрешении/FPS):

1. РОЗЫГРЫШИ (rallies). Плотность точек трека мяча во времени: кадры, где
   мяч виден, группируются в эпизоды; разрыв видимости > rally_gap_frames
   закрывает розыгрыш. Внутри розыгрыша позиция мяча по x относительно линии
   сетки net_x определяет сторону; устойчивая смена стороны (гистерезис
   ±net_hyst_frac*W) — это переход мяча через сетку, т.е. новая фаза розыгрыша.

2. ПАРАБОЛИЧЕСКИЕ УЧАСТКИ. На последовательности (frame, x, y) скользящим
   окном par_min_frames..par_max_frames ищутся интервалы, которые хорошо
   описываются моделью свободного полёта:
       x(t) = x0 + vx*t          (равномерно)
       y(t) = y0 + vy*t + 0.5*g*t^2  (равноускоренно, g — эмпирика px/s^2)
   Фит — обычный least squares по базису [t, t^2]; участок принимается, если
   RMSE <= par_max_rmse_frac*min(H,W), вертикальное ускорение близко к g
   (±par_grav_tol), а вершина/монотонность согласуются с «броском» (мяч
   сначала набирает высоту или летит вниз замедленно-ускоренно). Это отсекает
   ведение мяча руками (движение вместе с игроком — не баллистика) и отскоки.

3. НАПРАВЛЕНИЕ К СЕТКЕ. Для каждого параболического участка считается знак
   горизонтальной скорости vx: toward_net_left — мяч летит ВЛЕВО к сетке
   (vx < 0 и финишная точка ближе к сетке, чем стартовая), toward_net_right —
   ВПРАВО к сетке (vx > 0 и тоже приближается к сетке). Участки, летящие ОТ
   сетки (атака после передачи назад, отбой), помечаются direction="away" и
   пасами не считаются.

4. ПАС = параболический участок внутри розыгрыша, летящий К сетке, длиной не
   менее min_flight_frames и не «склеенный» с предыдущим пасом того же
   направления без приёмки между ними (после паса должен быть хотя бы один
   кадр вне полёта — контакт/пауза траектории).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Rally:
    """Розыгрыш: непрерывный эпизод видимости мяча на одной/обеих сторонах."""
    start_frame: int
    end_frame: int
    # фазы внутри розыгрыша (сегменты между переходами мяча через сетку)
    phases: list[tuple[int, int, str]] = field(default_factory=list)  # (f0,f1,side)


@dataclass
class ParabolicSegment:
    """Параболический (баллистический) участок траектории мяча."""
    start_frame: int
    end_frame: int
    points: list[tuple[int, float, float]]        # [(frame, x, y)] наблюдения
    vx_px_s: float                                # горизонтальная скорость, px/s
    vy0_px_s: float                               # вертикальная скорость в начале
    ay_px_s2: float                               # вертикальное ускорение (≈g)
    rmse_px: float
    apex_y_px: float                              # y вершины (меньше = выше)
    direction: str                                # "left" | "right" | "away"
    to_net_ratio: float                           # насколько точка финиша ближе к сетке


@dataclass
class PassSegment:
    release_frame: int
    catch_frame: int
    direction: str            # "left" | "right" — движение к сетки слева/справа
    segment: ParabolicSegment


# ------------------------------------------------------------------ helpers
def _fit_ballistic(frames: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                   fps: float) -> dict | None:
    """LSQ-фит x=vx*t+x0, y=0.5*a*t^2+vy*t+y0. Возвращает параметры и RMSE."""
    if len(frames) < 4:
        return None
    t = frames / fps
    A = np.stack([np.ones_like(t), t], axis=1)          # x ~ [1, t]
    try:
        cx, *_ = np.linalg.lstsq(A, xs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    vx = float(cx[1])
    pred_x = A @ cx
    B = np.stack([np.ones_like(t), t, t * t], axis=1)   # y ~ [1, t, t^2]
    try:
        cy, *_ = np.linalg.lstsq(B, ys, rcond=None)
    except np.linalg.LinAlgError:
        return None
    vy = float(cy[1])
    ay = float(2.0 * cy[2])
    pred_y = B @ cy
    rmse = float(np.sqrt(np.mean((pred_x - xs) ** 2 + (pred_y - ys) ** 2)))
    tt_apex = -vy / ay if ay != 0 else 0.0
    span = float(t[-1] - t[0])
    apex_t = float(np.clip(tt_apex - t[0], 0.0, span))
    apex_y = float(cy[0] + cy[1] * tt_apex + cy[2] * tt_apex ** 2)
    return {"vx": vx, "vy0": vy, "ay": ay, "rmse_px": rmse,
            "apex_y_px": apex_y, "apex_offset_s": apex_t}


def find_parabolic_segments(points: list[tuple[int, float, float]], fps: float,
                            *, min_frames: int, max_frames: int,
                            max_rmse_frac: float, gravity_px_s2: float,
                            grav_tol_rel: float, gap_break: int,
                            min_horizontal_disp_frac: float,
                            frame_small: int, frame_width: int) -> list[ParabolicSegment]:
    """Жадный поиск непересекающихся баллистических участков во фрагменте трека.

    Идём слева направо; для каждой стартовой точки пробуем окна от максимального
    размера к минимальному и берём первое окно с хорошим фитом (greedy longest).
    Условия «параболичности»:
      * RMSE фита <= max_rmse_frac * min(W,H);
      * вертикальное ускорение близко к эмпирическому g (мяч в свободном полёте,
        а не в руках/ведении — там ускорения произвольные);
      * горизонтальное смещение >= min_horizontal_disp_frac * W (настоящий
        перелёт, а не дребезг на месте);
      * вершина параболы лежит внутри окна (полнопролётный пас: взлёт+падение),
        либо окно целиком монотонно — тогда это часть одного большого полёта и
        оно будет расширено greedier-окном слева.
    """
    if len(points) < min_frames:
        return []
    pts = sorted(points, key=lambda p: p[0])
    segments: list[ParabolicSegment] = []
    i = 0
    n = len(pts)
    while i < n:
        placed = False
        for j in range(min(n - 1, i + max_frames), i + min_frames - 1, -1):
            f = np.array([p[0] for p in pts[i:j + 1]], float)
            x = np.array([p[1] for p in pts[i:j + 1]], float)
            y = np.array([p[2] for p in pts[i:j + 1]], float)
            # разрывы наблюдений внутри окна недопустимы
            if len(f) > 1 and int(np.max(np.diff(f))) > gap_break:
                continue
            fit = _fit_ballistic(f, x, y, fps)
            if fit is None:
                continue
            if fit["rmse_px"] > max_rmse_frac * frame_small:
                continue
            # гравитационный тест: вертикальное ускорение ≈ +g (ось y вниз)
            if abs(fit["ay"] - gravity_px_s2) > grav_tol_rel * gravity_px_s2:
                continue
            # заметное горизонтальное перемещение (иначе — не перелёт)
            if abs(fit["vx"]) * (f[-1] - f[0]) / fps < min_horizontal_disp_frac * frame_width \
                    and abs(x[-1] - x[0]) < min_horizontal_disp_frac * frame_width:
                continue
            seg_pts = pts[i:j + 1]
            direction, ratio = classify_direction(seg_pts, frame_width)
            segments.append(ParabolicSegment(
                start_frame=int(f[0]), end_frame=int(f[-1]),
                points=seg_pts, vx_px_s=fit["vx"], vy0_px_s=fit["vy0"],
                ay_px_s2=fit["ay"], rmse_px=fit["rmse_px"],
                apex_y_px=fit["apex_y_px"], direction=direction,
                to_net_ratio=ratio))
            i = j + 1
            placed = True
            break
        if not placed:
            i += 1
    return segments


def classify_direction(seg_pts: list[tuple[int, float, float]],
                       frame_width: int) -> tuple[str, float]:
    """Направление участка относительно сетки (центр кадра = проекция сетки).

    "left"  — мяч летит влево К сетке; "right" — вправо К сетке;
    "away"  — летит от сетки (или почти без горизонтального движения).
    """
    x0, x1 = seg_pts[0][1], seg_pts[-1][1]
    net_x = frame_width / 2.0
    dx = x1 - x0
    d0 = abs(x0 - net_x)
    d1 = abs(x1 - net_x)
    approaching = d1 < d0 * 0.85          # финиш заметно ближе к сетке, чем старт
    if abs(dx) < 1e-6 or not approaching:
        return ("away", 1.0 if d0 == 0 else d1 / max(d0, 1e-6))
    direction = "left" if dx < 0 else "right"
    ratio = d1 / max(d0, 1e-6)
    return (direction, ratio)


def split_rallies(ball_points: list[tuple[int, float, float]],
                  gap_frames: int) -> list[list[tuple[int, float, float]]]:
    """Группировка точек видимого мяча в эпизоды-розыгрыши по разрывам видимости."""
    if not ball_points:
        return []
    pts = sorted(ball_points, key=lambda p: p[0])
    rallies: list[list[tuple[int, float, float]]] = [[pts[0]]]
    for prev, cur in zip(pts, pts[1:]):
        if cur[0] - prev[0] > gap_frames:
            rallies.append([cur])
        else:
            rallies[-1].append(cur)
    return rallies


def detect_passes(ball_points: list[tuple[int, float, float]], fps: float,
                  width: int, height: int, *,
                  rally_gap_frames: int,
                  par_min_frames: int, par_max_frames: int,
                  par_max_rmse_frac: float, gravity_px_s2: float,
                  grav_tol_rel: float, min_flight_frames: int,
                  min_horizontal_disp_frac: float) -> tuple[list[Rally], list[PassSegment]]:
    """Главный вход: точки трека мяча → розыгрыши → пасы (параболы к сетке)."""
    rallies_raw = split_rallies(ball_points, rally_gap_frames)
    rallies: list[Rally] = []
    passes: list[PassSegment] = []
    small = min(width, height)
    for rp in rallies_raw:
        frames = [p[0] for p in rp]
        rally = Rally(start_frame=min(frames), end_frame=max(frames))
        # фазы по стороне относительно сетки (гистерезис)
        side = "unknown"
        phase_start = rally.start_frame
        net_x = width / 2.0
        hyst = net_hysteresis_px(width)
        for f, x, y in rp:
            new_side = side
            if x < net_x - hyst:
                new_side = "left"
            elif x > net_x + hyst:
                new_side = "right"
            if new_side != side and side != "unknown" and new_side != "unknown":
                rally.phases.append((phase_start, f, side))
                phase_start = f
            if new_side != "unknown":
                side = new_side
        rally.phases.append((phase_start, rally.end_frame, side))
        rallies.append(rally)

        # параболические участки по всему розыгрышу
        segs = find_parabolic_segments(
            rp, fps,
            min_frames=par_min_frames, max_frames=par_max_frames,
            max_rmse_frac=par_max_rmse_frac, gravity_px_s2=gravity_px_s2,
            grav_tol_rel=grav_tol_rel, gap_break=rally_gap_frames,
            min_horizontal_disp_frac=min_horizontal_disp_frac,
            frame_small=small, frame_width=width)
        # пас = участок «к сетке», достаточной длины, не продолжающий тот же
        # полёт (между соседними участками одного направления обязана быть пауза)
        last: PassSegment | None = None
        for s in segs:
            if s.direction == "away":
                last = None
                continue
            if s.end_frame - s.start_frame + 1 < min_flight_frames:
                continue
            if last is not None and s.start_frame - last.catch_frame <= 2 \
                    and s.direction == last.direction:
                # склейка осколков одного полёта: расширяем предыдущий пас
                last.catch_frame = s.end_frame
                last.segment.points.extend(s.points)
                last.segment.end_frame = s.end_frame
                continue
            ps = PassSegment(release_frame=s.start_frame, catch_frame=s.end_frame,
                             direction=s.direction, segment=s)
            passes.append(ps)
            last = ps
    return rallies, passes


def net_hysteresis_px(width: int) -> float:
    """Гистерезис определения стороны относительно сетки, px."""
    return max(20.0, width * 0.03)
