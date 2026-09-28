"""Фильтрация выбросов в траектории мяча.

Симптом, который чинит этот модуль: на части кадров полёта детектор ложно
захватывает другую область (игрок, фон, другой объект), и в траектории
появляются точки, далеко отстоящие от соседей — «полёт» перестаёт быть
параболическим (например: i-1 и i+1 стоят рядом друг с другом, а i улетела
на сотни пикселей).

Правило из постановки задачи: если точка i сильно отличается от соседей, а
соседи i-1 и i+1 близки между собой — точка i есть выброс (ложная детекция)
и учитывать её в траектории не нужно.

Конвейер фильтрации:
  1. V-выбросы: для внутренней точки i сравниваем d(i-1, i+1) и
     max(d(i-1, i), d(i, i+1)). Соседи прижаты друг к другу, а точка
     оторвана от обоих более чем на адаптивный порог
     max(min_jump_px, k * медианного шага) — удаляем. На медленных участках
     нормальные точки не выкидываем, на быстрых обычный большой шаг между
     кадрами выбросом не считаем.
  2. Разрывные сегменты: после удаления одиночных выбросов траектория может
     распасться на куски (прыжок между соседними наблюдениями > seg_gap_px —
     трек перескочил на ложную область). Куски короче min_seg_points точек
     — следы ложных детекций, вычищаем целиком.
  3. Баллистический RANSAC по каждому оставшемуся связному сегменту:
     x(t) ~ линейно, y(t) ~ квадратично (это ровно парабола полёта). Точки
     с ошибкой больше tol_px — ложные; так отсекаются «компактные» кластеры
     ложных детекций (несколько согласованных внутри себя кадров на неверной
     области), которые проходы 1-2 не видят. Сегменты плотнее 2 кадров/ед.
     времени (застой трека на одном месте) не фильтруются — там нет
     информации для различения инлиеров.

Гарантии (сервис обязан продолжать детектировать мяч):
  * список точек никогда не становится пустым и не теряет крайние
    наблюдения всего полёта (release/catch — границы метрик ToF);
  * если после фильтрации осталось меньше min_keep точек — возвращаем
    исходный список без изменений;
  * вход короче 5 точек не фильтруется.
"""

from __future__ import annotations

import math

import numpy as np


def _dist(a, b) -> float:
    return math.hypot(a[1] - b[1], a[2] - b[2])


def _drop_v_outliers(pts: list[tuple[int, float, float]],
                     min_jump_px: float, k_step: float) -> list[tuple[int, float, float]]:
    """Проход 1: одиночные V-образные выбросы (крайние точки не трогаем)."""
    pts = list(pts)
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        steps = [_dist(pts[i + 1], pts[i]) for i in range(len(pts) - 1)]
        med = float(np.median(steps)) if steps else 0.0
        thr = max(min_jump_px, k_step * med)
        for i in range(1, len(pts) - 1):
            d_prev = _dist(pts[i], pts[i - 1])
            d_next = _dist(pts[i + 1], pts[i])
            d_nbrs = _dist(pts[i + 1], pts[i - 1])
            if d_nbrs <= thr and max(d_prev, d_next) > thr:
                pts.pop(i)
                changed = True
                break
    return pts


def _split_segments(pts: list[tuple[int, float, float]],
                    seg_gap_px: float) -> list[list[tuple[int, float, float]]]:
    """Разбивает серию наблюдений на куски по разрывам траектории."""
    segs: list[list[tuple[int, float, float]]] = [[pts[0]]]
    for p, q in zip(pts, pts[1:]):
        if _dist(p, q) > seg_gap_px:
            segs.append([q])
        else:
            segs[-1].append(q)
    return segs


def _ransac_parabola_mask(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                          tol_px: float, rng: np.random.Generator,
                          iterations: int) -> np.ndarray | None:
    """RANSAC-фит параболы. Возвращает маску инлиеров или None, если фит
    ненадёжен (тогда ничего не удаляем)."""
    n = len(t)
    span_t = t[-1] - t[0]
    if n < 6 or span_t <= 0:
        return None
    best_inl = np.zeros(n, dtype=bool)
    best_n = -1
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        tt = t[idx]
        if tt.max() - tt.min() < 0.2 * span_t:
            continue          # слишком узкая выборка — плохая обусловленность
        try:
            cx = np.polyfit(tt, x[idx], 1)
            cy = np.polyfit(tt, y[idx], 2)
        except Exception:
            continue
        rx = np.abs(np.polyval(cx, t) - x)
        ry = np.abs(np.polyval(cy, t) - y)
        inl = (rx <= tol_px) & (ry <= tol_px)
        if int(inl.sum()) > best_n:
            best_n = int(inl.sum())
            best_inl = inl
    if best_n < max(4, int(0.3 * n)):
        return None           # доверия к фиту нет — ничего не удаляем
    try:
        cx = np.polyfit(t[best_inl], x[best_inl], 1)
        cy = np.polyfit(t[best_inl], y[best_inl], 2)
    except Exception:
        return best_inl
    rx = np.abs(np.polyval(cx, t) - x)
    ry = np.abs(np.polyval(cy, t) - y)
    return (rx <= tol_px) & (ry <= tol_px)


def _fit_segment(seg: list[tuple[int, float, float]], fps: float, tol_px: float,
                 rng: np.random.Generator, iters: int,
                 keep_first: bool, keep_last: bool) -> list[tuple[int, float, float]]:
    """Проход 3: RANSAC-парабола по связному сегменту. Крайние точки сегмента
    (а для первого/последнего сегмента — границы полёта) не удаляем."""
    if len(seg) < 6:
        return seg
    frames = np.array([p[0] for p in seg], dtype=float)
    # застой трека (мяч «прилип» к месту): густая решётка без движения —
    # парабола вырождена, фильтр бесполезен и опасен
    dt_med = float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0
    if dt_med > 2.5:
        return seg
    t = (frames - frames[0]) / max(fps, 1e-6)
    x = np.array([p[1] for p in seg], dtype=float)
    y = np.array([p[2] for p in seg], dtype=float)
    mask = _ransac_parabola_mask(t, x, y, tol_px, rng, iters)
    if mask is None:
        return seg
    out = []
    for i, (p, m) in enumerate(zip(seg, mask)):
        edge = (keep_first and i == 0) or (keep_last and i == len(seg) - 1)
        if m or edge:
            out.append(p)
    return out if len(out) >= 4 else seg


def filter_flight_outliers(tracked_points: list[tuple[int, float, float]], *,
                           fps: float,
                           min_jump_px: float = 60.0,
                           k_step: float = 3.0,
                           seg_gap_px: float = 250.0,
                           min_seg_points: int = 4,
                           tol_px: float = 25.0,
                           ransac_iters: int = 600,
                           seed: int = 12345,
                           min_keep: int = 5) -> list[tuple[int, float, float]]:
    """Чистит сегмент полёта от ложных точек (см. docstring модуля).

    tracked_points: [(frame_id, cx_px, cy_px), ...] — один связный полёт.
    Возвращает подмножество тех же кортежей (никогда не пустой; при
    недостатке данных — без изменений).
    """
    pts = list(tracked_points)
    n0 = len(pts)
    if n0 < 5:
        return pts

    cleaned = _drop_v_outliers(pts, min_jump_px=min_jump_px, k_step=k_step)
    segments = _split_segments(cleaned, seg_gap_px=seg_gap_px)
    # проход 2: короткие осколки (следы ложных областей) — долой,
    # но первый и последний сегменты всегда сохраняем (границы релиза/приёмки)
    if len(segments) > 1:
        kept_segs = []
        for si, seg in enumerate(segments):
            is_edge = si == 0 or si == len(segments) - 1
            if len(seg) >= min_seg_points or is_edge:
                kept_segs.append(seg)
        if kept_segs:
            segments = kept_segs

    # проход 3: баллистический RANSAC по каждому сегменту
    rng = np.random.default_rng(seed)
    final: list[tuple[int, float, float]] = []
    for si, seg in enumerate(segments):
        final.extend(_fit_segment(seg, fps, tol_px, rng, ransac_iters,
                                  keep_first=(si == 0),
                                  keep_last=(si == len(segments) - 1)))

    # страховка: фильтрация не должна обескровить трек
    if len(final) < max(min_keep, 4) or len(final) > n0:
        return pts
    return final
