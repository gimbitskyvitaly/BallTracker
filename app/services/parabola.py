"""Постобработка траектории: парабола по «чистому» участку полёта (ТЗ-алгоритм).

В одной траектории (один пас = один цвет на рендере), когда ВСЕ точки уже
известны, находится участок параболической формы — непрерывная цепочка точек,
лежащих вблизи одной параболы (МСК-ядро + итеративная обрезка выбросов +
мягкое расширение contiguum-участка). По этому участку методом МНК fit'ится
парабола y = a*x^2 + b*x + c, и вся траектория отрисовывается этой параболой
ОТ пересечения одной её ветви с линиями сетки ДО пересечения другой ветви с
сеткой: границы фита продлеваются за пределы чистого участка до ближайших
вертикальных линий сетки (разметки площадки) слева/справа от вершины;
горизонтальные линии сетки дополнительно обрезают концы дуги.

Почему предыдущая точечная чистка не помогала: медианный локальный масштаб
шага залатывал ОДИНОЧные шипы, но реальные ложные захваты на этом видео —
«хвосты» из 4–6 подряд фантомных кадров (зависание CSRT на фоне + прыжок на
100+ px), а интерполяция между уцелевшими соседями внутри такого хвоста сама
давала не-параболические куски. Здесь форма траектории ЗАДАНА параболой,
выращенной из чистого участка: ложные области в итоговую кривую не попадают
принципиально.

Гарантии сохранности сервиса (мяч продолжает детектиться, списки не пусты):
  * при любой неудаче (мало точек, вырожденная геометрия, исключение) —
    возвращается ИСХОДНЫЙ список без изменений;
  * крайние точки (релиз/приёмка) сохраняются побитово, метрики
    release_frame/catch_frame/time_of_flight не меняются; стык концов с
    дугой доклеивается короткими сегментами;
  * выходной список никогда не пуст (проверяется вызывающим кодом).
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------- утилиты
def _dedupe(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Убирает дословные повторы координат (трек «зависает» при окклюзии)."""
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def _polyval(c: tuple[float, float, float], x):
    a, b, d = c
    return a * x * x + b * x + d


def _fit_parabola(xs: np.ndarray, ys: np.ndarray, min_curv_px: float = 2.0):
    """y = a x^2 + b x + c по МСК; None если вырождено / почти прямая.

    min_curv_px — минимальная «стрела» кривизны a*span^2 (px), отличающая
    дугу от прямой. При итеративной очистке выбросов передаётся меньшим
    (фит по чистым инлайнерам обязан иметь кривизну реальной дуги)."""
    try:
        if len(xs) < 5 or float(np.ptp(xs)) < 1.0:
            return None
        a, b, c = np.polyfit(xs, ys, 2)
    except Exception:  # noqa: BLE001 — плохая обусловленность и т.п.
        return None
    if not np.isfinite([a, b, c]).all():
        return None
    span = float(xs.max() - xs.min())
    if abs(a) * span * span < min_curv_px:
        return None                     # кривизны нет — это не дуга
    return (float(a), float(b), float(c))


def _contiguous_arc(ux: np.ndarray, uy: np.ndarray, tol_px: float,
                    min_inliers: int, min_span_px: float):
    """Перебор всех contiguum-окон последовательности: находит САМОЕ длинное
    окно, точки которого лежат вблизи одной параболы (участок параболической
    формы). Внутри окна — итеративная МНК-очистка (Trimmed Least Squares).

    Почему перебор, а не один глобальный фит: ложные захваты образуют
    локальные «хвосты» по 4–6 подряд фантомных кадров — глобальная парабола
    по всем точкам (включая хвосты) не существует, а чистый участок между
    ними есть всегда. Возвращает (i_lo, i_hi, coef) или None."""
    n = len(ux)
    best = None
    for i in range(n - min_inliers + 1):
        for j in range(n - 1, i + min_inliers - 2, -1):
            if best is not None and (j - i) <= (best[1] - best[0]):
                break                       # дальше уже не длиннее
            if float(ux[j] - ux[i]) < min_span_px:
                continue
            seg_x, seg_y = ux[i:j + 1], uy[i:j + 1]
            c = _fit_parabola(seg_x, seg_y, min_curv_px=1.0)
            if c is None:
                continue
            keep = np.ones(len(seg_x), dtype=bool)
            for _ in range(8):
                dev = np.abs(seg_y - np.array([_polyval(c, x) for x in seg_x]))
                nk = dev <= tol_px
                if int(nk.sum()) < min_inliers or (nk == keep).all():
                    keep = nk if int(nk.sum()) >= min_inliers else keep
                    break
                nc = _fit_parabola(seg_x[nk], seg_y[nk])
                if nc is None:
                    break
                keep, c = nk, nc
            if int(keep.sum()) < min_inliers:
                continue
            kx, ky = seg_x[keep], seg_y[keep]
            if float(kx.max() - kx.min()) < min_span_px:
                continue
            rms = float(np.sqrt(np.mean((ky - np.array([_polyval(c, x)
                                                       for x in kx])) ** 2)))
            cand = (i, j, c, int(keep.sum()), rms)
            if best is None or (cand[3], -cand[4]) > (best[3], -best[4]):
                best = cand
    if best is None:
        return None
    return best[0], best[1], best[2]


# ------------------------------------------------------- линии сетки кадра
def detect_grid_lines(frames_bgr: list[np.ndarray], width: int, height: int
                      ) -> tuple[list[float], list[float]]:
    """Позиции линий сетки/разметки (BGR-кадры могут быть даунскейлены).

    Линии волейбольной разметки — длинные светлые прямые. Детект: маска
    яркости + морфологическое «открытие» длинными горизонтальным/вертикальным
    ядрами (остаются только протяжённые прямые), центроиды компонент. Плюс
    вертикальная стойка/полотно самой сетки: колонка с очень высокой долей
    ярких пикселей (белая топ-лента) голосуется по всем кадрам.

    Возвращает (xs_вертикальных_линий, ys_горизонтальных_линий) в координатах
    ПОЛНОГО кадра; если ничего не найдено — равномерная вспомогательная
    сетка (фит всегда имеет границы).
    """
    fallback = ([width * i / 8.0 for i in range(1, 8)],
                [height * i / 4.0 for i in range(1, 4)])
    if not frames_bgr:
        return fallback
    h0, w0 = frames_bgr[0].shape[:2]
    sx, sy = width / max(w0, 1), height / max(h0, 1)
    k_long_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w0 // 10), 1))
    k_long_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h0 // 6)))
    acc_v: dict[int, list[float]] = {}
    acc_h: dict[int, list[float]] = {}
    net_col_votes: dict[int, int] = {}
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, 170, 255)
        vert = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_long_v)
        horz = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_long_h)
        for img, acc, min_len, coord in ((vert, acc_v, h0 // 4, 0),
                                         (horz, acc_h, w0 // 5, 1)):
            n, _, stats, cents = cv2.connectedComponentsWithStats(img)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                ln = h if coord == 0 else w
                if ln < min_len or area < 1.5 * ln:
                    continue
                pos = cents[i][0] if coord == 0 else cents[i][1]
                key = int(round(pos / 15.0))          # кластеризация ±7px
                acc.setdefault(key, []).append(float(pos))
        col_frac = bright.mean(axis=0)
        for cx_ in np.nonzero(col_frac > 0.5)[0]:
            key = int(round(float(cx_) / 15.0))
            net_col_votes[key] = net_col_votes.get(key, 0) + 1
    vx = sorted(sum(v) / len(v) * sx for v in acc_v.values())
    hy = sorted(sum(v) / len(v) * sy for v in acc_h.values())
    net_cols = [k * 15.0 * sx for k, cnt in net_col_votes.items()
                if cnt >= max(2, len(frames_bgr) // 3)]
    merged = sorted(set([round(x, 1) for x in vx]
                        + [round(x, 1) for x in net_cols]))
    # склейка близких линий (< 25 px — одна и та же линия из разных кадров)
    vx_merged: list[float] = []
    for x in merged:
        if not vx_merged or x - vx_merged[-1] > 25.0:
            vx_merged.append(x)
    if not vx_merged and not hy:
        return fallback
    return (vx_merged or fallback[0], hy or fallback[1])


# ------------------------------------------------------------ основной API
def parabolicize(points: list[dict], *, x_key: str = "x_px", y_key: str = "y_px",
                 time_key: str | None = "t",
                 grid: tuple[list[float], list[float]] | None = None,
                 width: int = 1920, height: int = 1080, tol_px: float = 15.0,
                 min_inliers: int = 8, min_span_frac: float = 0.15,
                 extend_frac: float = 0.5) -> list[dict]:
    """Публичный вход: trajectory-API-совместимый список → очищенный фитом.

    points — dict-точки с ключами x_key/y_key (остальные ключи первой точки
    копируются в новые точки). При неудаче — исходный список без изменений.
    """
    try:
        out = _parabolicize(points, x_key, y_key, time_key, grid, width,
                            height, tol_px, min_inliers, min_span_frac,
                            extend_frac)
        if isinstance(out, list) and len(out) >= 3:
            return out
        return points
    except Exception:  # noqa: BLE001 — постобработка не должна валить job
        return points


def _parabolicize(points, x_key, y_key, time_key, grid, width, height,
                  tol_px, min_inliers, min_span_frac, extend_frac):
    n = len(points)
    if n < min_inliers:
        return points
    raw = [(float(p[x_key]), float(p[y_key])) for p in points]
    uniq = _dedupe(raw)
    if len(uniq) < min_inliers:
        return points
    ux = np.array([p[0] for p in uniq])
    uy = np.array([p[1] for p in uniq])

    # --- фазы 1–2: поиск УЧАСТКА параболической формы ----------------------
    # перебор contiguum-окон + итеративная МНК-очистка внутри лучшего окна
    found = _contiguous_arc(ux, uy, tol_px, min_inliers,
                            min_span_frac * width)
    if found is None:
        return points
    lo_i, hi_i, c = found
    seg_x, seg_y = ux[lo_i:hi_i + 1], uy[lo_i:hi_i + 1]
    span = float(seg_x.max() - seg_x.min())
    if span < min_span_frac * width:
        return points

    # --- границы фита: от пересечения одной ветви с сеткой до другой -------
    x_lo, x_hi = float(seg_x.min()), float(seg_x.max())
    if grid is None:
        grid = ([width * i / 8.0 for i in range(1, 8)],
                [height * i / 4.0 for i in range(1, 4)])
    vx, hy = grid
    a, b, d = c
    vertex_x = -b / (2 * a) if a else 0.5 * (x_lo + x_hi)
    max_ext = extend_frac * span
    left_c = [v for v in vx if v < x_lo and x_lo - v <= max_ext]
    right_c = [v for v in vx if v > x_hi and v - x_hi <= max_ext]
    gx_lo = max(left_c) if left_c else x_lo
    gx_hi = min(right_c) if right_c else x_hi
    # горизонтальные линии сетки, пересекающие дугу, срезают «хвосты» ветвей
    cuts: list[float] = []
    if a:
        for yline in hy:
            d2 = b * b - 4 * a * (d - yline)
            if d2 < 0:
                continue
            r = math.sqrt(d2)
            for root in ((-b + r) / (2 * a), (-b - r) / (2 * a)):
                if gx_lo <= root <= gx_hi:
                    cuts.append(float(root))
    left_cut = [t for t in cuts if gx_lo <= t <= vertex_x]
    right_cut = [t for t in cuts if vertex_x <= t <= gx_hi]
    if left_cut:
        gx_lo = max(gx_lo, max(left_cut))
    if right_cut:
        gx_hi = min(gx_hi, min(right_cut))
    if gx_hi - gx_lo < 0.6 * span:      # сетка съела почти всю дугу — откат
        gx_lo, gx_hi = x_lo, x_hi
    gx_lo = max(gx_lo, 0.0)
    gx_hi = min(gx_hi, float(width))

    # --- фаза 3: сборка новой траектории ------------------------------------
    arc_n = int(max(n, min(240, (gx_hi - gx_lo) / 4.0)))
    arc_x = np.linspace(gx_lo, gx_hi, arc_n)
    arc = [(float(x), float(_polyval(c, x))) for x in arc_x]
    t_first = float(points[0][time_key]) if time_key else 0.0
    t_last = float(points[-1][time_key]) if time_key else float(n - 1)
    lens = [0.0]
    for i in range(1, len(arc)):
        lens.append(lens[-1] + math.hypot(arc[i][0] - arc[i - 1][0],
                                          arc[i][1] - arc[i - 1][1]))
    total = max(lens[-1], 1e-9)

    # метрический масштаб м/px (для x_m/height_m в формате trajectory API)
    scale = None
    if "x_m" in points[0]:
        px_span = max(float(np.ptp(ux)), 1e-9)
        m_span = float(np.ptp([float(p["x_m"]) for p in points]))
        scale = m_span / px_span if px_span > 0 else None
    base = points[0]
    y_ref = float(base[y_key])

    def mk(xy: tuple[float, float], t: float) -> dict:
        q = dict(base)
        if time_key:
            q[time_key] = round(t, 3)
        q[x_key] = round(xy[0], 1)
        q[y_key] = round(xy[1], 1)
        if scale is not None:
            if "x_m" in q:
                q["x_m"] = round(xy[0] * scale, 3)
            if "height_m" in q:
                q["height_m"] = round((y_ref - xy[1]) * scale, 3)
            if "y_m" in q:
                q["y_m"] = round(xy[1] * scale, 3)
        return q

    first_xy, last_xy = raw[0], raw[-1]
    out: list[dict] = []
    # релиз остаётся на своём месте; к дуге ведёт короткий линейный стык
    gap_f = math.hypot(first_xy[0] - arc[0][0], first_xy[1] - arc[0][1])
    k_f = max(1, int(gap_f / 25)) if gap_f > 3.0 else 0
    for j in range(k_f):
        f = j / (k_f + 1)
        out.append(mk((first_xy[0] + (arc[0][0] - first_xy[0]) * f,
                       first_xy[1] + (arc[0][1] - first_xy[1]) * f), t_first))
    # дуга: тайминг пропорционален длине дуги между t релиза и t приёмки
    for (ax_, ay_), s in zip(arc, lens):
        out.append(mk((ax_, ay_), t_first + (t_last - t_first) * s / total))
    # приёмка остаётся на своём месте; стык от конца дуги к ней
    gap_l = math.hypot(last_xy[0] - arc[-1][0], last_xy[1] - arc[-1][1])
    k_l = max(1, int(gap_l / 25)) if gap_l > 3.0 else 0
    for j in range(1, k_l + 1):
        f = j / (k_l + 1)
        t_prev = float(out[-1][time_key]) if time_key else t_last
        t = t_prev + (t_last - t_prev) * f
        out.append(mk((arc[-1][0] + (last_xy[0] - arc[-1][0]) * f,
                       arc[-1][1] + (last_xy[1] - arc[-1][1]) * f), t))
    return out
