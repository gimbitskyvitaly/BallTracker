"""Постобработка траектории: удаление точек-выбросов (ложные захваты).

Мотивация (реальное видео VID_20260925_163505.mp4, job 75884ccef424): в
сегменте полёта трекер местами «срывается» на ложную область — одна точка
улетает на сотни пикселей от соседей (t=0.398 x_px=972 при соседях 1131/1152;
участок «зависания» с последующим прыжком на 148 px), и траектория перестаёт
быть параболической.

Критерий ровно из постановки: точка i — выброс, если она СИЛЬНО отличается
от i-1, а i-1 находится РЯДОМ с i+1 (соседние «хорошие» точки согласованы
между собой, а рассматриваемая выпадает из их локального масштаба движения).

Масштаб «сильно отличается» — ЛОКАЛЬНЫЙ: медиана попарных расстояний между
СОСЕДНИМИ точками в окне ±window вокруг проверяемой пары (без самих
проверяемых точек). Это принципиально: шаг мяча вдоль траектории меняется
на порядки (от ~1 px в замедлении/зависании до 100+ px на быстром участке),
поэтому глобальный порог либо молчал бы, либо резал нормальные данные.
Локальная медиана устойчива к единичному ложному захвату (он в меньшинстве)
и адаптируется к скорости мяча.

Алгоритм итеративный: после удаления точки её соседи становятся «предыдущей»
и «следующей», поэтому распознаются и цепочки из 2–3 подряд идущих ложных
захватов.

Гарантии сохранности сервиса (мяч продолжает детектиться, списки не пусты):
  * крайние точки (первая/последняя) не удаляются никогда — релиз/приёмка
    остаются на своих местах, классификатор _is_pass не теряет концы дуги;
  * суммарно удаляется не более max_remove_frac точек (по умолчанию 25%);
  * cleaned всегда содержит не менее min_keep (4) точек;
  * очистка НЕ меняет длину/тайминги сегмента: освобождённые кадры
    заполняются интерполяцией по времени между уцелевшими соседями —
    release_frame/catch_frame/time_of_flight остаются прежними, физика
    (polyfit/RK4-фит) просто перестает «тянуться» за фантомными точками;
  * при любом недостатке данных или исключении возвращается исходный список.
"""

from __future__ import annotations

import math


def _median(values: list[float]) -> float:
    s = sorted(values)
    m = len(s)
    if not m:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def _adjacent_steps(pts: list[tuple[float, float]], bad: list[bool],
                    center: int, radius: int, n_pts: int) -> list[float]:
    """Расстояния между СОСЕДНИМИ точками в окне ±radius от center.

    Проверяемая точка из окна исключена (bad-точки тоже): это «чем дышит
    мяч рядом», устойчивая оценка локального шага."""
    lo, hi = max(0, center - radius), min(n_pts - 1, center + radius)
    ds: list[float] = []
    for m in range(lo, hi):
        a, b = m, m + 1
        if a == center or b == center or bad[a] or bad[b]:
            continue
        ds.append(math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
    return ds


def _local_step_scale(pts: list[tuple[float, float]], bad: list[bool],
                      i: int, j: int, radius: int, n_pts: int) -> float | None:
    """Типичный шаг между точками рядом с парой (i, j)."""
    ds = _adjacent_steps(pts, bad, i, radius, n_pts) + \
        _adjacent_steps(pts, bad, j, radius, n_pts)
    if len(ds) < 4:
        return None
    return max(_median(ds), 1.0)


def mark_outliers(points: list, xy=None, *, window: int = 3, k: float = 3.0,
                  min_abs_px: float = 12.0, max_remove_frac: float = 0.25,
                  min_keep: int = 4, max_rounds: int = 8) -> list[bool] | None:
    """Помечает точки-выбросы. Возвращает mask (True = выброс) или None,
    если чистка невозможна/небезопасна (слишком мало точек, лимиты, ошибка).

    points: список anything; xy — callable(point) -> (x, y). По умолчанию
    берутся ключи "x_px"/"y_px" (формат trajectory API).
    """
    try:
        n = len(points)
        if n < 6:                       # чистить нечего / риск сломать фит
            return None
        getxy = xy or (lambda p: (float(p["x_px"]), float(p["y_px"])))
        pts = [getxy(p) for p in points]
        radius = max(2, min(window, n - 1))
        max_removed = max(1, int(n * max_remove_frac))
        bad = [False] * n
        removed_total = 0

        for _ in range(max_rounds):
            changed = False
            kept = [i for i in range(n) if not bad[i]]
            if len(kept) <= min_keep:
                break
            for pos_idx in range(1, len(kept) - 1):   # края не трогаем никогда
                pos = kept[pos_idx]
                pi_, ni_ = kept[pos_idx - 1], kept[pos_idx + 1]
                cur = pts[pos]
                d_prev = math.hypot(cur[0] - pts[pi_][0], cur[1] - pts[pi_][1])
                d_next = math.hypot(pts[ni_][0] - cur[0], pts[ni_][1] - cur[1])
                d_pn = math.hypot(pts[ni_][0] - pts[pi_][0],
                                  pts[ni_][1] - pts[pi_][1])
                sa = _local_step_scale(pts, bad, pi_, pos, radius, n)
                sb = _local_step_scale(pts, bad, pos, ni_, radius, n)
                if sa is None or sb is None:
                    continue
                # локальный масштаб шага СТОРОН СОГЛАСОВАННОСТИ (i-1 и i+1):
                # сравниваем d(i,i-1) с типичным шагом у i-1, а d(i,i+1) — у
                # i+1. Если бы вместо этого брать min(sa,sb), «быстрая» сторона
                # завышала порог для «спокойной» и одиночный выброс на границе
                # медленного/быстрого участка оставался незамеченным.
                # floor = min_abs_px защищает от шумового нуля в зависаниях.
                la = max(k * sa, min_abs_px)
                lb = max(k * sb, min_abs_px)
                # i далеко от обоих соседей, а сами соседи согласованы между
                # собой (не дальше локального шага) => i — ложная детекция
                if d_prev > la and d_next > lb and d_pn <= max(la, lb):
                    bad[pos] = True
                    removed_total += 1
                    changed = True
                    if removed_total >= max_removed or \
                            len(kept) - removed_total < min_keep:
                        break
            if not changed:
                break

        if removed_total == 0 or n - removed_total < min_keep:
            return None
        return bad
    except Exception:
        return None


def _interp_xy(xa: float, ya: float, xb: float, yb: float,
               ta: float, tb: float, t: float) -> tuple[float, float]:
    f = 0.5 if tb <= ta else (t - ta) / (tb - ta)
    f = min(1.0, max(0.0, f))
    return xa + (xb - xa) * f, ya + (yb - ya) * f


def clean_point_dicts(points: list[dict], coord_keys=("x_px", "y_px"),
                      time_key: str | None = None, **kw) -> list[dict]:
    """Очистка списка dict-точек с сохранением длины.

    Освобождённые кадры заполняются линейной интерполяцией координат между
    уцелевшими соседями (по time_key, иначе по индексу) — трек остаётся
    непрерывным покадрово, но ложных областей в нём больше нет.
    При любой неудаче возвращается исходный список без изменений.
    """
    mask = mark_outliers(points, lambda p: tuple(float(p[c]) for c in coord_keys),
                         **kw)
    if mask is None:
        return points
    n = len(points)
    kept_idx = [i for i, b in enumerate(mask) if not b]
    out: list[dict] = []
    for i in range(n):
        if not mask[i]:
            out.append(points[i])
            continue
        # позиция среди уцелевших: между left и right
        lo = max(j for j in kept_idx if j < i)
        hi = min(j for j in kept_idx if j > i)
        t = (lambda k: float(points[k].get(time_key, k))) if time_key \
            else (lambda k: float(k))
        nx, ny = _interp_xy(float(points[lo][coord_keys[0]]),
                            float(points[lo][coord_keys[1]]),
                            float(points[hi][coord_keys[0]]),
                            float(points[hi][coord_keys[1]]),
                            t(lo), t(hi), t(i))
        newp = dict(points[i])
        newp[coord_keys[0]] = round(nx, 1)
        newp[coord_keys[1]] = round(ny, 1)
        out.append(newp)
    return out


def clean_track_points(track: list[dict], width: int, height: int,
                       fps: float, **kw) -> list[dict]:
    """Очистка ball_track_points ({frame,x,y}) для рендера/трека.

    Прыжки между КАДРАМИ нормируются ожидаемым смещением кадра
    (fps * кадр-скорость), т.к. шаг зависит от fps; пороги мягче, чем для
    сегментов полёта (трек включает медленные фазы владения). Точки НЕ
    удаляются (кадры рендера фиксированы) — ложные позиции заменяются
    интерполяцией. Гарантии: края не трогаются, <=20% правок, при ошибке —
    исходный список."""
    try:
        n = len(track)
        if n < 8:
            return track
        v_norm = jump_per_frame(width, height, fps)
        pts = [(float(p["x"]), float(p["y"])) for p in track]

        def steps(center: int, radius: int) -> list[float]:
            lo, hi = max(0, center - radius), min(n - 1, center + radius)
            return [math.hypot(pts[m][0] - pts[m + 1][0],
                               pts[m][1] - pts[m + 1][1])
                    for m in range(lo, hi)]

        bad = [False] * n
        radius = 4
        max_fix = max(1, int(n * 0.2))
        fixed = 0
        for idx in range(1, n - 1):          # края не трогаем
            d_prev = math.hypot(pts[idx][0] - pts[idx - 1][0],
                                pts[idx][1] - pts[idx - 1][1])
            d_next = math.hypot(pts[idx + 1][0] - pts[idx][0],
                                pts[idx + 1][1] - pts[idx][1])
            d_pn = math.hypot(pts[idx + 1][0] - pts[idx - 1][0],
                              pts[idx + 1][1] - pts[idx - 1][1])
            med = max(_median(steps(idx - 1, radius)),
                      _median(steps(idx + 1, radius)), 1.0)
            lim = max(3.0 * med, 0.012 * min(width, height),
                      3.0 * v_norm)
            if d_prev > lim and d_next > lim and d_pn <= lim:
                bad[idx] = True
                fixed += 1
                if fixed >= max_fix:
                    break
        if fixed == 0:
            return track
        out = []
        last_good = 0
        for i, p in enumerate(track):
            if not bad[i]:
                out.append(p)
                last_good = i
                continue
            nxt = next((j for j in range(i + 1, n) if not bad[j]), n - 1)
            f = (i - last_good) / max(1, (nxt - last_good))
            g, q = track[last_good], track[nxt]
            out.append({"frame": p["frame"],
                        "x": round(float(g["x"])
                                   + (float(q["x"]) - float(g["x"])) * f, 1),
                        "y": round(float(g["y"])
                                   + (float(q["y"]) - float(g["y"])) * f, 1)})
        return out
    except Exception:
        return track


def jump_per_frame(width: int, height: int, fps: float) -> float:
    """Ожидаемый максимум смещения мяча за один кадр (px): доля меньшего
    измерения кадра, поправка на fps (30 fps baseline)."""
    return 0.02 * min(width, height) * (30.0 / max(fps, 1.0))
