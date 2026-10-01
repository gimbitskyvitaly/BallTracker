"""Фильтрация траектории мяча: ложные детекции и уровень сетки.

1) ЛОЖНЫЕ ДЕТЕКЦИИ (изолированные выбросы). Точка p_i выбрасывается, если
   она далеко от обеих соседок, а соседки близки друг к другу:
       ||p_i - p_{i-1}|| > eps  AND  ||p_i - p_{i+1}|| > eps
       AND ||p_{i-1} - p_{i+1}|| <= eps
   Так отсекаются фоновые срабатывания heatmap-детектора (блики, текстуры),
   которые «телепортируют» трек, не задевая реальных полётов: настоящая быстрая
   точка мяча далёка ОТ СОСЕДКИ, но соседки при этом тоже далёки друг от друга
   (условие не выполняется) — такие точки остаются.

   eps по умолчанию авто: max_jump_px * dt_кадров между соседями + multiplier *
   диаметр мяча. Максимальная линейная скорость волейбольного мяча ~40 м/с ≈
   5 px/кадр при 240 fps для типовой съёмки; одиночные пропуски детекции
   интерполируются ДО фильтра, поэтому dt обычно = 1 кадр.

2) УРОВЕНЬ СЕТКИ (отображаем только части траектории выше сетки).
   Верхняя лента сетки в волейболе — 243 см (мужчины; 224 см — женщины).
   Уровень в пикселях вычисляется из масштаба сцены:
       scale_px_per_m = net_top_px / net_top_height_m
   где scale оценивается по геометрии кадра (горизонт площадки → масштаб по
   ширине, игроки → рост 1.90 м по вертикали). Если явный уровень не задан и
   автооценка невозможна или неконсистентна (на видео нет сетки/площадки,
   ракурс неизвестен) — ограничение НЕ применяется, рисуется вся траектория.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BallPoint = tuple[int, float, float]          # (frame, x, y)


# --------------------------------------------------------- ложные детекции
def default_outlier_eps(fps: float, height: int, *, max_jump_px: float,
                        ball_diam_mult: float, ball_diam_px: float) -> float:
    """Авто-eps: максимальное честное перемещение за кадр + запас на диаметр мяча.

    max_jump_px задаётся как «px за кадр при 30 fps»; для более частых видео
    допустимое перемещение за кадр пропорционально меньше (скорость та же).
    Дополнительно ограничиваем сверху долей высоты кадра — защита от абсурдных
    eps на нестандартных разрешениях.
    """
    fps = max(fps, 1.0)
    per_frame = max_jump_px * (30.0 / fps)
    eps = per_frame + ball_diam_mult * max(ball_diam_px, 8.0)
    return min(eps, 0.25 * max(height, 1))


def remove_outliers(points: list[BallPoint], eps: float
                    ) -> tuple[list[BallPoint], list[BallPoint]]:
    """Возвращает (чищенные точки, выброшенные ложные детекции).

    Правило (ровно из ТЗ): p_i — ложная, если
        dist(p_i, p_{i-1}) > eps  &&  dist(p_i, p_{i+1}) > eps
        &&  dist(p_{i-1}, p_{i+1}) <= eps
    Проверка идёт по СОСЕДНИМ ТОЧКАМ трека (не по индексу массива), что корректно
    и при разрывах видимости. Итеративно (пока появляются новые изолированные
    выбросы после удаления предыдущих).
    """
    if len(points) < 3 or eps <= 0:
        return list(points), []
    pts = sorted(points, key=lambda p: p[0])
    removed: list[BallPoint] = []
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        keep: list[BallPoint] = [pts[0]]
        i = 1
        n = len(pts)
        while i < n - 1:
            pm1 = keep[-1]                       # предыдущая ОСТАВЛЕННАЯ точка
            cur = pts[i]
            pn1 = pts[i + 1]
            d_prev = math.hypot(cur[1] - pm1[1], cur[2] - pm1[2])
            d_next = math.hypot(cur[1] - pn1[1], cur[2] - pn1[2])
            d_nbrs = math.hypot(pm1[1] - pn1[1], pm1[2] - pn1[2])
            if d_prev > eps and d_next > eps and d_nbrs <= eps:
                removed.append(cur)
                changed = True
                i += 1
                continue
            keep.append(cur)
            i += 1
        keep.append(pts[-1])                     # последняя точка не может быть «средней»
        pts = keep
    return pts, removed


# ------------------------------------------------------------- уровень сетки
@dataclass
class NetLevel:
    """Уровень ВЕРХНЕЙ ленты сетки в координатах кадра + обоснование оценки."""
    y_px: float                 # строка пикселей: выше сетки ⇔ y < y_px
    source: str                 # "manual" | "auto"
    scale_px_per_m: float       # масштаб сцены, использованный при оценке
    net_height_m: float         # физическая высота сетки (2.43 м по умолчанию)
    notes: str = ""

    def above(self, y: float, margin_px: float = 0.0) -> bool:
        return y <= self.y_px + margin_px


@dataclass
class CourtGeometry:
    """Геометрия площадки, измеренная на кадре (для автооценки уровня сетки)."""
    near_edge_y: float | None = None    # нижняя ближняя граница линии площади
    far_edge_y: float | None = None     # верхняя дальняя граница той же зоны
    width_m: float = 9.0                # ширина зоны между боковыми (9 м)
    length_m: float = 9.0               # глубина половины (9 м)
    player_height_m: float = 1.90
    players_px: list[float] = field(default_factory=list)  # высоты игроков, px


def _scale_from_court(geo: CourtGeometry, width_px: float) -> float | None:
    """Горизонтальный масштаб px/м по линии площади (перспектива ⇒ берём ближнюю)."""
    if geo.near_edge_y is None or width_px <= 0:
        return None
    # Ширина площадки 9 м проецируется примерно на всю ширину кадра у ближней
    # границы; если кадр «режет» площадку — масштаб будет занижен, что даёт
    # завышенный net_top_px (сетка ниже) — консервативноacceptable, т.к. мы всё
    # равно проверяем консистентность с ростом игроков.
    return width_px / geo.width_m


def estimate_net_level(height: int, width: int, geo: CourtGeometry | None, *,
                       net_height_m: float, manual_y_px: float = 0.0,
                       scale_min: float, scale_max: float) -> NetLevel | None:
    """Оценка уровня верхней ленты сетки в px. None — определить невозможно.

    manual_y_px > 0 → явная отметка пользователя (BT_NET_TOP_PX).
    Иначе авто: scale по площадке (горизонталь) сверяется со scale по росту
    игроков (вертикаль); при расхождении >2x или отсутствии данных — None
    («на некоторых видео сетки может не быть или она плохо видна» → лимит не
    применяется, рисуем всю траекторию).
    """
    if manual_y_px and manual_y_px > 0:
        scale = manual_y_px / net_height_m
        return NetLevel(y_px=float(manual_y_px), source="manual",
                        scale_px_per_m=scale, net_height_m=net_height_m,
                        notes="BT_NET_TOP_PX")
    if geo is None:
        return None

    s_horiz = _scale_from_court(geo, width)
    s_vert = None
    if geo.players_px:
        med_h = float(np.median(geo.players_px))
        if med_h > 0:
            s_vert = med_h / geo.player_height_m

    candidates = [s for s in (s_horiz, s_vert) if s is not None]
    if not candidates:
        return None
    if len(candidates) == 2:
        a, b = candidates
        if max(a, b) / max(min(a, b), 1e-6) > 2.0:
            # геометрия неконсистентна (разные плоскости/ракурс) — не рискуем
            return None
        scale = math.sqrt(a * b)              # согласование двух оценок
    else:
        scale = candidates[0]
    if not (scale_min <= scale <= scale_max):
        return None
    y_net = scale * net_height_m
    if not (0.15 * height <= y_net <= 0.95 * height):
        return None                            # абсурдный уровень — считаем, что не нашли
    return NetLevel(y_px=y_net, source="auto", scale_px_per_m=scale,
                    net_height_m=net_height_m,
                    notes=f"s_horiz={s_horiz and round(s_horiz,1)} "
                         f"s_vert={s_vert and round(s_vert,1)}")


def clip_track_above_net(track: list[BallPoint], net: NetLevel | None, *,
                         min_points: int = 2
                         ) -> tuple[list[BallPoint], dict]:
    """Оставляет только точки ВЫШЕ сетки (y <= net.y_px), разрывая трек на куски.

    Точки ниже уровня сетки удаляются (мяч под сеткой/за линией лица нам неин-
    тересен при отображении). Возвращает (чищенный трек, статистику). Если net
    is None — трек возвращается без изменений (лимит неприменим: сетки нет/не
    видна).
    """
    stats = {"applied": net is not None,
             "net_y_px": round(net.y_px, 1) if net else None,
             "source": net.source if net else None,
             "removed_below": 0}
    if net is None:
        return list(track), stats
    kept = [p for p in track if p[2] <= net.y_px]
    stats["removed_below"] = len(track) - len(kept)
    if len(kept) < min_points:
        return [], stats
    return kept, stats
