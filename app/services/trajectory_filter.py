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
def default_outlier_eps(fps: float, height: int, *, max_speed_px_s: float,
                        ball_diam_mult: float, ball_diam_px: float) -> float:
    """Авто-eps: максимальное честное перемещение между соседними точками трека
    (v_max * dt_med) + запас на диаметр мяча.

    max_speed_px_s — «разумный максимум» скорости мяча в px/сек (реальный
    волейбольный мяч на типовой трансляции летит не быстрее ~1200 px/s);
    пересчитывается в px/кадр через медианный шаг кадров трека (fps), поэтому
    корректно работает и на 25 fps, и на 240 fps. Ограничиваем снизу диаметром
    мяча (иначе шум детекции на медленных участках сам станет «выбросом») и
    сверху долей высоты кадра — защита от абсурдных eps.
    """
    fps = max(fps, 1.0)
    diam = max(ball_diam_px, 8.0)
    per_frame = max_speed_px_s / fps
    eps = max(per_frame, diam) + ball_diam_mult * diam
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
        # явная отметка пользователя НЕ проверяется scale_min/max (масштаб
        # может быть любым из-за ракурса) — только физический смысл уровня
        return NetLevel(y_px=float(manual_y_px), source="manual",
                        scale_px_per_m=manual_y_px / net_height_m,
                        net_height_m=net_height_m, notes="BT_NET_TOP_PX")
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


# ------------------------------------------------ автооценка геометрии кадра
def estimate_ground_y(ball_points: list[BallPoint]) -> float | None:
    """Уровень «земли» (пол/руки игроков) по нижним огибающим точек мяча.

    Мяч большую часть розыгрыша находится ВЫШЕ уровня игры руками (~1.5–2 м),
    поэтому медиана y всех точек трека — устойчивая оценка горизонта игры;
    берём перцентиль 0.85 (нижняя огибающая) как уровень земли y_g в px.
    None — если данных мало (<20 точек).
    """
    if len(ball_points) < 20:
        return None
    ys = np.array([p[2] for p in ball_points], float)
    return float(np.percentile(ys, 85.0))


def auto_court_geometry(ball_points: list[BallPoint], height: int, *,
                        ground_frac_lo: float, ground_frac_hi: float,
                        player_height_frac: float
                        ) -> CourtGeometry | None:
    """CourtGeometry из одного только трека мяча (без разметки/детекции людей).

    Пайплайн не детектирует линию площади и игроков, поэтому near_edge_y и
    «рост игроков в px» напрямую взять неоткуда. Масштаб оцениваем по ДВУМ
    якорям плоскости сетки:
      * верхний якорь y_play = H * player_height_frac — типичная ВЕРХНЯЯ точка
        полётов (пас/атака на уровне вытянутых рук ~2.0–2.4 м); для волейболь-
        ной трансляции с потолком 5–7 м это 0.30–0.45H;
      * нижний якорь y_g = перцентиль 0.85 y-координат трека — уровень игры у
        земли (~0.6–1.2 м, приём/передача снизу).
    Между ними ≈ player_height_m метров (разница высот рук над головой и
    принятия мяча у пояса) ⇒ scale = (y_g − y_play)/player_height_m, а уровень
    сетки 2.43 м лежит НАД верхним якорем: net_y = y_play − scale*(2.43 −
    player_height_m + 0.45). При неконсистентности якорей (земля выше уровня
    игры или слишком далеко) вернём None → estimate_net_level откажет и лимит
    не применится («сетки нет / плохо видна»).
    """
    y_g = estimate_ground_y(ball_points)
    if y_g is None:
        return None
    y_play = height * ground_frac_lo          # верхняя огибающая полётов
    # земля должна быть НИЖЕ уровня игры и в разумных пределах кадра
    if not (y_play < y_g < ground_frac_hi * height):
        return None
    span_px = max(y_g - y_play, 1.0)
    scale = span_px / player_height_frac      # px на метр разницы высот якорей
    return CourtGeometry(near_edge_y=None, far_edge_y=y_g,
                         players_px=[scale * player_height_m_default])


player_height_m_default = 1.90   # рост игрока, м (для перевода scale в px-«рост»)
