"""Фильтрация траектории мяча: ложные детекции и уровень сетки.

1) ЛОЖНЫЕ ДЕТЕКЦИИ (изолированные выбросы). Явный небольшой epsilon (px),
   правило ровно из ТЗ:
       p_i — ложная, если  ||p_i - p_{i-1}|| > eps  И  ||p_i - p_{i+1}|| > eps
                            И ||p_{i-1} - p_{i+1}|| <= eps
   т.е. соседки стоят «рядом» (мяч никуда не делся), а p_i телепортировалась —
   это фоновое срабатывание heatmap-детектора (блик, текстура). Точка просто
   выбрасывается.

   ВАЖНОЕ отличие от «быстрого полёта»: если p_i далеко от p_{i-1}, а соседки
   p_{i-1} и p_{i+1} тоже НЕ рядом друг с другом — значит мяч действительно
   быстро перелетел (атака/подача). Такую точку НЕ удаляем; вместо этого трек
   режется в этом месте на части (split_track_by_jumps): каждый кусок — это
   отдельный розыгрыш, и дальше по нему пасы ищутся независимо.

2) УРОВЕНЬ СЕТКИ (отображаем только части траектории выше сетки).
   Верхняя лента сетки в волейболе — 243 см (мужчины; 224 см — женщины).
   Автоопределение по треку мяча без разметки кадра (старая оценка была
   нестабильна: на одном видео линия выше сетки, на другом «на земле», на
   третьем ничего не задетектилось), поэтому теперь просто:
     * явный уровень BT_NET_TOP_PX > 0 → используется он;
     * иначе берётся ФИКСИРОВАННЫЙ типовой уровень трансляции
       BT_NET_AUTO_DEFAULT_FRAC * H (для стандартного кадра волейбольной
       трансляции верхняя лента ≈ 65% высоты);
     * если включён фильтр согласованности BT_NET_AUTO_CHECK=1, то уровень
       принимается только когда в треке есть точки как ВЫШЕ него (мяч летает
       над сеткой — уровень имеет смысл), так и НИЖЕ него (мяч бывает у пола/
       рук — трек реально пересекает зону игры). Если данных мало или трек
       целиком по одну сторону — считаем, что сетку «не видно», лимит НЕ
       применяется, рисуется вся траектория.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BallPoint = tuple[int, float, float]          # (frame, x, y)


# --------------------------------------------------------- ложные детекции
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


def split_track_by_jumps(points: list[BallPoint], eps: float
                         ) -> tuple[list[BallPoint], int]:
    """Разрезает трек там, где мяч ЧЕСТНО быстро перелетел (атака/подача).

    После remove_outliers любая пара соседних точек трека с расстоянием > eps
    — это уже НЕ ложная детекция (одиночные «телепортации» удалены), а реальный
    быстрый перелёт. По ТЗ такой момент означает начало нового розыгрыша,
    поэтому трек разрывается ровно здесь. Возвращает (ПЕРВЫЙ кусок до прыжка,
    1) либо (весь трек, 0) — вызывающий цикл забирает хвост rest[len(head):]
    и повторяет, пока разрывы не кончатся.
    """
    if len(points) < 2 or eps <= 0:
        return list(points), 0
    pts = sorted(points, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if math.hypot(b[1] - a[1], b[2] - a[2]) > eps:
            return pts[:i + 1], 1
    return pts, 0


# ------------------------------------------------------------- уровень сетки
@dataclass
class NetLevel:
    """Уровень ВЕРХНЕЙ ленты сетки в координатах кадра + обоснование оценки."""
    y_px: float                 # строка пикселей: выше сетки ⇔ y < y_px
    source: str                 # "manual" | "auto"
    scale_px_per_m: float       # масштаб сцены (для manual: y_px / net_height_m)
    net_height_m: float         # физическая высота сетки (2.43 м по умолчанию)
    notes: str = ""

    def above(self, y: float, margin_px: float = 0.0) -> bool:
        return y <= self.y_px + margin_px


def estimate_net_level(height: int, ball_points: list[BallPoint], *,
                       net_height_m: float, manual_y_px: float = 0.0,
                       default_frac: float = 0.65,
                       consistency_check: bool = True) -> NetLevel | None:
    """Уровень верхней ленты сетки в px. None — определить невозможно.

    manual_y_px > 0 → явная отметка пользователя (BT_NET_TOP_PX).
    Иначе фиксированный типовой уровень трансляции default_frac*H (сетка
    243 см ≈ 65% высоты кадра для стандартной камеры). Фильтр согласованности:
    уровень принимается, только если трек мяча его «осмысляет» — есть точки
    и выше, и ниже сетки (мяч пересекает зону игры). Мало точек / трек целиком
    по одну сторону → сетку считать невидимой → None (лимит не применяется).
    """
    if manual_y_px and manual_y_px > 0:
        y = float(min(manual_y_px, height - 1))
        return NetLevel(y_px=y, source="manual",
                        scale_px_per_m=y / max(net_height_m, 1e-6),
                        net_height_m=net_height_m, notes="BT_NET_TOP_PX")

    y_net = float(np.clip(default_frac, 0.05, 0.98)) * height
    if not consistency_check:
        return NetLevel(y_px=y_net, source="auto",
                        scale_px_per_m=y_net / max(net_height_m, 1e-6),
                        net_height_m=net_height_m,
                        notes=f"default {default_frac:.2f}*H, check off")
    if len(ball_points) < 20:
        return None                              # данных нет — не гадаем
    ys = np.array([p[2] for p in ball_points], float)
    frac_above = float((ys < y_net).mean())
    if not (0.02 < frac_above < 0.98):
        return None                              # весь трек по одну сторону —
    return NetLevel(y_px=y_net, source="auto",   # уровень нам не виден
                    scale_px_per_m=y_net / max(net_height_m, 1e-6),
                    net_height_m=net_height_m,
                    notes=f"default {default_frac:.2f}*H, above={frac_above:.0%}")


def clip_track_above_net(track: list[BallPoint], net: NetLevel | None, *,
                         min_points: int = 2
                         ) -> tuple[list[BallPoint], dict]:
    """Оставляет только точки ВЫШЕ сетки (y <= net.y_px).

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
