"""Фильтрация траектории мяча: ложные детекции и уровень сетки.

1) ЛОЖНЫЕ ДЕТЕКЦИИ — СКОРОСТНОЙ ШЛЮЗ (velocity gate).
   Раньше стояло чисто геометрическое правило по явному epsilon:
       p_i — ложная, если ||p_i-p_{i-1}|| > eps И ||p_i-p_{i+1}|| > eps
                           И ||p_{i-1}-p_{i+1}|| <= eps
   На реальном видео оно НЕ РАБОТАЕТ: ложные срабатывания heatmap-детектора
   приходят ЦЕПОЧКАМИ/чередованиями (горячая точка фона детектируется в
   нескольких подряд идущих кадрах, чередуясь с настоящим мячом), поэтому
   условие «соседки рядом» для них никогда не выполняется — выбросы оставались
   в треке при любом eps. А уменьшение eps лишь резало трек на осколки:
   расстояние между соседними ВАЛИДНЫМИ точками на быстром полёте само по
   себе больше eps → фильтр считал настоящий полёт «цепочкой прыжков»,
   trек рвался на микро-куски (< par_min_frames) и API возвращал flights=[].

   Новая логика (remove_outliers_velocity): точка отвергается, если она
   отклоняется от ЛОКАЛЬНОЙ ЭКСТРАПОЛЯЦИИ ТРАЕКТОРИИ сильнее, чем допускает
   физика мяча:
       predicted = p_{i-1} + v_hat * dt            (v_hat — оценка скорости
       residual  = ||p_i - predicted||             по предыдущим ~W точкам)
       порог:    max_gate = gate_mult * max(||v_hat|| * dt, v_min_px)
   Статичная горячая точка (v_hat≈0) даёт residual >> v_min → удаляется;
   быстрый честный полёт имеет большую v_hat → порог пропорционален скорости,
   точки полёта НЕ удаляются. После удаления точки оценка v_hat пересчиты-
   вается по чищенным соседям, поэтому цепочки выбросов вычищаются итеративно.

2) РАЗРЫВ ТРЕКА (split_track_by_jumps) — теперь по физической максимальной
   скорости мяча, а не по eps: разрыв ставится там, где шаг между соседними
   ОСТАВЛЕННЫМИ точками превышает speed_max_px_f px/кадр (даже с учётом
   интерполяции через удалённые точки). Это границы реальных розыгрышей;
   микро-разрывы на каждом шаге быстрого полёта больше не возникают.

3) УРОВЕНЬ СЕТКИ (отображаем только части траектории выше сетки).
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
def _local_velocity(keep: list[BallPoint], window: int) -> tuple[float, float] | None:
    """Оценка скорости мяча (px/кадр) по последним `window` оставленным точкам.

    LSQ-наклон берётся по РЕАЛЬНЫМ номерам кадров (не по позиции в массиве):
    после удаления выбросов соседние элементы могут отстоять на много кадров,
    и «наклон по индексу» занинял скорость в разы — шлюз съедал весь честный
    полёт сразу после дыры в наблюдениях.
    """
    tail = keep[-window:] if window > 0 else keep
    if len(tail) < 2:
        return None
    t = np.array([p[0] for p in tail], float)
    x = np.array([p[1] for p in tail], float)
    y = np.array([p[2] for p in tail], float)
    dt = t[-1] - t[0]
    if dt <= 0:
        return None
    A = np.stack([np.ones_like(t), t], axis=1)
    try:
        cx, *_ = np.linalg.lstsq(A, x, rcond=None)
        cy, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(cx[1]), float(cy[1])


def remove_outliers_velocity(points: list[BallPoint], *, gate_mult: float,
                             v_min_px: float, speed_max_px_f: float,
                             window: int = 6, min_streak: int = 2
                             ) -> tuple[list[BallPoint], list[BallPoint]]:
    """Скоростной шлюз ложных детекций. Возвращает (чищенные, выброшенные).

    Для каждой очередной точки p_i строим прогноз из локальной экстраполяции
    траектории:

        v_hat     — LSQ-скорость (px/кадр) по последним `window` ОСТАВЛЕННЫМ
                    наблюдениям (после удаления предыдущих выбросов — поэтому
                    цепочки/чередования горячих точек вычищаются);
        predicted = p_{last} + v_hat * dt,   dt = f_i - f_last;
        residual  = ||p_i - predicted||.

    Точка СЧИТАЕТСЯ ЛОЖНОЙ, если одновременно
        a) residual > gate_mult * max(||v_hat||*dt, v_min_px) — далеко от
           продолжения траектории; И
        b) dist(p_i, p_{last}) > gate_mult * max(speed_max_px_f,
           ||v_hat||*dt + v_min_px) * dt — она физически недосяжна от
           ближайшей оставленной точки даже с учётом резкого ускорения.
    Условие (b) спасает быстрый полёт сразу ПОСЛЕ большой дыры в наблюдениях
    (первый кадр розыгрыша после длинного перерыва): там v_hat ≈ 0 и один
    лишь residual съедал бы начало каждого полёта.

    НАДЁЖНОСТЬ (главный урок реального видео): одиночная «несогласованная»
    точка обычно — честный шум или смена направления мяча после удара,
    поэтому удаляется только ЦЕПОЧКА из >= min_streak подряд несогласованных
    точек, причём ВСЕГДА сохраняется самая правдоподобная из них — точка с
    минимальным residual (как правило это настоящий мяч; ложные hotspot'ы
    статичны и дают больший разброс).

    ИСКЛЮЧЕНИЕ — классический одиночный «телепорт туда-обратно»: если сама
    точка недосяжна (b), а предыдущая оставленная и следующая точки трека
    рядом друг с другом (мяч никуда не делся) — это фоновое срабатывание,
    оно удаляется даже без цепочки (условие старого eps-правила:
    ||p_{i-1}-p_{i+1}|| <= gate_mult * v_min_px).

    Первые две точки всегда остаются (экстраполировать не от чего).
    """
    if len(points) < 3:
        return sorted(points, key=lambda p: p[0]), []
    pts = sorted(points, key=lambda p: p[0])
    keep: list[BallPoint] = [pts[0], pts[1]]
    removed: list[BallPoint] = []

    def is_fake(cur: BallPoint) -> tuple[bool, float]:
        last = keep[-1]
        dt = max(float(cur[0] - last[0]), 1e-9)
        v = _local_velocity(keep, window)
        d_last = math.hypot(cur[1] - last[1], cur[2] - last[2])
        if v is None:
            bad = speed_max_px_f > 0 and d_last > gate_mult * speed_max_px_f * dt
            return bad, d_last
        vx, vy = v
        speed_now = math.hypot(vx, vy)
        pred_x = last[1] + vx * dt
        pred_y = last[2] + vy * dt
        residual = math.hypot(cur[1] - pred_x, cur[2] - pred_y)
        gate = gate_mult * max(speed_now * dt, v_min_px)
        # недосягаемость: даже при резком ускорении мяч за dt не может
        # пройти больше (v_текущ + a_max*dt)*dt; иначе первый кадр ПОСЛЕ
        # длинной дыры наблюдений (v_hat≈0 от старых точек) ошибочно
        # считался бы ложью и съедал начало каждого полёта
        reach = gate_mult * max(speed_max_px_f, speed_now * dt + v_min_px) * dt
        return (residual > gate and d_last > reach), residual

    i = 2
    n = len(pts)
    while i < n:
        bad0, r0 = is_fake(pts[i])
        if not bad0:
            keep.append(pts[i])
            i += 1
            continue
        # собрать максимальную цепочку подряд несогласованных точек, НЕ
        # изменяя keep (прогноз для всех — от текущего хвоста чищенного трека)
        chain: list[tuple[BallPoint, float]] = [(pts[i], r0)]
        j = i + 1
        while j < n:
            bad, r = is_fake(pts[j])
            if not bad:
                break
            chain.append((pts[j], r))
            j += 1
        if len(chain) < min_streak:
            # Одиночная несогласованность. Обычно оставляем её (шум/смена
            # направления). НО если точка физически недосяжна, а следующая
            # точка трека ВЕРНУЛАСЬ к траектории (рядом с предыдущей
            # оставленной) — это телепорт «туда-обратно» = hotspot-ложь.
            nxt = pts[j] if j < n else None
            single_unreachable = False
            if nxt is not None:
                # «телепорт туда-обратно»: сама точка недосяжна, а следующая
                # согласована со шлюзом И вернулась к предыдущей оставленной
                # точке (мяч никуда не делся) → это фоновая hotspot-ложь.
                bad_next, _ = is_fake(nxt)
                nbr_gap = math.hypot(nxt[1] - keep[-1][1], nxt[2] - keep[-1][2])
                dt_n = max(float(nxt[0] - keep[-1][0]), 1.0)
                # допуск «рядом» с учётом честного движения за dt_n; при
                # реальном быстром перелёте nxt сама недосяжна (ok_next=True)
                # и правило не срабатывает — полёт сохраняется
                nbr_tol = gate_mult * v_min_px * dt_n
                if (not bad_next) and nbr_gap <= nbr_tol:
                    single_unreachable = True
            if single_unreachable:
                removed.append(pts[i])
                i += 1
                continue
            # одиночная несогласованность — оставляем (не теряем реальные
            # шумы/смену направления честного полёта)
            keep.append(pts[i])
            i += 1
            continue
        # из цепочки сохраняем САМУЮ правдоподобную точку (min residual) —
        # но только если она согласуется с СОСЕДНИМИ наблюдениями: при
        # чередовании «мяч/hotspot» самая близкая к экстраполяции точка и
        # есть настоящий мяч. Если же вся цепочка стоит в одной точке фона
        # (последующие точки рядом с первой, v≈0), правдоподобная — первая,
        # а остальные — ложь; удаляется вся цепочка целиком
        best_k = min(range(len(chain)), key=lambda k: chain[k][1])
        best = chain[best_k][0]
        nbr_ok = True
        if j < n:                                   # следующая СОГЛАСОВАНА
            nxt = pts[j]
            d_nb = math.hypot(nxt[1] - best[1], nxt[2] - best[2])
            dt_nb = max(float(nxt[0] - best[0]), 1.0)
            nbr_ok = d_nb <= gate_mult * max(v_min_px * dt_nb, speed_max_px_f)
        elif len(keep) >= 2:                        # хвост трека: согласованность
            prev = keep[-1]                         # с предыдущей оставленной
            d_pb = math.hypot(best[1] - prev[1], best[2] - prev[2])
            dt_pb = max(float(best[0] - prev[0]), 1.0)
            nbr_ok = d_pb <= gate_mult * max(v_min_px * dt_pb, speed_max_px_f)
        if not nbr_ok:
            removed.extend(p for p, _ in chain)     # статичный hotspot-фон
            i = j
            continue
        for k, (p, _) in enumerate(chain):
            if k == best_k:
                keep.append(p)
            else:
                removed.append(p)
        i = j
    return keep, removed


def remove_static_hotspots(points: list[BallPoint], *, cell_px: float = 50.0,
                           radius_px: float = 45.0, min_streak: int = 2,
                           speed_max_px_f: float = 120.0,
                           rally_gap_frames: int = 0
                           ) -> tuple[list[BallPoint], list[BallPoint]]:
    """Удаление СТАТИЧНЫХ hotspot-цепочек — ложных срабатываний heatmap-детектора.

    Главный урок реального видео (VID_20260925_163505.mp4): ложные детекции
    приходят не одиночными телепортами, а ЦЕПОЧКАМИ в одной точке фона
    (рекламный щит, мяч на полке, пятно), которые ЧЕРЕДУЮТСЯ с настоящим
    мячом. Старый eps-фильтр («обе соседки рядом, точка — телепорт») таких
    цепочек не видел; скоростной шлюз по экстраполяции тоже пасовал — при
    чередовании «мяч/hotspot» прогноз строился от предыдущей hotspot-точки и
    съедал честный полёт (регресс kept=8 из 160).

    Новый критерий — физическая связность цепочки наблюдений:
      * наблюдения склеиваются в ЦЕПОЧКИ: p_j продолжает цепочку, если
        dist(p_{j-1}, p_j) <= max(chain_speed_limit*dt, static_radius_px)
        (статичная группа удерживается внутри своего радиуса; движущийся мяч
        продолжается своей согласованной скоростью);
      * цепочка — ЛОЖЬ, если её НЕВОЗМОЖНО физически связать ни с какой
        соседней цепочкой: минимальная требуемая скорость стыковки
        (слева/справа по времени) > speed_max_px_f. Настоящий мяч всегда
        «вписывается» в физику хотя бы с одним соседом; изолированная
        статичная точка фона посреди кадра — нет;
      * ОДИНОЧНЫЕ несогласованные точки НЕ удаляются без веской причины —
        правило «телепорт туда-обратно»: сама точка недосяжна, а следующая
        вернулась к предыдущей оставленной (мяч никуда не делся).

    Возвращает (чищенные, выброшенные).
    """
    pts = sorted(points, key=lambda p: p[0])
    n = len(pts)
    if n < 3:
        return pts, []

    # --- 1. склейка наблюдений в когерентные цепочки ------------------------
    # Стык ЦЕЛОСТНЫХ эпизодов (граница розыгрышей) отличается от «дыр»
    # пропусков детекции: если между соседними наблюдениями мяч НЕ
    # детектировался дольше rally_gap_frames, это новый эпизод — здесь трек
    # рвётся гарантированно (иначе интерполяция через дыру рисует фантомную
    # линию между двумя разными полётами).
    chains: list[list[int]] = [[0]]                  # индексы pts
    for j in range(1, n):
        c = chains[-1]
        last = pts[c[-1]]
        cur = pts[j]
        dt = max(float(cur[0] - last[0]), 1.0)
        if rally_gap_frames and float(cur[0] - last[0]) > rally_gap_frames:
            chains.append([j])
            continue
        d = math.hypot(cur[1] - last[1], cur[2] - last[2])
        if len(c) >= 2:
            a, b = pts[c[-2]], pts[c[-1]]
            vprev = math.hypot(b[1] - a[1], b[2] - a[2]) / max(float(b[0] - a[0]), 1.0)
        else:
            vprev = 0.0
        # статичная группа держится в своём радиусе; движущийся мяч — за
        # согласованной скоростью (предыдущая + физический запас на ускорение)
        limit = max(min(vprev, speed_max_px_f) * dt + radius_px,
                    radius_px, speed_max_px_f * dt)
        if d <= limit:
            c.append(j)
        else:
            chains.append([j])

    def span(ci: int) -> tuple[float, float, float, float]:
        c = chains[ci]
        xs = [pts[k][1] for k in c]; ys = [pts[k][2] for k in c]
        return (min(xs), max(xs), min(ys), max(ys))

    def dock_speed(ci: int, cj: int) -> float | None:
        """Мин. средняя скорость (px/кадр) стыковки цепочки ci с cj по времени.

        Дистанция — между «габаритами» (bounding boxes) цепочек: если боксы
        пересекаются, достаточно скорости ~0 (мяч мог остаться на месте)."""
        fi = [pts[k][0] for k in chains[ci]]
        fj = [pts[k][0] for k in chains[cj]]
        x0i, x1i, y0i, y1i = span(ci)
        x0j, x1j, y0j, y1j = span(cj)
        ex = max(x0i - x1j, x0j - x1i, 0.0)
        ey = max(y0i - y1j, y0j - y1i, 0.0)
        dist = math.hypot(ex, ey)
        best = None
        for fa in (fi[-1], fi[0]):
            for fb in (fj[0], fj[-1]):
                dt = abs(float(fb - fa))
                if dt <= 0:
                    continue
                v = dist / dt
                best = v if best is None or v < best else best
        return best

    # --- 2. помечаем ложью несвязуемые цепочки -------------------------------
    # Кандидат на удаление — СТАТИЧНАЯ цепочка (все точки в пределах
    # radius_px): настоящий мяч между контактами НЕ стоит на месте несколько
    # кадров подряд, а heatmap-hotspot фона — стоит. Далее достаточно ЛЮБОГО
    # физически достижимого соседа по времени (мяч мог прилететь/улететь),
    # иначе цепочка — изолированный артефакт. Одиночные точки (len <
    # min_streak) не судим вовсе; для очень коротких (len == min_streak == 2)
    # требуется «телепорт туда-обратно»: сосед слева и сосед справа рядом
    # ДРУГ С ДРУГОМ (мяч никуда не делся), но оба далеко от нас.
    fake = [False] * len(chains)
    for ci in range(len(chains)):
        c = chains[ci]
        if len(c) < min_streak:
            continue                                # одиночные — не трогаем
        x0, x1, y0, y1 = span(ci)
        static = math.hypot(x1 - x0, y1 - y0) <= radius_px
        if not static:
            continue                                # движется — это мяч
        f_first = pts[c[0]][0]; f_last = pts[c[-1]][0]
        left = right = None
        near_any = False
        for cj in range(len(chains)):
            if cj == ci:
                continue
            fj = chains[cj]
            gj_first = pts[fj[0]][0]; gj_last = pts[fj[-1]][0]
            v = dock_speed(ci, cj)
            if v is None:
                continue
            if v <= speed_max_px_f:
                near_any = True
            if gj_last <= f_first:                  # cj целиком левее по времени
                left = v if left is None or v < left else left
            elif gj_first >= f_last:                # cj целиком правее
                right = v if right is None or v < right else right
        anchors = [v for v in (left, right) if v is not None]
        if not anchors:
            isolated = False                        # вне кадра — не судим
        else:
            isolated = not near_any
        # сверхстрогое правило для самых коротких статичных цепочек: соседи
        # рядом друг с другом (мяч продолжал полёт рядом), а мы — всплеск
        if isolated and len(c) <= 2:
            nb = [(dock_speed(ci, cj), cj) for cj in range(len(chains))
                  if cj != ci]
            nb = [(v, cj) for v, cj in nb if v is not None]
            pair_ok = False
            for (va, ca), (vb, cb) in zip(nb, nb[1:]):
                da, db = span(ca), span(cb)
                gap = math.hypot(max(da[0] - db[1], db[0] - da[1], 0.0),
                                 max(da[2] - db[3], db[2] - da[3], 0.0))
                fa_ = pts[chains[ca][-1]][0]; fb_ = pts[chains[cb][0]][0]
                dt_ = max(abs(float(fb_ - fa_)), 1.0)
                if gap / dt_ <= speed_max_px_f and va > speed_max_px_f \
                        and vb > speed_max_px_f:
                    pair_ok = True
            isolated = pair_ok
        if isolated:
            fake[ci] = True
    keep = [pts[k] for ci, c in enumerate(chains) if not fake[ci] for k in c]
    removed = [pts[k] for ci, c in enumerate(chains) if fake[ci] for k in c]
    return sorted(keep, key=lambda p: p[0]), sorted(removed, key=lambda p: p[0])


def split_track_by_jumps(points: list[BallPoint], speed_max_px_f: float,
                         rally_gap_frames: int = 0, jump_tolerance: float = 0.25
                         ) -> tuple[list[BallPoint], int]:
    """Разрезает трек на розыгрыши по физической границе скорости мяча.

    ВАЖНО (главный баг прошлой версии): разрыв ищем ТОЛЬКО по РЕАЛЬНЫМ
    наблюдениям — парам соседних точек с dt == 1 кадр. Линейно
    интерполированные точки (после _interp_track) образуют «мосты», у которых
    фиктивные концевые точки всегда имеют dt == 1 и мгновенную скорость,
    равную средней на всём пролёте моста. На реальном видео это убивало
    трек: длинный честный перелёт (например, frames 12→25 через ~1400 px)
    давал на конце моста ~95 px/кадр, следующий за ним короткий шаг
    (25→26, ~7 px) формально превышал порог 90*1.2 — трек рвался посреди
    одного полёта на десятки микро-осколков (< par_min_frames), и API
    возвращал flights=[]. Поэтому теперь:
      * пары с dt != 1 (вершина «моста») пропускаются — там мгновенная
        скорость неопределима;
      * разрыв ставится, если dist/dt > speed_max_px_f * jump_tolerance
        (допуск на шум/ускорение) для REAL-пары;
      * ЛИБО разрыв в кадрах > rally_gap_frames (мяч не детектировался
        дольше окна розыгрыша — граница эпизода, даже если точки рядом).
    Обычный пас/атака (шаг 20–80 px/кадр) остаётся цельным; рвутся только
    настоящие склейки разных эпизодов (teleport через весь кадр).
    Возвращает (ПЕРВЫЙ кусок до разрыва, 1) либо (весь трек, 0).
    """
    if len(points) < 2 or speed_max_px_f <= 0:
        return sorted(points, key=lambda p: p[0]), 0
    pts = sorted(points, key=lambda p: p[0])
    tol = 1.0 + max(jump_tolerance, 0.0)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        dt = float(b[0] - a[0])
        # кадр-разрыв эпизода — граница розыгрыша даже при малом смещении
        if rally_gap_frames and dt > rally_gap_frames:
            return pts[:i + 1], 1
        # мгновенную скорость проверяем ТОЛЬКО на реальных соседних кадрах
        # (dt == 1); вершины интерполяционных «мостов» (dt != 1) пропускаем,
        # иначе длинный перелёт рвётся посреди честного полёта (см. docstring)
        if dt == 1.0:
            if math.hypot(b[1] - a[1], b[2] - a[2]) > speed_max_px_f * tol:
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
