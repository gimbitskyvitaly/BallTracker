"""Отрисовка траекторий пасов поверх исходного видео (аналог оверлея BallTime™).

Результат — видео, покадрово идентичное загруженному, но поверх него:
  * во время параболического полёта (паса) рисуется шлейф траектории за мячом,
    полная дуга пунктиром, маркеры RELEASE / CATCH;
  * направление паса кодируется цветом и подписью: PASS LEFT (мяч летит влево
    к сетке) / PASS RIGHT (вправо к сетке); участки «от сетки» пасами не
    считаются и не рисуются;
  * при наличии фита физмодели — тонкая белая линия смоделированной траектории;
  * HUD: номер паса, направление, ToF, apex, дальность, v0;
  * после окончания полёта вся дуга паса остаётся на кадре ещё tail_hold_frames
    кадров («след сыгранного розыгрыша»), затем исчезает.

Всё рисуется в пиксельных координатах трека, поэтому привязка к кадру точна
независимо от калибровки масштаба.
"""

from __future__ import annotations

import os
import traceback
from collections import deque

import cv2
import numpy as np

from app.config import settings

# палитра: чётные пасы — тёплые, нечётные — холодные; направление задаёт
# подпись и форму маркера (см. draw_flight)
_COLORS = [(0, 200, 255), (0, 255, 140), (255, 160, 60), (200, 80, 255),
           (80, 230, 230), (255, 255, 255)]


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _point_at_time(pts: list[tuple[int, float, float]], frame: float):
    """Интерполированная позиция мяча на кадре frame (float) по точкам трека."""
    if not pts or frame < pts[0][0] or frame > pts[-1][0]:
        return None
    for i in range(len(pts) - 1):
        f0, x0, y0 = pts[i]
        f1, x1, y1 = pts[i + 1]
        if f0 <= frame <= f1:
            span = max(f1 - f0, 1e-9)
            t = (frame - f0) / span
            return _lerp((x0, y0), (x1, y1), t)
    return (pts[-1][1], pts[-1][2])


class FlightOverlay:
    """Один пас (параболический участок к сетке) с данными для рендера."""

    def __init__(self, index: int, release_frame: int, catch_frame: int,
                 pts_px: list[tuple[int, float, float]],
                 metrics: dict | None = None,
                 sim_pts_px: list[tuple[int, float, float]] | None = None,
                 direction: str = "left"):
        self.index = index
        self.release_frame = int(release_frame)
        self.catch_frame = int(catch_frame)
        self.pts = pts_px                       # трек мяча в px: [(frame,x,y)]
        self.sim_pts = sim_pts_px or []         # симуляция физмодели в px
        self.metrics = metrics or {}
        self.direction = direction              # "left" | "right"
        self.color = _COLORS[index % len(_COLORS)]

    @property
    def label(self) -> str:
        return f"PASS {'LEFT' if self.direction == 'left' else 'RIGHT'} #{self.index + 1}"

    def active(self, frame_id: int, hold: int) -> bool:
        """Полёт активен или находится в «хвосте удержания» после приёмки."""
        return self.release_frame <= frame_id <= self.catch_frame + hold

    def contains(self, frame_id: int) -> bool:
        return self.release_frame <= frame_id <= self.catch_frame


def draw_flight(frame: np.ndarray, fl: FlightOverlay, ball_radius: int,
                trail_length: int, fid: int, hold_left: int = 0) -> None:
    """Рисуем на frame всё, что относится к пасу fl на кадре fid.

    hold_left > 0 — полёт уже завершён, осталось «эхо»: без текущего мяча
    и хвоста, только полная дуга и маркеры, постепенно бледнеющие.
    """
    h_img, w_img = frame.shape[:2]
    fading = hold_left > 0
    cur = None if fading else _point_at_time(fl.pts, float(fid))

    # --- полная дуга паса ------------------------------------------------------
    done = [p for p in fl.pts if p[0] <= fid]
    if len(done) >= 2:
        pts_np = np.array([(int(x), int(y)) for _, x, y in done], dtype=np.int32)
        mask = np.zeros((h_img, w_img), np.uint8)
        cv2.polylines(mask, [pts_np.reshape(-1, 1, 2)], False, 255, 2)
        region = mask.astype(bool)
        col_arr = np.array(fl.color, np.float64)
        alpha = 0.25 if fading else 0.4
        f32 = frame.astype(np.float64)
        blended = f32 * (1 - alpha) + col_arr * alpha
        frame[region] = blended[region].astype(np.uint8)

    # --- линия физмодели (если есть) -------------------------------------------
    if len(fl.sim_pts) > 1 and not fading:
        sim_done = [p for p in fl.sim_pts if p[0] <= fid]
        cur_sim = _point_at_time(fl.sim_pts, float(fid))
        if cur_sim is not None:
            sim_done = sim_done + [(fid, cur_sim[0], cur_sim[1])]
        for i in range(len(sim_done) - 1):
            p0, p1 = sim_done[i], sim_done[i + 1]
            cv2.line(frame, (int(p0[1]), int(p0[2])), (int(p1[1]), int(p1[2])),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # --- маркеры RELEASE / CATCH -------------------------------------------------
    rx, ry = fl.pts[0][1], fl.pts[0][2]
    kx, ky = fl.pts[-1][1], fl.pts[-1][2]
    caught = fid >= fl.catch_frame
    mk = ((rx, ry), "RELEASE"), ((kx, ky), "CATCH")
    for (px, py), label in mk:
        if label == "CATCH" and not caught:
            continue
        cv2.drawMarker(frame, (int(px), int(py)), fl.color, cv2.MARKER_CROSS,
                       16, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (int(px) + 10, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fl.color, 1, cv2.LINE_AA)

    # --- стрелка направления к сетке ---------------------------------------------
    mx, my = (rx + kx) / 2, (ry + ky) / 2
    dx = 1 if fl.direction == "right" else -1
    arr_len = 26
    cv2.arrowedLine(frame, (int(mx) - dx * arr_len // 2, int(my) - 18),
                    (int(mx) + dx * arr_len // 2, int(my) - 18),
                    fl.color, 2, cv2.LINE_AA, tipLength=0.4)

    if not fading and cur is not None:
        cx, cy = cur
        # --- затухающий хвост ------------------------------------------------------
        past = [p for p in fl.pts if p[0] <= fid][-trail_length:]
        if not past or past[-1][0] != fid:
            past = past + [(fid, cx, cy)]
        n = len(past)
        for i in range(n - 1):
            f0, x0, y0 = past[i]
            f1, x1, y1 = past[i + 1]
            t = (i + 1) / max(n - 1, 1)               # 0..1, к мячу ярче
            col = tuple(int(c * t) for c in fl.color)
            thick = max(1, int(round(1 + 3 * t)))
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), col, thick, cv2.LINE_AA)
        for i in range(0, n, max(1, n // 12)):
            _, x, y = past[i]
            cv2.circle(frame, (int(x), int(y)), 2, fl.color, -1, cv2.LINE_AA)
        # --- текущее положение мяча -----------------------------------------------
        cv2.circle(frame, (int(cx), int(cy)), ball_radius + 3, fl.color, 2, cv2.LINE_AA)

    # --- HUD ------------------------------------------------------------------------
    lines = [fl.label]
    m = fl.metrics
    if m.get("time_of_flight_s") is not None:
        lines.append(f"TOF {m['time_of_flight_s']:.2f}s")
    if m.get("apex_height_m") is not None:
        lines.append(f"apex {m['apex_height_m']:.2f}m")
    if m.get("distance_m") is not None:
        lines.append(f"dist {m['distance_m']:.1f}m")
    if m.get("initial_speed_mps") is not None:
        lines.append(f"v0 {m['initial_speed_mps']:.1f}m/s")
    x0, y0 = 10, 10
    bw = min(190, max(40, w_img - 20))
    bh = min(24 * len(lines) + 12, max(20, h_img - 20))
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
    roi = frame[y0:y0 + bh, x0:x0 + bw]
    hud = roi.copy()
    cv2.addWeighted(hud, 0.55, frame[y0:y0 + bh, x0:x0 + bw], 0.45, 0,
                    frame[y0:y0 + bh, x0:x0 + bw])
    for li, text in enumerate(lines):
        yy = y0 + 22 + li * 24
        cv2.putText(frame, text, (x0 + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, fl.color if li == 0 else (240, 240, 240), 1, cv2.LINE_AA)


def _build_overlays(analysis) -> list[FlightOverlay]:
    """VideoAnalysis → список FlightOverlay (трек в px + метрики + симуляция)."""
    fps = analysis.fps
    overlays: list[FlightOverlay] = []
    for idx, p in enumerate(analysis.passes):
        seg = getattr(p, "points_px", None)
        if not seg:
            track_by_frame = {q["frame"]: (float(q["x"]), float(q["y"]))
                              for q in analysis.ball_track_points}
            seg = [(f, x, y) for f, (x, y) in sorted(track_by_frame.items())
                   if p.release_frame <= f <= p.catch_frame]
        if len(seg) < 2:
            continue
        sim_pts: list[tuple[int, float, float]] = []
        fp = getattr(p.flight, "fit_params_px", None) if p.flight else None
        if p.flight and p.flight.method == "physics_fit" and fp and len(fp) == 5:
            from app.services.physics import simulate, G_PX
            dur = (p.catch_frame - p.release_frame) / fps
            ts, sx, sy = simulate(fp[0], fp[1], fp[2], fp[3], abs(fp[4]),
                                  G_PX, 1.0 / fps, int(dur * fps) + 2)
            sim_pts = [(p.release_frame + tt * fps, float(xx), float(yy))
                       for tt, xx, yy in zip(ts, sx, sy)]
        metrics = {
            "time_of_flight_s": p.flight.time_of_flight_s if p.flight else None,
            "apex_height_m": p.flight.apex_height_m if p.flight else None,
            "distance_m": p.flight.distance_m if p.flight else None,
            "initial_speed_mps": p.flight.initial_speed_mps if p.flight else None,
        }
        overlays.append(FlightOverlay(idx, p.release_frame, p.catch_frame,
                                      seg, metrics, sim_pts,
                                      direction=getattr(p, "direction", "left")))
    return overlays


def _draw_full_trajectory(frame: np.ndarray, analysis, fid: int,
                          ball_r: int, trail_length: int) -> None:
    """render_mode="full": полная траектория мяча ДО поиска параболических участков.

    Рисуется (для проверки гипотезы «баллистический фит не находится»):
      * линия сетки (центр кадра) — ориентир направления пасов;
      * ВСЕ точки трека мяча: уже пройденные — сплошная белая полилиния + точ-
        ки, текущая позиция — жёлтый маркер с хвостом; ещё не пройденные —
        тусклый пунктир (видно форму всей траектории заранее);
      * границы розыгрышей RALLY#k (по analysis.rallies);
      * баллистические LSQ-фиты БЕЗ фильтров из analysis.fit_diagnostics:
        зелёный = прошёл бы все фильтры (кандидат в пасы), красный = завалил
        фильтр (подпись FAIL:gravity/rmse/disp), голубой = direction away
        (летит от сетки); поверх фита рисуется его парабола-фит и подписи
        g_fit/g_ref, rmse_frac;
      * найденные пасы (analysis.passes) — как обычно, через draw_flight;
      * HUD: кадры/трек/покрытие трека баллистическими фитами (coverage).
    """
    h_img, w_img = frame.shape[:2]
    track = [(p["frame"], p["x"], p["y"]) for p in analysis.ball_track_points]
    if not track:
        return

    # --- линия сетки -----------------------------------------------------------
    net_x = int(w_img / 2)
    cv2.line(frame, (net_x, 0), (net_x, h_img - 1), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "NET", (net_x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # --- непройденная часть: тусклый пунктир ----------------------------------
    future = [(f, x, y) for f, x, y in track if f > fid]
    for i in range(0, len(future) - 1, 2):          # каждый второй сегмент → пунктир
        p0, p1 = future[i], future[i + 1]
        cv2.line(frame, (int(p0[1]), int(p0[2])), (int(p1[1]), int(p1[2])),
                 (90, 90, 90), 1, cv2.LINE_AA)

    # --- пройденная часть: сплошная полилиния + точки --------------------------
    past = [(f, x, y) for f, x, y in track if f <= fid][-400:]   # окно памяти кадра
    if len(past) >= 2:
        pts_np = np.array([(int(x), int(y)) for _, x, y in past], np.int32)
        cv2.polylines(frame, [pts_np.reshape(-1, 1, 2)], False,
                      (235, 235, 235), 1, cv2.LINE_AA)
    for f, x, y in past[::max(1, len(past) // 60)]:
        cv2.circle(frame, (int(x), int(y)), 1, (200, 200, 200), -1, cv2.LINE_AA)

    # --- текущая позиция мяча + короткий хвост --------------------------------
    cur = next(((f, x, y) for f, x, y in reversed(past) if f == fid), None)
    if cur is not None:
        tail = past[-trail_length // 3:]
        for i in range(len(tail) - 1):
            t = (i + 1) / max(len(tail) - 1, 1)
            col = tuple(int(c * t) for c in (0, 220, 255))
            cv2.line(frame, (int(tail[i][1]), int(tail[i][2])),
                     (int(tail[i + 1][1]), int(tail[i + 1][2])), col, 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(cur[1]), int(cur[2])), (0, 220, 255),
                       cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

    # --- границы розыгрышей -----------------------------------------------------
    for ri, r in enumerate(getattr(analysis, "rallies", []) or []):
        if r.start_frame <= fid <= r.end_frame + 12:
            x0, y0 = 8, h_img - 12 - ri * 18
            cv2.putText(frame, f"RALLY#{ri + 1} [{r.start_frame}-{r.end_frame}]",
                        (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)

    # --- баллистические фиты без фильтров ---------------------------------------
    diag = getattr(analysis, "fit_diagnostics", None) or {}
    fits = diag.get("fits", [])
    for ft in fits:
        if ft["start_frame"] > fid or ft["end_frame"] < fid - 30:
            continue                                   # рисуем вокруг окна фита
        pts = ft["points"]
        ok = ft["would_pass_filters"]
        away = ft["direction"] == "away"
        col = (80, 220, 80) if ok else ((120, 120, 255) if away else (60, 60, 255))
        seg = np.array([(int(x), int(y)) for _, x, y in pts], np.int32)
        cv2.polylines(frame, [seg.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
        mx = int(pts[len(pts) // 2][1])
        my = int(pts[len(pts) // 2][2]) - 14
        tag = ("FIT OK" if ok else "FAIL:" + ",".join(ft["failed_checks"]))
        if away:
            tag += " AWAY"
        lab = f"#{ft['fit_index']} {tag} g={ft['grav_err_rel']:.2f} rmse={ft['rmse_frac']:.3f}"
        cv2.putText(frame, lab, (mx - 40, my), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1, cv2.LINE_AA)

    # --- найденные пасы (как в обычном режиме) ----------------------------------
    overlays = _build_overlays(analysis)
    for o in overlays:
        if o.active(fid, 12):
            try:
                draw_flight(frame, o, ball_r, trail_length, fid,
                            hold_left=max(0, fid - o.catch_frame))
            except Exception:  # noqa: BLE001
                pass

    # --- HUD --------------------------------------------------------------------
    cov = diag.get("coverage_px", {})
    lines = [
        f"MODE FULL  f{fid}/{analysis.n_frames}",
        f"track={len(track)} ({analysis.detection_rate:.0%}) rallies={len(analysis.rallies)}",
        f"raw fits={diag.get('total_raw_fits', 0)} pass-filters={diag.get('total_would_pass', 0)}"
        f" passes={len(analysis.passes)}",
        f"ballistic coverage={cov.get('coverage', 0):.0%}",
    ]
    bw, bh = min(330, max(40, w_img - 20)), 4 * 20 + 12
    cv2.rectangle(frame, (10, 10), (10 + bw, 10 + bh), (0, 0, 0), -1)
    roi = frame[10:10 + bh, 10:10 + bw]
    cv2.addWeighted(roi.copy(), 0.55, frame[10:10 + bh, 10:10 + bw], 0.45, 0,
                    frame[10:10 + bh, 10:10 + bw])
    for li, text in enumerate(lines):
        cv2.putText(frame, text, (18, 10 + 18 + li * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def render_tracked_video(src_path: str, dst_path: str, analysis,
                         trail_length: int | None = None,
                         tail_hold_frames: int = 24,
                         mode: str | None = None) -> str:
    """Основной вход: исходное видео + VideoAnalysis → видео с траекториями.

    mode=None → settings.render_mode:
      "passes" — рисовать только траектории найденных пасов;
      "full"   — рисовать ПОЛНУЮ траекторию мяча до поиска параболических
                 участков + все баллистические фиты без фильтров (диагностика).

    Возвращает путь к результату. Покадрово сохраняет исходные кадры, поверх
    рисует активные(ый) пас(ы); завершённые пасы остаются «эхом» ещё
    tail_hold_frames кадров. Кадры вне всех полётов остаются чистыми.
    """
    trail_length = trail_length or settings.trail_length
    mode = (mode or settings.render_mode or "passes").lower()
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlays = [] if mode == "full" else _build_overlays(analysis)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():  # запасной кодек
        out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    ball_r = max(4, int(getattr(analysis, "ball_radius_px", 6)))
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if mode == "full":
            try:
                _draw_full_trajectory(frame, analysis, fid, ball_r, trail_length)
            except Exception:  # noqa: BLE001 — рендер не должен валить job
                traceback.print_exc()
        else:
            for o in overlays:
                if not o.active(fid, tail_hold_frames):
                    continue
                try:
                    draw_flight(frame, o, ball_r, trail_length, fid,
                                hold_left=max(0, fid - o.catch_frame))
                except Exception:  # noqa: BLE001 — рендер не должен валить job
                    pass
        out.write(frame)
    cap.release()
    out.release()
    return dst_path
