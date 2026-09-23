"""Отрисовка траекторий поверх исходного видео (аналог оверлея BallTime™).

Результат — видео, покадрово идентичное загруженному, но во время полёта мяча
за ним формируется «шлейф» траектории:

  * затухающий хвост из точек/линий за текущим положением мяча;
  * полная дуга паса пунктиром + маркеры RELEASE / CATCH;
  * при наличии фита физмодели — тонкая линия смоделированной траектории;
  * HUD-панель: номер паса, Time-of-Flight, apex, дальность, начальная скорость.

Всё рисуется в пиксельных координатах трека (trajectory в API отдаётся в
метрах; здесь используется original x_px/y_px), поэтому привязка к кадру
точная независимо от калибровки масштаба.
"""

from __future__ import annotations

import math
import os
from collections import deque

import cv2
import numpy as np

from app.config import settings

# палитра пасов (BGR)
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
    """Один полёт (пас) с подготовленными для рендера данными."""

    def __init__(self, index: int, release_frame: int, catch_frame: int,
                 pts_px: list[tuple[int, float, float]],
                 metrics: dict | None = None,
                 sim_pts_px: list[tuple[int, float, float]] | None = None):
        self.index = index
        self.release_frame = int(release_frame)
        self.catch_frame = int(catch_frame)
        self.pts = pts_px                       # трек мяча в px: [(frame,x,y)]
        self.sim_pts = sim_pts_px or []         # симуляция физмодели в px
        self.metrics = metrics or {}
        self.color = _COLORS[index % len(_COLORS)]

    def contains(self, frame_id: int) -> bool:
        return self.release_frame <= frame_id <= self.catch_frame


def draw_flight(frame: np.ndarray, fl: FlightOverlay, ball_radius: int,
                trail_length: int, show_full_arc: bool = True) -> None:
    """Рисуем на frame всё, что относится к активному полёту fl."""
    fid = getattr(fl, "_cur_frame", None)
    cur = _point_at_time(fl.pts, float(fid))
    if cur is None:
        return
    cx, cy = cur

    # --- маркеры RELEASE / CATCH (рисуем первыми — они не перекрываются дугой)
    rx, ry = fl.pts[0][1], fl.pts[0][2]
    kx, ky = fl.pts[-1][1], fl.pts[-1][2]
    for (px, py), label in (((rx, ry), "RELEASE"), ((kx, ky), "CATCH")):
        cv2.drawMarker(frame, (int(px), int(py)), fl.color, cv2.MARKER_CROSS,
                       16, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (int(px) + 10, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fl.color, 1, cv2.LINE_AA)

    h_img, w_img = frame.shape[:2]
    # --- полная дуга паса «призрачной» линией (формируется по мере полёта) --
    if show_full_arc and len(fl.pts) > 1:
        done = [p for p in fl.pts if p[0] <= fid]
        pts_np = np.array([(int(x), int(y)) for _, x, y in done], dtype=np.int32)
        if len(pts_np) >= 2:
            mask = np.zeros((h_img, w_img), np.uint8)
            cv2.polylines(mask, [pts_np.reshape(-1, 1, 2)], False, 255, 2)
            region = mask.astype(bool)
            col_arr = np.array(fl.color, np.float64)
            f32 = frame.astype(np.float64)
            blended = f32 * 0.6 + col_arr * 0.4
            frame[region] = blended[region].astype(np.uint8)

    # --- линия физмодели (если есть) -----------------------------------------
    if len(fl.sim_pts) > 1:
        sim_done = [p for p in fl.sim_pts if p[0] <= fid]
        cur_sim = _point_at_time(fl.sim_pts, float(fid))
        if cur_sim is not None:
            sim_done = sim_done + [(fid, cur_sim[0], cur_sim[1])]
        for i in range(len(sim_done) - 1):
            p0, p1 = sim_done[i], sim_done[i + 1]
            cv2.line(frame, (int(p0[1]), int(p0[2])), (int(p1[1]), int(p1[2])),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # --- затухающий хвост ------------------------------------------------------
    past = [p for p in fl.pts if p[0] <= fid][-trail_length:]
    if cur is not None and (not past or past[-1][0] != fid):
        past = past + [(fid, cx, cy)]
    n = len(past)
    for i in range(n - 1):
        f0, x0, y0 = past[i]
        f1, x1, y1 = past[i + 1]
        t = (i + 1) / max(n - 1, 1)               # 0..1, к мячу ярче
        col = tuple(int(c * t) for c in fl.color)
        thick = max(1, int(round(1 + 3 * t)))
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), col, thick, cv2.LINE_AA)
    for i in range(0, n, max(1, n // 12)):        # точки-маркеры вдоль хвоста
        _, x, y = past[i]
        cv2.circle(frame, (int(x), int(y)), 2, fl.color, -1, cv2.LINE_AA)

    # --- маркеры RELEASE / CATCH ----------------------------------------------
    # --- текущее положение мяча -----------------------------------------------
    cv2.circle(frame, (int(cx), int(cy)), ball_radius + 3, fl.color, 2, cv2.LINE_AA)

    # --- HUD --------------------------------------------------------------------
    m = fl.metrics
    lines = [f"PASS #{fl.index + 1}"]
    if m.get("time_of_flight_s") is not None:
        lines.append(f"TOF {m['time_of_flight_s']:.2f}s")
    if m.get("apex_height_m") is not None:
        lines.append(f"apex {m['apex_height_m']:.2f}m")
    if m.get("distance_m") is not None:
        lines.append(f"dist {m['distance_m']:.1f}m")
    if m.get("initial_speed_mps") is not None:
        lines.append(f"v0 {m['initial_speed_mps']:.1f}m/s")
    fw, fh = frame.shape[1], frame.shape[0]
    bw = min(170, max(40, fw - 20))
    bh = min(24 * len(lines) + 12, max(20, fh - 20))
    x0, y0 = 10, 10
    roi = frame[y0:y0 + bh, x0:x0 + bw]
    hud = roi.copy()
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
    cv2.addWeighted(hud, 0.55, frame[y0:y0 + bh, x0:x0 + bw], 0.45, 0,
                    frame[y0:y0 + bh, x0:x0 + bw])
    for li, text in enumerate(lines):
        yy = y0 + 22 + li * 24
        cv2.putText(frame, text, (x0 + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, fl.color if li == 0 else (240, 240, 240), 1, cv2.LINE_AA)


def render_tracked_video(src_path: str, dst_path: str, analysis,
                         trail_length: int | None = None) -> str:
    """Основной вход: исходное видео + VideoAnalysis → видео с траекториями.

    Возвращает путь к результату. Покадрово сохраняет исходные кадры, поверх
    рисует активный(е) полёт(ы). Если кадр вне всех полётов — остаётся чистым.
    """
    trail_length = trail_length or settings.trail_length
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # строим оверлеи из анализа: трек в px + метрики + симуляция (если physics_fit)
    overlays: list[FlightOverlay] = []
    track_by_frame = {p["frame"]: (float(p["x"]), float(p["y"]))
                      for p in analysis.ball_track_points}
    for idx, p in enumerate(analysis.passes):
        seg = [(f, x, y) for f, (x, y) in sorted(track_by_frame.items())
               if p.release_frame <= f <= p.catch_frame]
        if len(seg) < 2:
            continue
        sim_pts: list[tuple[int, float, float]] = []
        fp = getattr(p.flight, "fit_params_px", None) if p.flight else None
        if p.flight and p.flight.method == "physics_fit" and fp and len(fp) == 5:
            # разворачиваем физмодель заново в пиксельных координатах
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
                                      seg, metrics, sim_pts))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():  # запасной кодек
        out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    ball_r = max(4, int(getattr(analysis, "ball_radius_px", 6)))
    active: list[FlightOverlay] = []
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        active = [o for o in overlays if o.contains(fid)]
        for o in active:
            o._cur_frame = fid
            try:
                draw_flight(frame, o, ball_r, trail_length)
            except Exception:  # noqa: BLE001 — рендер не должен валить job
                pass
        out.write(frame)
    cap.release()
    out.release()
    return dst_path
