"""Физическая модель полёта мяча (подход BallTime™).

BallTime™ восстанавливает траекторию и время в воздухе, подгоняя
физическую модель баллистики с сопротивлением воздуха к наблюдаемым
точкам трека. Модель (плоскость изображения, y вниз):

    dv/dt = g - (k/m) * |v| * v          (гравитация + quadratic drag)

Интегрируется численно (RK4). Параметры [x0, y0, vx0, vy0, k]
подбираются least_squares так, чтобы предсказанные позиции совпали
с наблюдениями трекера. Это даёт:

  * оценку времени полёта даже при пропусках детекции (окклюзия);
  * начальную скорость паса;
  * вершину траектории и точку приземления/приёмки.

Масштаб: перевод px -> метры через диаметр мяча (volleyball ≈ 0.21 м,
basketball ≈ 0.24 м, football ≈ 0.22 м) — по умолчанию волейбольный.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

G_MPS2 = 9.81
BALL_DIAMETER_M = {"volleyball": 0.21, "basketball": 0.24, "football": 0.22,
                   "tennis": 0.067}


@dataclass
class FlightEstimate:
    time_of_flight_s: float
    release_frame: int
    catch_frame: int
    apex_height_m: float
    distance_m: float
    initial_speed_mps: float
    peak_speed_mps: float
    fit_rmse_px: float
    trajectory: list[dict] = field(default_factory=list)  # [{t, x_m, y_m}]
    fit_params_px: list[float] | None = None  # [x0,y0,vx0,vy0,k] в px — для рендера
    method: str = "physics_fit"   # или "tracked_direct"


def _rk4_step(x, y, vx, vy, k, g, dt):
    """Один шаг RK4 для dv/dt = (0, g) - k*|v|v (y направлен вниз)."""

    def acc(vx_, vy_):
        sp = math.hypot(vx_, vy_)
        return (-k * sp * vx_, g - k * sp * vy_)

    a1x, a1y = acc(vx, vy)
    x2, y2 = x + vx * dt / 2, y + vy * dt / 2
    vx2, vy2 = vx + a1x * dt / 2, vy + a1y * dt / 2
    a2x, a2y = acc(vx2, vy2)
    x3, y3 = x + vx2 * dt / 2, y + vy2 * dt / 2
    vx3, vy3 = vx + a2x * dt / 2, vy + a2y * dt / 2
    a3x, a3y = acc(vx3, vy3)
    x4, y4 = x + vx3 * dt, y + vy3 * dt
    vx4, vy4 = vx + a3x * dt, vy + a3y * dt
    a4x, a4y = acc(vx4, vy4)
    x = x + dt / 6 * (vx + 2 * vx2 + 2 * vx3 + vx4)
    y = y + dt / 6 * (vy + 2 * vy2 + 2 * vy3 + vy4)
    vx = vx + dt / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
    vy = vy + dt / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)
    return x, y, vx, vy


def simulate(x0, y0, vx0, vy0, k, g, dt, n_steps, ground_y=None):
    """Численная траектория полёта мяча (списки ts, xs, ys)."""
    xs, ys, ts = [float(x0)], [float(y0)], [0.0]
    x, y, vx, vy = float(x0), float(y0), float(vx0), float(vy0)
    for i in range(int(n_steps)):
        x, y, vx, vy = _rk4_step(x, y, vx, vy, k, g, dt)
        if not (math.isfinite(x) and math.isfinite(y)):
            break
        xs.append(x); ys.append(y); ts.append((i + 1) * dt)
        if ground_y is not None and y >= ground_y and vy > 0 and i > 2:
            break
    return np.array(ts), np.array(xs), np.array(ys)


# глобальное ускорение в px/s^2 (задаётся перед фитом)
G_PX = 500.0


def set_gravity_px(fps: float, ratio: float) -> None:
    global G_PX
    G_PX = ratio * fps * fps


def fit_physics_trajectory(frames: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                           fps: float, ball_diam_px: float,
                           drag_coefficient: float) -> dict | None:
    """Подгонка физмодели к точкам (frames, xs, ys). Возвращает параметры фита."""
    t = frames / fps
    n = len(t)
    if n < 4:
        return None
    g = G_PX

    # начальные догадки из полиномиальной аппроксимации (парабола по времени)
    cx = np.polyfit(t, xs, 1)
    cy = np.polyfit(t, ys, 2)          # y ~ 0.5*g*t^2 + vy0*t + y0
    vx0 = float(cx[0])
    vy0 = float(cy[0] * 0 + cy[1])     # линейный член
    dt = 1.0 / fps
    n_steps = int(round((t[-1] - t[0]) * fps)) + 4

    def _simulate(params, horizon_steps):
        x0, y0, ivx, ivy, k = params
        ts, simx, simy = simulate(x0, y0, ivx, ivy, abs(k), g, dt, horizon_steps)
        if len(ts) < 2:
            return None
        px = np.interp(t, ts, simx)
        py = np.interp(t, ts, simy)
        return px, py

    horizon = n_steps + int(fps)

    def residuals(params):
        m = _simulate(params, horizon)
        if m is None:
            return np.full(2 * n, 1e6)
        px, py = m
        r = np.concatenate([px - xs, py - ys])
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    try:
        sol = least_squares(
            residuals,
            x0=[xs[0], ys[0], vx0, vy0, drag_coefficient],
            bounds=([xs[0] - 80, ys[0] - 80, -4000, -4000, 0.0],
                    [xs[0] + 80, ys[0] + 80, 4000, 4000, 0.5]),
            max_nfev=500,
        )
    except Exception:
        return None

    m = _simulate(sol.x, horizon)
    if m is None:
        return None
    px, py = m
    rmse = float(np.sqrt(np.mean((px - xs) ** 2 + (py - ys) ** 2)))
    ts_full, sx, sy = simulate(*sol.x[:4], abs(sol.x[4]), g, dt, horizon)
    speeds = []
    prev = None
    for xx, yy in zip(sx, sy):
        if prev is not None:
            speeds.append(math.hypot(xx - prev[0], yy - prev[1]) / dt)
        prev = (xx, yy)
    return {
        "params": sol.x.tolist(),
        "rmse_px": rmse,
        "sim_t": ts_full.tolist(),
        "sim_x": sx.tolist(),
        "sim_y": sy.tolist(),
        "v0_px_s": math.hypot(sol.x[2], sol.x[3]),
        "peak_speed_px_s": max(speeds) if speeds else 0.0,
    }


def estimate_flight(tracked_points: list[tuple[int, float, float]], fps: float,
                    ball_diam_px: float, drag: float,
                    use_physics: bool = True) -> FlightEstimate | None:
    """Главная функция: по точкам трека мяча оценить параметры полёта.

    tracked_points: [(frame_id, cx_px, cy_px), ...] — один связный полёт.
    Время в воздухе = момент релиза (первая точка с ускорением вверх/от руки)
    до момента приёмки (последняя точка трека или пересечение «зоны игрока»).
    """
    if len(tracked_points) < 3:
        return None
    pts = np.array(tracked_points, dtype=float)
    frames, xs, ys = pts[:, 0], pts[:, 1], pts[:, 2]
    scale = BALL_DIAMETER_M["volleyball"] / max(ball_diam_px, 1.0)  # м/px

    rel_f, cat_f = int(frames[0]), int(frames[-1])
    tof_direct = (cat_f - rel_f) / fps

    result = None
    if use_physics and len(frames) >= 4:
        fit = fit_physics_trajectory(frames, xs, ys, fps, ball_diam_px, drag)
        if fit and fit["rmse_px"] < ball_diam_px * 2.5:
            st, sx, sy = np.array(fit["sim_t"]), np.array(fit["sim_x"]), np.array(fit["sim_y"])
            # физика уточняет моменты: релиз = t=0 модели, приёмка = конец наблюдений
            traj = [{"t": round(float(tt), 3),
                     "x_m": round(float(xx) * scale, 3),
                     "y_m": round(float(yy) * scale, 3),
                     "x_px": round(float(xx), 1), "y_px": round(float(yy), 1)}
                    for tt, xx, yy in zip(st, sx, sy)]
            apex_m = (sy.min() - ys[0]) * scale * -1  # y вниз => высота = -(y-y0)
            dist_m = math.hypot(sx[-1] - sx[0], sy[-1] - sy[0]) * scale
            result = FlightEstimate(
                time_of_flight_s=round(tof_direct, 3),
                release_frame=rel_f, catch_frame=cat_f,
                apex_height_m=round(max(apex_m, 0.0), 3),
                distance_m=round(dist_m, 2),
                initial_speed_mps=round(fit["v0_px_s"] * scale, 2),
                peak_speed_mps=round(fit["peak_speed_px_s"] * scale, 2),
                fit_rmse_px=round(fit["rmse_px"], 2),
                trajectory=traj, method="physics_fit",
                fit_params_px=[float(v) for v in fit["params"]],
            )
    if result is None:
        apex_px = ys.min()
        # height_m: высота относительно релиза (вверх положительная)
        traj = [{"t": round(float((f - rel_f) / fps), 3),
                 "x_m": round(float(x) * scale, 3),
                 "height_m": round(float(ys[0] - y) * scale, 3),
                 "x_px": round(float(x), 1), "y_px": round(float(y), 1)}
                for f, x, y in tracked_points]
        result = FlightEstimate(
            time_of_flight_s=round(tof_direct, 3),
            release_frame=rel_f, catch_frame=cat_f,
            apex_height_m=round(max((ys[0] - apex_px) * scale, 0.0), 3),
            distance_m=round(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) * scale, 2),
            initial_speed_mps=round(math.hypot(xs[1] - xs[0], ys[1] - ys[0]) * fps * scale, 2),
            peak_speed_mps=round(max(np.hypot(np.diff(xs), np.diff(ys))) * fps * scale, 2),
            fit_rmse_px=-1.0, trajectory=traj, method="tracked_direct",
        )
    return result
