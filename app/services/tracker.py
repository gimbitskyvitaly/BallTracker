"""SORT-трекер (Simple Online and Realtime Tracking).

Классический алгоритм из статьи Bewley et al. 2016, тот же подход
используется в BallTime™ (там — DeepSORT с appearance-эмбеддингами).

Компоненты:
  - KalmanFilter — предсказание позиции [x, y, a, h, vx, vy, va, vh]
    (центр, аспект бокса, высота) с моделью постоянного скорости;
  - linear_assignment (scipy) — венгерское сопоставление по стоимости IoU;
  - ведение треков: confirmed / time_since_update (устойчивость к окклюзиям).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cho_factor, cho_solve

from app.config import settings


class KalmanBoxTracker:
    """Фильтр Калмана для одного объекта (bbox xywh-центрическое представление)."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        # состояние: [cx, cy, aspect, h, vx, vy, va, vh]
        self.dim_w = 8
        dt = 1.0
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        self.x = np.array([cx, cy, w / h if h else 1.0, h, 0, 0, 0, 0], dtype=float)

        self.F = np.eye(self.dim_w)
        self.F[:4, 4:] = np.eye(4) * dt

        self.H = np.zeros((4, self.dim_w))
        self.H[:4, :4] = np.eye(4)

        std_pos = max(h, 10.0)
        std_vel = max(h * 2.0, 50.0)
        self.P = np.diag([std_pos**2, std_pos**2, 1.0, std_pos**2,
                          std_vel**2, std_vel**2, 1.0, std_vel**2])

        self.R = np.diag([max(h / 2, 2.0)**2] * 3 + [max(h / 2, 2.0)**2])
        self.Q = np.eye(self.dim_w) * 0.1
        self.Q[4:, 4:] *= 0.01

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.time_since_update = 0
        self.history: list[np.ndarray] = []   # [(frame, cx, cy, w, h), ...]
        self._frames: list[int] = []

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])

    def predict(self, frame_id: int) -> np.ndarray:
        """Предсказывает bbox на следующем кадре; возвращает текущий прогноз."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        self.history.append(np.array([frame_id, *self._state_bbox()]))
        self._frames.append(frame_id)
        return self.get_state()

    def update(self, bbox: np.ndarray, frame_id: int) -> None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        z = np.array([x1 + w / 2.0, y1 + h / 2.0, w / h if h else 1.0, h])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_w) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.hits += 1
        self.time_since_update = 0
        if self.history:
            self.history[-1] = np.array([frame_id, *self._state_bbox()])

    def _state_bbox(self) -> tuple[float, float, float, float]:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return cx, cy, w, h

    def get_state(self) -> np.ndarray:
        cx, cy, w, h = self._state_bbox()
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Матрица IoU между двумя наборами xyxy-боксов."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((len(bb_test), len(bb_gt)))
    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_t = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_g = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_t[:, None] + area_g[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Венгерское сопоставление детекций и треков по IoU."""
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, 5), dtype=float))
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            cost = -iou_matrix
            r, c = linear_sum_assignment(cost)
            matched_indices = np.stack([r, c], axis=1) if len(r) else np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(detections.shape[0])
                      if d not in matched_indices[:, 0]] if len(matched_indices) else list(range(detections.shape[0]))
    unmatched_trks = [t for t in range(trackers.shape[0])
                      if t not in matched_indices[:, 1]] if len(matched_indices) else list(range(trackers.shape[0]))

    results = np.stack([matched_indices[:, 0],
                        matched_indices[:, 1],
                        iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]], axis=1) \
        if len(matched_indices) else np.empty((0, 3))
    return (results.astype(int) if len(results) else results,
            np.array(unmatched_dets, dtype=int),
            np.array(unmatched_trks, dtype=int))


class SORTTracker:
    """Менеджер жизненного цикла треков (пороги — из Settings)."""

    def __init__(self, max_age: int | None = None, min_hits: int | None = None,
                 iou_threshold: float | None = None):
        self.max_age = max_age if max_age is not None else settings.max_age
        self.min_hits = min_hits if min_hits is not None else settings.min_hits
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.frame_size: tuple[int, int] | None = None   # (W, H) для физ. gate

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (width, height)

    def _gate_cost(self, dets: np.ndarray, predicted_arr: np.ndarray) -> np.ndarray:
        """Стоимостная матрица: -IoU для допустимых пар, +inf для запрещённых.

        Пара детекция-трек отклоняется, если:
          * центр предсказанного бокса трека вышел за кадр с запасом в его
            ширину/высоту (сорванный трек улетает за границу по инерции);
          * расстояние между центрами больше max_jump_frac * min(W, H)
            (нефизичный скачок — вероятная ложная детекция).
        """
        iou = np.asarray(iou_batch(dets, predicted_arr), dtype=float)
        cost = -iou                              # форма (n_det, n_trk)
        dcx = (dets[:, 0] + dets[:, 2])[:, None] / 2.0
        dcy = (dets[:, 1] + dets[:, 3])[:, None] / 2.0
        tcx = (predicted_arr[None, :, 0] + predicted_arr[None, :, 2]) / 2.0
        tcy = (predicted_arr[None, :, 1] + predicted_arr[None, :, 3]) / 2.0
        dist = np.hypot(dcx - tcx, dcy - tcy)
        if self.frame_size:
            w_img, h_img = self.frame_size
            limit = settings.max_jump_frac * min(w_img, h_img)
            cost = np.where(dist > limit, np.inf, cost)
            tw = (predicted_arr[:, 2] - predicted_arr[:, 0])
            th = (predicted_arr[:, 3] - predicted_arr[:, 1])
            # трек «за кадром»: центр предсказания вышел за границу более чем
            # на размер собственного бокса — сопоставление с ним запрещено
            out_x = (tcx[0] < -tw) | (tcx[0] > w_img + tw)
            out_y = (tcy[0] < -th) | (tcy[0] > h_img + th)
            offscreen = (out_x | out_y)[None, :]   # (1, n_trk) -> broadcast по dets
            cost = np.where(offscreen, np.inf, cost)
        return cost

    def update(self, dets: np.ndarray) -> list[tuple[int, np.ndarray, float, float]]:
        """Принимает Nx4 xyxy-массив лучших детекций мяча.

        Возвращает список (track_id, bbox_xyxy, vx, vy) для треков, у которых
        есть наблюдение в текущем кадре ИЛИ трек переживает короткую окклюзию
        (time_since_update <= max_age) — так траектория полёта остаётся
        непрерывной при пропусках детекции (подход BallTime™).

        Физический gate (защита от срывов): если предсказанная позиция трека
        уходит за пределы кадра или прыжок детекции к треку превышает
        max_jump_frac * min(W,H) — сопоставление запрещена: такая пара либо
        ложная детекция, либо «сорванный» трек. Трек без подтверждений быстро
        умирает (max_age мал), и захват начнётся заново с чистой детекции.
        Размеры кадра задаются через set_frame_size() (по умолчанию — без gate
        по границам, обратная совместимость со старыми тестами).
        """
        self.frame_count += 1

        # предсказание всех треков
        predicted = []
        to_remove = []
        for tr in self.trackers:
            try:
                predicted.append(tr.predict(self.frame_count))
            except Exception:
                to_remove.append(tr)
        for tr in to_remove:
            self.trackers.remove(tr)
        predicted_arr = np.array(predicted) if predicted else np.empty((0, 4))

        if len(dets) and len(predicted_arr):
            cost = self._gate_cost(dets, predicted_arr)
            finite = np.isfinite(cost)
            matched_d: dict[int, int] = {}      # det_idx -> trk_idx
            matched_t: dict[int, int] = {}      # trk_idx -> det_idx
            cand_lists = []
            for d in range(dets.shape[0]):
                cand = np.where(finite[d])[0]
                if len(cand):
                    cand = cand[(-cost[d, cand]) >= self.iou_threshold]
                cand_lists.append(cand)
            # одна венгерская задача по всем допустимым парам (гарантирует
            # взаимно-однозначное сопоставление: один трек <- одна детекция)
            ncols = predicted_arr.shape[0]
            big = float(np.nanmax(cost[np.isfinite(cost)])) + 1e6 \
                if np.isfinite(cost).any() else 1e6
            full = np.full((dets.shape[0], ncols), big)
            for d, cand in enumerate(cand_lists):
                if len(cand):
                    full[d, cand] = cost[d, cand]
            r_, c_ = linear_sum_assignment(full)
            for di, ti in zip(r_, c_):
                if full[di, ti] < big:
                    matched_d[int(di)] = int(ti)
                    matched_t[int(ti)] = int(di)
            for di, ti in matched_d.items():
                self.trackers[ti].update(dets[di], self.frame_count)
            for i in range(dets.shape[0]):
                if i not in matched_d:
                    self.trackers.append(KalmanBoxTracker(dets[i]))
        elif len(dets):
            for d in dets:
                self.trackers.append(KalmanBoxTracker(d))

        # очистка старых + физическая чистка «сорванных» треков за кадром
        out: list[tuple[int, np.ndarray, float, float]] = []
        alive: list[KalmanBoxTracker] = []
        for tr in self.trackers:
            if tr.time_since_update > self.max_age:
                continue
            if self.frame_size and tr.time_since_update > 0:
                # трек живёт только на предсказаниях и ушёл далеко за кадр —
                # это сорванный объект, немедленно снимаем его с наблюдения
                cx, cy, w, h = tr._state_bbox()
                W, H = self.frame_size
                margin = max(w, h, 2.0 * settings.max_jump_frac * min(W, H))
                if cx < -margin or cx > W + margin or cy < -margin or cy > H + margin:
                    continue
            alive.append(tr)
            confirmed = (self.frame_count <= self.min_hits or tr.hits >= self.min_hits)
            if confirmed and tr.time_since_update <= self.max_age:
                vx, vy = tr.velocity
                out.append((tr.id, tr.get_state(), vx, vy))
        self.trackers = alive
        return out

    def tracks_history(self) -> dict[int, list[tuple[int, float, float]]]:
        """История {track_id: [(frame, cx, cy), ...]} для анализа полёта."""
        hist: dict[int, list[tuple[int, float, float]]] = {}
        for tr in self.trackers:
            pts = []
            for row in tr.history:
                f, cx, cy, w, h = row
                pts.append((int(f), float(cx), float(cy)))
            hist[tr.id] = pts
        return hist
