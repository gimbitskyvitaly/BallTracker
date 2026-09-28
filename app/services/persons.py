"""Стабильная детекция игроков без нейросети (fallback-каскад).

Проблемы прежней реализации (detector._cv_persons, квантованный HSV +
connected components):
  * фон на реальных видео НЕоднороден (трибуны, разметка, тени) — карта
    «всё, что не доминирующий цвет» распадается на сотни мелких лоскутов,
    ни один из которых не проходит фильтр h >= 0.15*H: persons == [] всегда;
  * следствие: ветка contact/release в pipeline бессильна — «мяч не
    детектится вообще, никаких отметок нет» (баг).

Здесь — классический background subtraction (OpenCV MOG2/KNN), который
спроектирован ровно под такие сцены: неподвижный фон отбрасывается,
остаются движущиеся объекты (игроки). Дополнительно:
  * морфология + медиана по bbox за скользящее окно — боксы не «мерцают»
    покадрово (стабильный track id игрока критичен для passer != catcher);
  * фильтр по высоте/ширине blob'а отсекает мяч и мелкий шум;
  * если движение есть, но все blob'ы мелкие (низкая камера, только ноги
    в кадре) —blob'ы склеиваются по горизонтали в «зоны активности», чтобы
    контакт мяча с игроком всё равно можно было подтвердить.

Все пороги настраиваются через .env (BT_PERSON_*).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from app.services.detector import Detection


class PersonDetectorCV:
    """MOG2/KNN background-subtraction детектор движущихся людей."""

    def __init__(self, person_cls: int = 0):
        self.PERSON_CLS = person_cls
        try:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=int(os.getenv("BT_BG_HISTORY", "300")),
                varThreshold=32,
                detectShadows=False,
            )
        except Exception:  # noqa: BLE001 — fallback на KNN в старых сборках
            self._bg = cv2.createBackgroundSubtractorKNN(history=300)
        H, W = int(os.getenv("BT_BG_HIST_BINS", "64")), int(os.getenv("BT_BG_WIDE_HIST", "48"))
        self._min_h_frac = float(os.getenv("BT_PERSON_MIN_H_FRAC", "0.12"))
        self._min_area_frac = float(os.getenv("BT_PERSON_MIN_AREA_FRAC", "0.0008"))
        self._max_w_frac = float(os.getenv("BT_PERSON_MAX_W_FRAC", "0.5"))
        self._hist_h, self._hist_w = H, W
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._prev_frame = None
        # скользящее окно усреднения боксов: гасит покадровое мерцание
        self._buf_len = int(os.getenv("BT_PERSON_SMOOTH_WINDOW", "5"))
        self._buf: dict[int, list[tuple[float, float, float, float]]] = {}
        self._next_id = 0
        self._warm = 0

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-6)

    def _match_id(self, box) -> int:
        best_id, best_iou = None, 0.2
        for pid, hist in self._buf.items():
            last = hist[-1]
            iou = self._iou(box, last)
            if iou > best_iou:
                best_id, best_iou = pid, iou
        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
        return best_id

    def update(self, frame: np.ndarray) -> list[Detection]:
        H_img, W_img = frame.shape[:2]
        fg = self._bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_h = self._min_h_frac * H_img
        min_area = self._min_area_frac * W_img * H_img
        max_w = self._max_w_frac * W_img
        blobs = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w > max_w:
                continue
            blobs.append([x, y, w, h, cv2.contourArea(c)])

        big = [b for b in blobs if b[3] >= min_h and b[2] >= 0.25 * b[3]]
        if not big and blobs:
            # движение есть, но вертикально вытянутых фигур нет (камера низко,
            # в кадре только ноги/частичные тела): склеиваем blob'ы по
            # горизонтальным проекциям в «зоны активности» игроков.
            blobs.sort(key=lambda b: b[0])
            merged = [blobs[0][:4] + [blobs[0][4]]]
            for b in blobs[1:]:
                m = merged[-1]
                mx2, bx2 = m[0] + m[2], b[0] + b[2]
                overlap_x = min(mx2, bx2) - max(m[0], b[0])
                gap = bx2 - mx2
                if overlap_x > 0 or gap < 0.05 * W_img:
                    nx1 = min(m[0], b[0]); ny1 = min(m[1], b[1])
                    nx2 = max(mx2, bx2); ny2 = max(m[1] + m[3], b[1] + b[3])
                    merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1,
                                  m[4] + b[4]]
                else:
                    merged.append(b[:4] + [b[4]])
            big = [m for m in merged
                   if m[3] >= 0.5 * min_h and m[4] >= min_area]

        raw: list[tuple[int, tuple[float, float, float, float], float]] = []
        for x, y, w, h, area in big:
            box = (float(x), float(y), float(x + w), float(y + h))
            pid = self._match_id(box)
            conf = float(min(0.9, 0.3 + 4.0 * area / float(W_img * H_img)))
            raw.append((pid, box, conf))

        # буферизируем и усредняем (скользящее окно)
        used_ids = set()
        out: list[Detection] = []
        for pid, box, conf in raw:
            buf = self._buf.setdefault(pid, [])
            buf.append(box)
            if len(buf) > self._buf_len:
                del buf[:-self._buf_len]
            used_ids.add(pid)
            arr = np.array(buf, dtype=float)
            sm = arr.mean(axis=0)
            out.append(Detection(sm, conf, self.PERSON_CLS))
        # мёртвые треки чистим, чтобы id не копились бесконечно
        for pid in [p for p in self._buf if p not in used_ids]:
            hist = self._buf[pid]
            hist.append(hist[-1])          # держим последний бокс ещё кадр
            if len(hist) > 2 * self._buf_len:
                del self._buf[pid]
        out.sort(key=lambda d: -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        return out
