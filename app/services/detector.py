"""Детекция мяча и игроков на кадрах видео с помощью YOLO (ultralytics).

Используется предобученная модель yolo11n (COCO): класс 32 — "sports ball",
класс 0 — "person". Для продакшена рекомендуется fine-tuned sports-веса
(TrackNet / кастомный датасет) — путь задаётся через BT_MODEL_PATH.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from app.config import settings


@dataclass
class Detection:
    """Одна детекция: [x1, y1, x2, y2], confidence, class_id."""

    bbox: np.ndarray
    conf: float
    cls: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def diag(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(np.hypot(w, h))


class BallDetector:
    """Потокобезопасная обёртка над YOLO для покадровой детекции."""

    BALL_CLS = 32
    PERSON_CLS = 0

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self._model = YOLO(model_path or settings.model_path)
        self._device = device or settings.device
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """Возвращает (мячи, игроки) для одного кадра BGR."""
        with self._lock:
            results = self._model.predict(
                frame,
                conf=settings.conf_threshold,
                classes=[self.BALL_CLS, self.PERSON_CLS],
                device=self._device,
                verbose=False,
            )
        balls: list[Detection] = []
        persons: list[Detection] = []
        r = results[0]
        if r.boxes is None:
            return balls, persons
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            det = Detection(bbox=xyxy, conf=conf, cls=cls)
            if cls == self.BALL_CLS:
                balls.append(det)
            else:
                persons.append(det)
        # мяч: берём наиболее вероятную детекцию первой
        balls.sort(key=lambda d: -d.conf)
        persons.sort(key=lambda d: -d.conf)
        return balls, persons
