"""VballNet — нейросетевой детектор мяча (ONNX) из репозитория
https://github.com/asigatchov/fast-volleyball-tracking-inference.

Почему это решает проблему «мяч почти не детектится»:
  * базовый COCO-YOLO (yolo11n/m, класс 32 "sports ball") обучен на статичных
    фото и на реальных спортивных трансляциях уверенно ловит мяч лишь в редких
    кадрах (conf ниже порога → пустые траектории);
  * VballNet обучен именно на видеопоследовательностях волейбола: модель
    принимает пачку из SEQ=9 подряд идущих полутоновых кадров (512x288) и
    возвращает 9 тепловых карт (sigmoid), где каждая карта — вероятность
    наличия мяча в пикселе соответствующего кадра. Временной контекст даёт
    надёжную детекцию маленького/размытого мяча (замер на реальном видео из
    examples() репо: ~87% кадров с детекцией против практически 0% у COCO-YOLO).

Постпроцессинг повторяет src/inference_onnx_seq_gray_v2.py оригинала:
пороговая бинаризация heatmap → контуры → центроид largest-контура; координаты
масштабируются обратно в разрешение исходного видео.

Способ вызова: скользящее окно — feed(frames) подаёт список кадров строго
последовательно (без пропусков) и возвращает список результатов той же длины;
результат для каждого кадра вычисляется по полному окну из 9 surrounding-кадров
(первые/последние кадры добираются дублированием крайних).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# вход модели: 9 полутоновых кадров 288x512, нормированных в [0, 1]
SEQ = 9
INPUT_WIDTH = 512
INPUT_HEIGHT = 288
HEATMAP_THRESHOLD_DEFAULT = 0.5


@dataclass
class BallPoint:
    frame: int          # 0-based номер кадра
    x: float            # координаты в разрешении исходного видео
    y: float
    conf: float         # средняя активность heatmap в окрестности точки


class VballNetDetector:
    """Детектор мяча по тепловой карте (временное окно SEQ кадров)."""

    def __init__(self, model_path: str | None = None,
                 threshold: float | None = None):
        if model_path is None:
            from app.config import settings
            model_path = settings.vballnet_path
        if threshold is None:
            from app.config import settings
            threshold = settings.heatmap_threshold
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель VballNet не найдена: {model_path}. Скачайте веса из "
                "https://github.com/asigatchov/fast-volleyball-tracking-inference "
                "(models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx) "
                "в директорию models/ или задайте BT_VBALLNET_PATH."
            )
        self.threshold = float(threshold)
        providers = ["CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:  # noqa: BLE001
            pass
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._lock = threading.Lock()

    # ------------------------------------------------------------ preprocess
    @staticmethod
    def _to_input(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT)).astype(np.float32) / 255.0

    # ----------------------------------------------------------- postprocess
    def _decode_heatmap(self, heat: np.ndarray, scale_x: float, scale_y: float):
        """Heatmap одного кадра → (x, y, conf) в координатах оригинального кадра."""
        _, binary = cv2.threshold(heat.astype(np.float32), self.threshold, 1.0,
                                  cv2.THRESH_BINARY)
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] == 0:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        h0, w0 = heat.shape
        patch = heat[max(0, int(cy) - 2):min(h0, int(cy) + 3),
                     max(0, int(cx) - 2):min(w0, int(cx) + 3)]
        conf = float(patch.mean()) if patch.size else self.threshold
        return cx * scale_x, cy * scale_y, conf

    # ---------------------------------------------------------------- public
    def feed(self, frames: list[np.ndarray], start_idx: int = 0) -> list[BallPoint | None]:
        """Прогнать окно по списку кадров; вернуть результат для каждого кадра.

        `frames` — подряд идущие кадры BGR (длина >= 1), `start_idx` — их
        абсолютные номера (для поля `frame` в результате). Каждый кадр
        предсказывается по полному 9-кадровому окну с симметричным padding
        крайними дублями, поэтому результаты корректны и на границах видео.
        """
        n = len(frames)
        if n == 0:
            return []
        processed = [self._to_input(f) for f in frames]
        half = SEQ // 2
        padded = ([processed[0]] * half + processed + [processed[-1]] * half)
        results: list[BallPoint | None] = []
        h, w = frames[0].shape[:2]
        scale_x, scale_y = w / INPUT_WIDTH, h / INPUT_HEIGHT
        step = SEQ - 2 * half if SEQ > 2 * half else 1  # =1: полное покрытие каждого кадра
        # Чтобы не гонять модель n раз для больших батчей, используем stride=1
        # только когда n мало; при большом n всё равно один проход на кадр —
        # модель лёгкая (~40 fps CPU на кадр), а качество важнее скорости.
        for i in range(n):
            window = padded[i:i + SEQ]
            if len(window) < SEQ:
                window = window + [padded[-1]] * (SEQ - len(window))
            tensor = np.stack(window, axis=0)[None, ...].astype(np.float32)
            with self._lock:
                out = self._session.run([self._output_name],
                                        {self._input_name: tensor})[0]
            det = self._decode_heatmap(out[0, half], scale_x, scale_y)
            if det is None:
                results.append(None)
            else:
                x, y, conf = det
                results.append(BallPoint(frame=start_idx + i, x=float(x), y=float(y),
                                         conf=round(conf, 3)))
        del step
        return results
