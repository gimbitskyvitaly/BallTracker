"""Конфигурация сервиса (параметры в стиле BallTime™).

BallTime™ (Second Spectrum / Snap) использует:
  - детектор мяча на базе YOLO, обученного на спортивных данных;
  - трекер SORT-семейства (в оригинале — DeepSORT) для восстановления траектории;
  - физическую модель полёта с воздушным сопротивлением (drag + Magnus),
    которая позволяет оценить время в воздухе даже при окклюзиях мяча.

Здесь собраны все настраиваемые параметры единым конфигом.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- Модель детекции ----------------------------------------------------
    # sports-specific веса (TrackNet-подобные) дают лучшее качество на мяче,
    # но COCO-класс "sports ball" (id=32) из yolo11n работает из коробки.
    model_path: str = os.getenv("BT_MODEL_PATH", "app/yolo11n.pt")
    device: str = os.getenv("BT_DEVICE", "cpu")          # "cpu" / "0" (GPU)
    conf_threshold: float = float(os.getenv("BT_CONF", "0.15"))
    class_ids: tuple[int, ...] = (32,)                    # COCO: 32 = sports ball

    # --- Трекинг (SORT / IoU-сопоставление) ---------------------------------
    max_age: int = 12              # кадров держать трек без детекций (окклюзии)
    min_hits: int = 2              # мин. подтверждений, чтобы трек считался валидным
    iou_threshold: float = 0.25    # порог IoU для венгерского сопоставления

    # --- Детекция событий паса ------------------------------------------------
    release_speed_px: float = 6.0   # мгновенная скорость (px/кадр) для "отрыва"
    catch_radius_scale: float = 3.0 # радиус "приёмки" относительно диагонали бокса игрока
    contact_expand: float = 0.45    # расширение бокса игрока для зоны контакта мяч-игрок
    min_flight_frames: int = 4      # минимальная длительность полёта в кадрах

    # --- Физика полёта ---------------------------------------------------------
    # Оценка g в px/с^2: gravity_px_per_s2 = fps^2 * gravity_ratio
    # (для волейбола ~9.8 м/с^2 при мяче ~0.22 м => ~19 единиц diam/s^2;
    #  на видео мяч занимает ~15-25% высоты кадра, коэффициент подобран эмпирически)
    gravity_ratio: float = float(os.getenv("BT_GRAVITY_RATIO", "0.55"))
    drag_coefficient: float = 0.02     # безразмерный коэффициент сопротивления воздуха
    use_physics_fit: bool = True       # подгонять баллистическую модель к трек-данным

    # --- Хранилище ------------------------------------------------------------
    db_path: str = os.getenv("BT_DB_PATH", "data/balltime.db")
    upload_dir: str = os.getenv("BT_UPLOAD_DIR", "data/uploads")

    tracknet_weights: str = field(default="")  # заглушка под TrackNet-веса


settings = Settings()
