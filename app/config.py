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
from dotenv import load_dotenv


load_dotenv()

@dataclass
class Settings:
    # --- Модель детекции ----------------------------------------------------
    # ВАЖНО: базовая yolo11n (nano, COCO) почти НЕ детектит маленький/размытый
    # мяч на реальных спортивных видео (conf < 0.1 → flights=[]). По умолчанию
    # используется yolo11m (medium): ~4x точнее на мелких объектах, COCO-класс
    # 32 "sports ball" работает из коробки. Для продакшена ещё лучше:
    #   - fine-tuned YOLO на Roboflow sports-датасетах (soccer/tennis/volleyball);
    #   - TrackNetv2/3 (сегментационный детектор мяча, SOTA для тенниса/бадминтона);
    #   путь к своим весам — env BT_MODEL_PATH (см. detector.py: поддержка .pt/.onnx).
    model_path: str = os.getenv("BT_MODEL_PATH", "app/yolo11m.pt")
    # Fallback: если основная модель недоступна (файл удалён/не скачан) —
    # пробуем nano; обе модели используют те же классы COCO.
    model_fallback_path: str = os.getenv("BT_MODEL_FALLBACK_PATH", "app/yolo11n.pt")
    device: str = os.getenv("BT_DEVICE", "cpu")          # "cpu" / "0" (GPU)
    conf_threshold: float = float(os.getenv("BT_CONF", "0.15"))
    # Порог доверия для мяча: yolo11m даёт на реальном мяче уверенно 0.3+,
    # поэтому порог поднят с 0.18 до 0.25 — это отсекает текстуры поля/тени,
    # которые срывали трекер на «неестественные траектории». На мелком/быстром
    # мяче можно опустить через BT_BALL_CONF=0.20.
    ball_conf_threshold: float = float(os.getenv("BT_BALL_CONF", "0.25"))
    class_ids: tuple[int, ...] = (32,)                    # COCO: 32 = sports ball

    # --- Трекинг (SORT / IoU-сопоставление) ---------------------------------
    # max_age уменьшен: дольше держать трек без наблюдений — больше шансов
    # «улететь» по инерции Kalman-предсказания после срыва.
    max_age: int = 5               # кадров держать трек без детекций (окклюзии)
    min_hits: int = 2              # мин. подтверждений, чтобы трек считался валидным
    iou_threshold: float = 0.25    # порог IoU для венгерского сопоставления
    # Жёсткий физический gate: прыжок центра мяча между кадрами не должен
    # превышать разумную долю высоты кадра (на реальных играх мяч летит
    # быстрее, но и FPS там выше; параметр масштабируется с fps).
    max_jump_frac: float = float(os.getenv("BT_MAX_JUMP_FRAC", "0.18"))
    # Множитель сигмы предсказания Kalman в адаптивном шлюзе сопоставления
    # (Mahalanobis-gate): пара трек-детекция допускается, если отклонение
    # центра <= gate_k * sigma_pred И <= hard-limit (max_jump_frac*min(W,H)).
    # Раньше шлюз был чистым IoU>=iou_threshold: для маленького мяча это
    # означало практически точное совпадение боксов -> быстрый/ускоренный пас
    # разрывал трек на осколки («мяч не детектится»), а обрывки треков,
    # подхваченные ложными детекциями, «летали по экрану».
    gate_k: float = float(os.getenv("BT_GATE_K", "3.0"))
    # При очень большой неопределённости предсказания (долгая окклюзия) пары
    # с крошечным IoU всё ещё допускаются, если отклонение в пределах gate_k*sigma
    gate_iou_exempt_cost: float = 0.0
    # Склейка фрагментов: новый трек, рождённый в пределах N кадров и M пикселей
    # от недавно умершего, наследует его историю (один полёт = один трек)
    merge_max_gap: int = int(os.getenv("BT_MERGE_MAX_GAP", "6"))

    # --- Детекция событий паса ------------------------------------------------
    release_speed_px: float = 6.0   # мгновенная скорость (px/кадр) для "отрыва"
    catch_radius_scale: float = 3.0 # радиус "приёмки" относительно диагонали бокса игрока
    contact_expand: float = 0.45    # расширение бокса игрока для зоны контакта мяч-игрок
    min_flight_frames: int = 4      # минимальная длительность полёта в кадрах
    # Мяч отслеживается ТОЛЬКО во время полёта (pas-by-pass), а не весь ролик:
    # сегмент регистрируется как пас, только если перед ним зафиксирован
    # контакт с игроком (release) и завершается приёмкой/концом видео.
    require_release_contact: bool = os.getenv("BT_REQUIRE_CONTACT", "1") == "1"
    contact_streak: int = 2         # сколько подряд кадров мяч должен быть «в руке»,
                                    # чтобы засчитать владение (защита от одиночных
                                    # ложных срабатываний near-player)
    no_person_reset: int = 10       # если N кадров вообще нет ни одного человека —
                                    # сбрасываем состояние владения (смена эпизода)

    # --- Физика полёта ---------------------------------------------------------
    # Гравитация для ФИЗИЧЕСКОЙ модели (px/s^2): g_px_s2 = gravity_ratio*fps^2.
    # Используется только fit_physics_trajectory (метрики ToF/apex/дальность).
    gravity_ratio: float = float(os.getenv("BT_GRAVITY_RATIO", "0.55"))
    # Гравитация для Kalman-модели трекера, px/frame^2 — ЭМПИРИЧЕСКАЯ величина.
    # КРИТИЧНО: раньше в трекер передавался gravity_ratio напрямую (0.55), а в
    # pipeline вызывался set_physics(settings.gravity_ratio) вместо
    # ratio*fps*fps/fps^2... фактически в модель подставлялось значение из
    # размерности «долей кадра», не согласованное с масштабом видео: трекер
    # либо недо-ускорялся (рвал пас на осколки — «мяч не детектится»), либо
    # пере-ускорялся («catch» улетал на потолок при отрисовке). Значение по
    # умолчанию подобрано так, чтобы за ~20 кадров вертикальная составляющая
    # скорости росла на 0.4*H px (типичный бросок через половину кадра).
    # Для конкретного вида спорта калибруется env BT_KALMAN_G_PX_F2.
    kalman_gravity_px_f2: float = float(os.getenv("BT_KALMAN_G_PX_F2", "1.1"))
    drag_coefficient: float = 0.02     # безразмерный коэффициент сопротивления воздуха
    use_physics_fit: bool = True       # подгонять баллистическую модель к трек-данным

    # --- VballNet (детекция мяча) + детекция пасов по траектории --------------
    # Модель из https://github.com/asigatchov/fast-volleyball-tracking-inference
    vballnet_path: str = os.getenv("BT_VBALLNET_PATH", "models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx")
    heatmap_threshold: float = float(os.getenv("BT_HEATMAP_THRESHOLD", "0.5"))
    # Розыгрыш: разрыв видимости мяча больше этого числа кадров закрывает эпизод.
    rally_gap_frames: int = int(os.getenv("BT_RALLY_GAP", "15"))
    # Параболические участки (свободный полёт): окно фита и допуски.
    par_min_frames: int = int(os.getenv("BT_PAR_MIN_FRAMES", "6"))
    par_max_frames: int = int(os.getenv("BT_PAR_MAX_FRAMES", "45"))
    par_max_rmse_frac: float = float(os.getenv("BT_PAR_RMSE_FRAC", "0.03"))
    # Эмпирическое гравитационное ускорение в px/s^2 = ratio * fps^2
    # (подбирается под ракурс/масштаб съёмки; 0.55 — типовой зал).
    gravity_fit_ratio: float = float(os.getenv("BT_GRAVITY_FIT_RATIO", "0.55"))
    grav_tol_rel: float = float(os.getenv("BT_GRAV_TOL_REL", "0.75"))
    # Минимальное горизонтальное перемещение участка (доля ширины кадра),
    # отличающее настоящий перелёт «к сетке» от дребезга/ведения мяча.
    min_horizontal_disp_frac: float = float(os.getenv("BT_MIN_DISP_FRAC", "0.06"))

    # --- Фильтрация ложных детекций ------------------------------------------
    # Точка p_i выбрасывается как ложная, если она «выпрыгивает» из окрестности
    # обеих соседок, а соседи близки друг к другу:
    #   ||p_i - p_{i-1}|| > eps  И  ||p_i - p_{i+1}|| > eps
    #   И  ||p_{i-1} - p_{i+1}|| <= eps
    # т.е. точка — изолированный выброс на траектории (фоновая текстура, блик).
    # eps по умолчанию ≈ диагонали мяча при максимальном реальном перемещении
    # за кадр (~5 px @ 240 fps для волейбольного мяча 21 см) + удвоенный радиус.
    outlier_eps_px: float = float(os.getenv("BT_OUTLIER_EPS_PX", "0"))  # 0 → авто
    outlier_eps_max_jump_px: float = float(os.getenv("BT_OUTLIER_MAX_JUMP", "5"))
    outlier_eps_ball_diam_mult: float = float(os.getenv("BT_OUTLIER_BALL_MULT", "2"))

    # --- Отображение только части траектории ВЫШЕ сетки ------------------------
    # Высота верхней ленты сетки в волейболе — 243 см (мужчины; женщины 224 см
    # задаются через BT_NET_TOP_HEIGHT_M). Уровень сетки в пикселях вычисляется
    # из масштаба сцены: scale_px_per_m = net_top_px / net_top_height_m. Если
    # уровень не задан явно и автооценка невозможна (площадка/игроки не видны
    # или их геометрия неконсистентна — например, сетки в кадре нет вовсе),
    # ограничение НЕ применяется и рисуются все траектории целиком.
    net_top_px: int = int(os.getenv("BT_NET_TOP_PX", "0"))          # явный уровень, px (0 → авто)
    net_top_height_m: float = float(os.getenv("BT_NET_TOP_HEIGHT_M", "2.43"))
    net_scale_min_px_per_m: float = float(os.getenv("BT_NET_SCALE_MIN", "50.0"))
    net_scale_max_px_per_m: float = float(os.getenv("BT_NET_SCALE_MAX", "800.0"))
    only_above_net: bool = os.getenv("BT_ONLY_ABOVE_NET", "1") == "1"

    # --- Отрисовка траекторий поверх видео -------------------------------------
    render_tracked_video: bool = os.getenv("BT_RENDER_VIDEO", "1") == "1"
    trail_length: int = int(os.getenv("BT_TRAIL_LENGTH", "25"))  # «хвост» за мячом, точек
    render_dir: str = os.getenv("BT_RENDER_DIR", "data/renders")
    # Режим отрисовки (диагностика гипотезы «баллистический фит не находится»):
    #   "passes" — рисовать только траектории найденных пасов (по умолчанию);
    #   "full"   — рисовать ПОЛНУЮ траекторию мяча (весь трек по всем розыгрышам)
    #              ДО поиска параболических участков + все баллистические сег-
    #              менты (и пасы «к сетке», и «от сетки») с их RMSE/grav-метками.
    render_mode: str = os.getenv("BT_RENDER_MODE", "passes")

    # --- Хранилище ------------------------------------------------------------
    db_path: str = os.getenv("BT_DB_PATH", "data/balltime.db")
    upload_dir: str = os.getenv("BT_UPLOAD_DIR", "data/uploads")

    tracknet_weights: str = field(default="")  # заглушка под TrackNet-веса


settings = Settings()
