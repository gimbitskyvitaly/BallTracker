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
    # СКОРОСТНОЙ ШЛЮЗ (trajectory_filter.remove_outliers_velocity) вместо
    # старого геометрического eps-правила: последнее на реальном видео не
    # работало — ложные heatmap-срабатывания приходят цепочками (соседние
    # кадры в одной горячей точке), условие «обе соседки рядом» для них
    # никогда не выполнялось, а уменьшение BT_OUTLIER_EPS_PX лишь резало
    # трек на микро-осколки (треков меньше, выбросы остаются, flights=[]).
    #
    #   1) remove_outliers_velocity: p_i отвергается, если её отклонение от
    #      локальной экстраполяции траектории (LSQ-скорость по последним
    #      BT_GATE_WINDOW точкам) превышает gate_mult * max(|v|*dt, v_min_px)
    #      или мгновенная скорость > speed_max_px_f. Статичная hotspot-ложь
    #      удаляется, быстрый честный полёт сохраняется.
    #   2) split_track_by_jumps: трек рвётся только там, где dist/dt между
    #      соседними оставленными точками > BT_MAX_BALL_SPEED_PX_F
    #      (физически невозможный телепорт = новый розыгрыш), а НЕ на каждом
    #      шаге быстрого полёта, как это делал eps.
    #
    # НАСТРОЙКА:
    #   BT_GATE_MULT — множитель порога (меньше = агрессивнее удаление);
    #   BT_V_MIN_PX  — минимальный допуск смещения, px (защита при v≈0;
    #                  типично ~1.5% высоты кадра);
    #   BT_MAX_BALL_SPEED_PX_F — физический максимум скорости мяча, px/кадр
    #                  (волейбольная атака ~30 м/с при масштабе зала ≈ 80–120
    #                   px/кадр на 1080p30; больше — уже teleport-ложь).
    #   BT_OUTLIER_EPS_PX сохранён как УСТАРЕВШИЙ (не используется новой
    #   логикой) — только для совместимости внешних вызовов/скриптов.
    gate_mult: float = float(os.getenv("BT_GATE_MULT", "3.0"))
    v_min_px: float = float(os.getenv("BT_V_MIN_PX", "16"))
    max_ball_speed_px_f: float = float(os.getenv("BT_MAX_BALL_SPEED_PX_F", "120"))
    gate_window: int = int(os.getenv("BT_GATE_WINDOW", "6"))
    # Минимум подряд идущих несогласованных точек, чтобы удалить цепочку
    # (одиночные шумы/смена направления полёта сохраняются всегда).
    min_streak: int = int(os.getenv("BT_MIN_STREAK", "2"))
    outlier_eps_px: float = float(os.getenv("BT_OUTLIER_EPS_PX", "60"))  # deprecated
    # --- Статичный hotspot-фильтр (основной, trajectory_filter.remove_static_hotspots)
    # На реальном видео ложные срабатывания — ЦЕПОЧКИ в одной точке фона,
    # чередующиеся с мячом; их не брал ни eps-фильтр, ни шлюз по экстраполяции.
    # Критерий: цепочка наблюдений удаляется, если её физически невозможно
    # связать ни с одной соседней цепочкой (требуемая скорость стыковки >
    # BT_MAX_BALL_SPEED_PX_F). Допуски:
    #   BT_STATIC_RADIUS_PX — радиус «стояния на месте» (разброс hotspot'а);
    #   BT_JUMP_TOLERANCE   — относительный допуск порога при разрыве трека.
    static_filter_enabled: bool = os.getenv("BT_STATIC_FILTER", "1") == "1"
    static_cell_px: float = float(os.getenv("BT_STATIC_CELL_PX", "50"))
    static_radius_px: float = float(os.getenv("BT_STATIC_RADIUS_PX", "45"))
    jump_tolerance: float = float(os.getenv("BT_JUMP_TOLERANCE", "0.25"))

    # --- Отображение только части траектории ВЫШЕ сетки ------------------------
    # Высота верхней ленты сетки — 243 см (женщины 224 см: BT_NET_TOP_HEIGHT_M).
    # Уровень в пикселях: BT_NET_TOP_PX > 0 — явная отметка (надёжнее всего);
    # иначе — ФИКСИРОВАННЫЙ типовой уровень трансляции BT_NET_AUTO_DEFAULT_FRAC*H
    # (без «плавающей» автооценки по геометрии — она скакала от видео к видео).
    # Уровень принимается только если трек его осмысляет: есть точки и выше, и
    # ниже (BT_NET_AUTO_CHECK=1); иначе (сетки нет/плохо видна, мало точек) —
    # лимит НЕ применяется, рисуется вся траектория.
    # Применяется в pipeline.analyze_video: estimate_net_level + clip_track_above_net.
    net_top_px: int = int(os.getenv("BT_NET_TOP_PX", "0"))          # явный уровень, px (0 → типовой)
    net_top_height_m: float = float(os.getenv("BT_NET_TOP_HEIGHT_M", "2.43"))
    only_above_net: bool = os.getenv("BT_ONLY_ABOVE_NET", "1") == "1"
    net_auto_default_frac: float = float(os.getenv("BT_NET_AUTO_DEFAULT_FRAC", "0.65"))
    net_auto_check: bool = os.getenv("BT_NET_AUTO_CHECK", "1") == "1"

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
