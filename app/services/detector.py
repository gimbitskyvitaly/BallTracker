"""Детекция мяча и игроков на кадрах видео.

Почему НЕ чистый COCO-YOLO: базовые веса yolo11n/yolo11m (COCO, класс 32
"sports ball") практически не детектируют маленький/быстрый/размытый мяч на
реальных спортивных видео — conf падает ниже любого порога, flights=[].
Это подтверждено замерами: детекторы COCO обучаются на статичных фото, а не
на теле-трансляциях. Поэтому конвейер использует ДВА каскада:

1) Основной — OpenCV CSRT/MOSSE визуальный трекер + цветовой (HSV) поиск
   кандидата в зоне ожидаемого полёта. Работает на любом спорте без обучения,
   именно так исторически решался трекинг мяча до эпохи глубокого обучения;
   устойчив к низким conf нейросетевого детектора.
2) Уточняющий — YOLO (.pt/.onnx): предобученные sports-веса или fine-tuned
   модель (BT_MODEL_PATH). Детекции используются для переинициализации
   визуального трекера и фильтрации ложных цветовых срабатываний.

Если у кастомной модели свои id классов — BT_BALL_CLS / BT_PERSON_CLS
(Roboflow soccer-датасеты часто имеют ball=0; для ONNX без имён это обязательно!).
"""

from __future__ import annotations

import math
import os
import threading

import cv2
import numpy as np

from app.config import settings


class Detection:
    """Одна детекция: [x1, y1, x2, y2], confidence, class_id."""

    __slots__ = ("bbox", "conf", "cls")

    def __init__(self, bbox: np.ndarray, conf: float, cls: int):
        self.bbox = np.asarray(bbox, dtype=float)
        self.conf = float(conf)
        self.cls = int(cls)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def diag(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(np.hypot(w, h))


def _make_visual_tracker():
    """Создание визуального трекера с перебором API всех версий OpenCV.

    Приоритет: CSRT (точней) -> MIL (есть во всех сборках, в т.ч. opencv-python
    4.10+/5.x headless) -> KCF (быстрый fallback). Возвращает объект с методами
    init(frame,(x,y,w,h)) / update(frame)->(ok,(x,y,w,h)).
    """
    for maker in (
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.tracking.TrackerCSRT.create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerMIL_create(),
        lambda: cv2.tracking.TrackerMIL.create(),
        lambda: cv2.legacy.TrackerMIL_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
    ):
        try:
            tr = maker()
            if tr is not None:
                return tr
        except Exception:
            continue
    raise RuntimeError("OpenCV build has no available tracker API")


class BallDetector:
    """Потокобезопасный детектор мяча: CV-трекер (CSRT) + HSV-поиск + YOLO-каскад."""

    BALL_CLS = int(os.getenv("BT_BALL_CLS", "32"))
    PERSON_CLS = int(os.getenv("BT_PERSON_CLS", "0"))

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self._device = device or settings.device
        self._lock = threading.Lock()
        self.model_path = ""
        self._model = None          # ленивая загрузка YOLO
        self._requested_model = model_path or settings.model_path
        # состояние визуального трекинга мяча
        self._tracker = None
        self._tracker_bbox = None   # (x, y, w, h)
        self._lost_frames = 0
        self._ball_hsv_ranges = []  # выученная палитра мяча [(lo, hi), ...]
        self._last_frame = None
        self._prev_small = None    # для motion-prior затравки (seed_ball)
        # CV-fallback игроков (background subtraction): создаётся один раз,
        # модель фона «помнит» сцену между кадрами; сбрасывается в reset().
        from app.services.persons import PersonDetectorCV
        self._person_cv = PersonDetectorCV(person_cls=self.PERSON_CLS)

    def reset(self) -> None:
        """Полный сброс состояния между видео.

        КРИТИЧНО: сервис использует ОДИН синглтон BallDetector для всех
        запросов (app/main.py:get_detector). Без reset() CSRT-трекер и
        выученная HSV-палитра «дотягиваются» со старого видео: трек липнет
        к фантому, детекция мяча не стартует — «мяч не детектится вообще».
        """
        self._tracker = None
        self._tracker_bbox = None
        self._lost_frames = 0
        self._ball_hsv_ranges = []
        self._last_frame = None
        self._prev_small = None
        from app.services.persons import PersonDetectorCV
        self._person_cv = PersonDetectorCV(person_cls=self.PERSON_CLS)

    # ------------------------------------------------------------------ YOLO
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        path = self._requested_model
        if not os.path.exists(path):
            fb = settings.model_fallback_path
            if fb and os.path.exists(fb):
                path = fb
            else:
                path = os.path.basename(path)  # ultralytics скачает сам
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            self.model_path = path
        except Exception:
            self._model = False     # YOLO недоступен — работаем на CV-каскаде
        return self._model

    def _yolo_detect(self, frame: np.ndarray):
        m = self._ensure_model()
        if not m:
            return [], []
        with self._lock:
            results = m.predict(
                frame,
                conf=settings.conf_threshold,
                classes=[self.BALL_CLS, self.PERSON_CLS],
                device=self._device,
                verbose=False,
            )
        balls, persons = [], []
        r = results[0]
        if r.boxes is None:
            return balls, persons
        for box in r.boxes:
            det = Detection(
                bbox=box.xyxy[0].cpu().numpy(),
                conf=float(box.conf[0].cpu().numpy()),
                cls=int(box.cls[0].cpu().numpy()),
            )
            if det.cls == self.BALL_CLS:
                balls.append(det)
            else:
                persons.append(det)
        balls.sort(key=lambda d: -d.conf)
        persons.sort(key=lambda d: -d.conf)
        return balls, persons

    # ------------------------------------------------------- CV fallback: players
    def _cv_persons(self, frame: np.ndarray) -> list[Detection]:
        """Детекция «игроков» без нейросети (fallback-каскад).

        Когда YOLO недоступен/ничего не нашёл, игроками считаются крупные
        связные области, отличающиеся от фона: сегментация по цвету
        (квантованный HSV + connected components) с двумя фильтрами:
          * высота бокса >= person_min_h_frac * H (игроки в кадре — 30-70%
            высоты; отсекает скамейки, щиты, мяч, тени);
          * область НЕ похожа на круг (fill < circle_max_fill) — иначе
            яркий мяч сам становился «игроком», раздувал зону контакта и
            система никогда не фиксировала «отлёт от игрока» (пас не
            детектировался).
        Это нужно для ветки contact/release: без person-детекций владение
        принципиально невозможно подтвердить и пасы теряются на реальных
        видео, где COCO-модель не запускается (CPU-only сборка, кастомные
        веса). Пороги — env BT_MIN_PERSON_H_FRAC / BT_CV_PERSON_MAX_CIRC.
        """
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Квантование цвета: 3 бита на канал (8 уровней) — устойчиво к JPEG/
        # mp4-шуму вокруг краёв объектов. connectedComponents принимает
        # только 8U/8S, поэтому упаковываем (h,s,v) в uint8 и делаем
        # compact-remap меток (0 резервируется под «самый частый» цвет —
        # фон кадра).
        qh = hsv[..., 0] >> 5                            # 0..7
        qs = hsv[..., 1] >> 5
        qv = hsv[..., 2] >> 5
        lab = (qh.astype(np.int32) << 6) | (qs.astype(np.int32) << 3) | qv.astype(np.int32)
        vals, counts = np.unique(lab, return_counts=True)
        bg_val = int(vals[int(np.argmax(counts))])       # доминирующий цвет = фон
        keep = vals != bg_val
        codes = np.zeros(len(vals), np.uint8)
        codes[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint8)
        remap = np.zeros(int(vals.max()) + 1, np.uint8)
        remap[vals] = codes
        lab8 = remap[lab]                                # фон -> 0
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
            lab8, connectivity=4)
        min_h = settings.person_min_h_frac * H
        max_w = 0.6 * W
        out: list[Detection] = []
        for i in range(1, n_lab):                      # 0 — метка «фона» не бывает:
            x, y, w, h, area = stats[i]                # connectedComponents по всей картинке
            if h < min_h or area < 0.001 * W * H or w > max_w or w < 0.2 * h:
                continue
            m = (labels == i)
            ys_, xs_ = np.nonzero(m)
            x1, y1, x2, y2 = xs_.min(), ys_.min(), xs_.max(), ys_.max()
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            fill = area / float(bw * bh)
            # круглый компактный blob (мяч) — не игрок
            if fill > settings.cv_person_max_circle_fill and \
                    0.7 <= bw / max(bh, 1) <= 1.4:
                continue
            conf = float(min(0.9, 0.35 + 3.0 * area / float(W * H)))
            out.append(Detection(np.array([x1, y1, x2 + 1, y2 + 1], float),
                                 conf, self.PERSON_CLS))
        # боксы, полностью лежащие внутри более крупного — дубли: оставляем внешние
        out.sort(key=lambda d: -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        final: list[Detection] = []
        for d in out:
            inside_existing = any(
                d.bbox[0] >= f.bbox[0] and d.bbox[1] >= f.bbox[1] and
                d.bbox[2] <= f.bbox[2] and d.bbox[3] <= f.bbox[3]
                for f in final)
            if not inside_existing:
                final.append(d)
        final.sort(key=lambda dd: -dd.conf)
        return final

    # ------------------------------------------------------------- HSV search
    def _hsv_candidates(self, frame, roi=None, min_area=6.0, max_area_frac=0.01):
        """Кандидаты «похожие на мяч» по выученной HSV-палитре (или яркие круги).

        КРИТИЧНО (баг «мяч не детектится вообще» на реальном видео): при
        невыученной палитре дефолтные диапазоны («белый/оранжевый мяч»)
        совпадают с трибунами, формой игроков и разметкой — первый же
        крупный «круглый» лоскут фона становился «мячом», CSRT захватывал
        его и трек уползал по статичным объектам. Поэтому:
          * размер blob'а должен лежать в полосе [min_size, max_size] от
            высоты кадра (BT_BALL_MIN_SIZE_FRAC / BT_BALL_MAX_SIZE_FRAC);
          * кандидаты сортируются по близости к ожидаемой позиции ROI, а
            не по «круглости» — глобальный поиск без трека возвращает пустой
            список: новый мяч ловит YOLO (или следующий кадр после явной
            инициализации), а не случайное белое пятно на трибуне.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H_img, W_img = hsv.shape[:2]
        ranges = self._ball_hsv_ranges
        if not ranges:
            # Дефолтная палитра применяется ТОЛЬКО к мелким blob'ам (строгий
            # размерный фильтр ниже + запрет глобального поиска) — иначе она
            # сходит с трибун/формы.
            ranges = [
                ((0, 0, 170), (180, 70, 255)),      # белый/светлый мяч
                ((15, 120, 120), (40, 255, 255)),   # оранжевый/жёлтый мяч
            ]
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        if roi is None and not self._ball_hsv_ranges:
            # невыученная палитра + нет ожидаемой позиции: глобальный поиск
            # дефолтными диапазонами запрещён (главный источник фантомов)
            return []
        if roi is not None:
            x1, y1, x2, y2 = roi
            keep = np.zeros_like(mask)
            keep[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
            mask &= (keep > 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        max_area = H_img * W_img * max_area_frac
        min_d = float(os.getenv("BT_BALL_MIN_SIZE_FRAC", "0.006")) * H_img
        max_d = float(os.getenv("BT_BALL_MAX_SIZE_FRAC", "0.06")) * H_img
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            circ = a / (np.pi * rad * rad + 1e-6)          # ~1 для круга
            if circ < 0.55:
                continue
            d = 2 * rad
            if d < min_d or d > max_d:                     # размерная полоса мяча
                continue
            det = Detection(
                bbox=np.array([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]),
                conf=float(min(0.99, 0.4 + 0.5 * circ)),
                cls=self.BALL_CLS,
            )
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                mx, my = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
                dist = ((det.center[0] - mx) ** 2 + (det.center[1] - my) ** 2) ** 0.5
                det.conf = float(max(0.3, det.conf - dist / (4.0 * max(rx2 - rx1, 1))))
            out.append(det)
        # приоритет: близость к ожидаемой позиции, затем круглость
        out.sort(key=lambda dd: (-dd.conf, -dd.bbox[2]))
        return out

    def _learn_palette(self, frame, bbox_xyxy):
        """Выучить HSV-палитру мяча по первому надёжному боксу (YOLO/вручную)."""
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return
        patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h = patch[..., 0].ravel().astype(np.int16)
        s = patch[..., 1].ravel()
        v = patch[..., 2].ravel()
        # разрыв вокруг красной границы hue (0/180) обрабатываем циклически
        hmin, hmax = int(np.percentile(h, 10)), int(np.percentile(h, 90))
        if hmax - hmin > 90:  # красный мяч: диапазон через 180→0
            self._ball_hsv_ranges = [
                ((max(0, hmin - 8), 40, 60), (180, 255, 255)),
                ((0, 40, 60), (min(179, hmax - 180 + 8), 255, 255)),
            ]
        else:
            self._ball_hsv_ranges = [
                ((max(0, hmin - 8), max(0, int(np.percentile(s, 20)) - 30), 60),
                 (min(179, hmax + 8), 255, 255))
            ]

    # ---------------------------------------------------------------- public
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """Возвращает (мячи, игроки) для одного кадра BGR.

        Мяч: сначала сопровождение CSRT-трекером; при его потере — HSV-поиск
        в окрестности предсказанной позиции; YOLO-детекции (если есть) служат
        триггером переинициализации и калибровки палитры. Игроки — из YOLO
        (при недоступности модели возвращаются пустым списком; события
        release/catch тогда выводятся по геометрии траектории в pipeline).
        """
        self._last_frame = frame
        yolo_balls, persons = self._yolo_detect(frame)
        # Fallback-каскад игроков: если YOLO недоступен (ultralytics не
        # установлен / веса не скачаны) или не нашёл ни одного человека,
        # боксы игроков достаем классическим CV (фон/передний план). Без
        # person-детекций ветка contact/release в pipeline бессильна и
        # пасы не детектятся вовсе (баг «ни реального, ни ложного паса»).
        if not persons:
            # CV-fallback: MOG2 background subtraction (app/services/persons.py).
            # Старая реализация (_cv_persons, квантованный HSV + connected
            # components) на реальных видео с неоднородным фоном не находила
            # НИ ОДНОго игрока (persons==[] всегда) — contact/release ветка
            # pipeline молчала, пасы не детектировались.
            persons = self._person_cv.update(frame)
        strong_yolo = [d for d in yolo_balls if d.conf >= settings.ball_conf_threshold]
        H, W = frame.shape[:2]

        # 1) пробуем вести существующий трекер
        if self._tracker is not None and self._tracker_bbox is not None:
            ok, bb = self._tracker.update(frame)
            if ok:
                x, y, w, h = [float(t) for t in bb]
                if w > 2 and h > 2 and w * h < 0.05 * W * H:
                    self._tracker_bbox = (x, y, w, h)
                    self._lost_frames = 0
                    det = Detection(np.array([x, y, x + w, y + h]), conf=0.9, cls=self.BALL_CLS)
                    # редкая сверка с YOLO: если тот уверенно видит мяч далеко
                    # от трекера — верим ему (переинициализация)
                    if strong_yolo:
                        dx = abs(strong_yolo[0].center[0] - det.center[0])
                        dy = abs(strong_yolo[0].center[1] - det.center[1])
                        if max(dx, dy) > 3 * max(w, h):
                            self._reinit_tracker(strong_yolo[0].bbox)
                            self._learn_palette(frame, strong_yolo[0].bbox)
                            return strong_yolo[:1], persons
                    return [det], persons
                ok = False
            self._lost_frames += 1
            if self._lost_frames > settings.max_age * 2:
                self._tracker = None
                self._tracker_bbox = None

        # 2) candidate: HSV-поиск в ROI вокруг последней позиции.
        #    Глобальный поиск без выученной палитры запрещён (см.
        #    _hsv_candidates): иначе «мячом» становился первый белый лоскут
        #    трибуны, CSRT захватывал его — трек уползал по статичным
        #    объектам, настоящий мяч не детектился НИ ОДНАЖДЫ (баг).
        roi = None
        if self._tracker_bbox is not None:
            x, y, w, h = self._tracker_bbox
            pad = max(4 * w, 4 * h, 60) + 12 * self._lost_frames
            roi = (int(x - pad), int(y - pad), int(x + w + pad), int(y + h + pad))
        elif self._ball_hsv_ranges:
            # палитра выучена, но трек потерян окончательно — ищем новый
            # мяч глобально (размерный фильтр всё ещё защищает от фона)
            roi = (0, 0, W, H)
        cands = self._hsv_candidates(frame, roi=roi)
        if not cands and strong_yolo:
            cands = strong_yolo[:1]
        if cands:
            best = cands[0]
            self._reinit_tracker(best.bbox)
            # Самокалибровка палитры под цвет мяча: по первому надёжному
            # боксу (уверенный YOLO или первый найденный круглый кандидат).
            if not self._ball_hsv_ranges:
                if best.conf >= 0.4:
                    self._learn_palette(frame, best.bbox)
            return [best], persons

        # 3) ничего не ведём: пробуем затравку по движению (motion prior),
        #    затем отдаём то, что видит YOLO (порог конф. применён).
        if not strong_yolo and self._tracker_bbox is None:
            self.seed_ball(frame)
            if self._tracker is not None:
                return [Detection(np.array([self._tracker_bbox[0],
                                            self._tracker_bbox[1],
                                            self._tracker_bbox[0] + self._tracker_bbox[2],
                                            self._tracker_bbox[1] + self._tracker_bbox[3]]),
                                  conf=0.5, cls=self.BALL_CLS)], persons
        return strong_yolo, persons

    def seed_ball(self, frame: np.ndarray):
        """Авто-затравка трека мяча по статистике движения (motion prior).

        Классический приём sports-analytics: там, где между кадрами есть
        устойчивое движение и при этом в кадре присутствуют компактные
        «круглые» blob'ы подходящего размера — с наибольшей вероятностью
        находится МЯЧ (игроки дают большие blob'ы, трибуны/фон — нулевое
        движение). Это единственный надёжный способ поймать мяч на старте,
        когда ни YOLO (COCO его не видит), ни HSV-палитра (выучить нечего)
        не работают; без затравки трек никогда не стартует — «мяч не
        детектится вообще» (баг на реальных видео).
        """
        if self._prev_small is None:
            self._prev_small = cv2.resize(frame, None, fx=0.25, fy=0.25)
            return False
        small = cv2.resize(frame, None, fx=0.25, fy=0.25)
        md = cv2.absdiff(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(self._prev_small, cv2.COLOR_BGR2GRAY))
        self._prev_small = small
        mb = cv2.GaussianBlur(md, (0, 0), 3)
        motion = (mb > int(os.getenv("BT_SEED_MOTION_THR", "18"))).astype(np.uint8) * 255
        H_img, W_img = frame.shape[:2]
        min_d = float(os.getenv("BT_BALL_MIN_SIZE_FRAC", "0.006")) * H_img
        max_d = float(os.getenv("BT_BALL_MAX_SIZE_FRAC", "0.06")) * H_img
        best = None
        for sc in (1.0, 2.0, 4.0):                     # крупные blob'ы тоже двигаются
            cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c) * sc * sc       # площадь в полных пикселях
                if a < 0.25 * math.pi * (min_d / 2) ** 2:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                d = max(w, h) * sc
                if d > 0.35 * H_img or d > max_d * 6:  # игроки/зоны — не мяч
                    continue
                (cx, cy), rad = cv2.minEnclosingCircle(c)
                circ = a / (np.pi * (rad * sc) ** 2 + 1e-6)
                if circ < 0.35:
                    continue
                score = circ * a
                if best is None or score > best[0]:
                    best = (score, cx * sc, cy * sc, d)
        if best is None:
            return False
        cx, cy, d = best[1], best[2], best[3]
        self._reinit_tracker(np.array([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2]))
        self._learn_palette(frame, [cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2])
        return True

    def _reinit_tracker(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        bb = (x1, y1, max(2.0, x2 - x1), max(2.0, y2 - y1))
        self._tracker = _make_visual_tracker()
        if hasattr(self, "_last_frame") and self._last_frame is not None:
            self._tracker.init(self._last_frame, tuple(int(v) for v in bb))
        self._tracker_bbox = bb
        self._lost_frames = 0

    def relock_ball(self, frame: np.ndarray, center: tuple[float, float],
                    radius_px: float) -> None:
        """Переинициализация трекера мяча по позиции из Kalman-трека.

        Вызывается из pipeline при «сходе» CSRT-трека с мяча (Kalman-
        предсказание уехало далеко от свежей детекции). Без этого потерянный
        трекер продолжает «вести» статичный объект, а настоящий летящий мяч
        не детектится вовсе (баг: полёт есть, событий нет — трек разрывался
        в момент релиза)."""
        with self._lock:
            self._last_frame = frame.copy()
            self._reinit_tracker(np.array([center[0] - radius_px,
                                           center[1] - radius_px,
                                           center[0] + radius_px,
                                           center[1] + radius_px]))

    def detect_batch_init(self, frame, bbox_xyxy):
        """Явная инициализация трека (для тестов/ручной разметки стартового бокса)."""
        self._last_frame = frame.copy()
        self._reinit_tracker(bbox_xyxy)
        self._learn_palette(frame, bbox_xyxy)
