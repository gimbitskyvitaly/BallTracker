"""Пайплайн анализа видео: VballNet-детекция мяча → траектория → розыгрыши → пасы.

Раньше здесь стоял COCO-YOLO (класс 32 "sports ball") + SORT + эвристика
пасов по «контакту с боксом игрока». Проблемы, зафиксированные на реальных
волейбольных видео:
  * COCO-YOLO почти не детектит маленький/размытый мяч в зале → пустые
    траектории («мяч детектится плохо, часто вообще не детектится»);
  * «пас» определялся как полёт между контактами с людьми → любой удар/подача/
    отбой выглядел пасом, а настоящий пас терялся при пропусках детекции
    («пас не отличается от не паса»).

Новая схема:
  1. Детекция мяча — VballNet (ONNX, окно 9 полутоновых кадров, heatmap) из
     https://github.com/asigatchov/fast-volleyball-tracking-inference
     (~87% видимости мяча вместо ~0% у COCO-YOLO).
  2. Трек строится прямо по точкам детекции: одиночные выпадения закрываются
     линейной интерполяцией (разрыв <= rally_gap_frames), поэтому параболы
     не рвутся на осколки.
  3. Розыгрыши и пасы — app.services.pass_detector: внутри розыгрышей ищутся
     параболические (баллистические) участки, летящие ВЛЕВО/ВПРАВО К сетке;
     только они считаются пасами. Направление «от сетки» (атака/отбой) —
     не пас.
  4. Для каждого паса метрики (ToF, apex, дальность, v0) оцениваются физмо-
     делью app.services.physics по точкам траектории участка.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings
from app.services.pass_detector import (Rally, detect_passes,
                                         raw_fit_diagnostics)
from app.services.physics import estimate_flight, set_gravity_px
from app.services.trajectory_filter import (BallPoint, NetLevel,
                                            auto_court_geometry,
                                            clip_track_above_net,
                                            default_outlier_eps,
                                            estimate_net_level,
                                            remove_outliers)
from app.services.vballnet import VballNetDetector


@dataclass
class PassEvent:
    release_frame: int
    catch_frame: int
    direction: str                      # "left" | "right" — к сетке слева/справа
    to_net_ratio: float                 # финиш/старт расстояния до сетки (<1 = сблизился)
    rmse_px: float
    points_px: list[tuple[int, float, float]] = field(default_factory=list)
    flight: object = None               # FlightEstimate | None
    rally_index: int = -1


@dataclass
class VideoAnalysis:
    fps: float
    width: int
    height: int
    n_frames: int
    ball_track_points: list[dict] = field(default_factory=list)   # [{frame,x,y}]
    rallies: list[Rally] = field(default_factory=list)
    passes: list[PassEvent] = field(default_factory=list)
    ball_radius_px: float = 6.0
    detection_rate: float = 0.0         # доля кадров с детекцией мяча
    # статистика фильтрации ложных детекций (remove_outliers): eps, число
    # выброшенных точек, сами точки (для отладки порогов BT_OUTLIER_*)
    removed_outliers: dict = field(default_factory=dict)
    # уровень верхней ленты сетки (243 см) и результат клипа «только выше
    # сетки»; net=None → уровень не определён (сетки нет/плохо видна), лимит
    # НЕ применялся
    net_level: NetLevel | None = None
    net_clip_stats: dict = field(default_factory=dict)
    # диагностика режима render_mode="full": баллистические LSQ-фитЫ БЕЗ
    # фильтров (rmse/gravity/disp) по каждому розыгрышу + границы розыгрышей;
    # None в обычном режиме "passes"
    fit_diagnostics: dict | None = None


_detector_singleton: VballNetDetector | None = None


def get_vballnet_detector() -> VballNetDetector:
    """Ленивый singleton: загрузка ONNX-сессии дорогая."""
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = VballNetDetector(settings.vballnet_path,
                                               threshold=settings.heatmap_threshold)
    return _detector_singleton


def _interp_track(raw: list[tuple[int, float, float]], gap_max: int
                  ) -> list[tuple[int, float, float]]:
    """Сырые детекции → плотный трек: разрывы <= gap_max закрываются интерполяцией.

    Это замена SORT/Kalman: VballNet даёт стабильные точки, а интерполяция
    убирает одиночные пропуски, не придумывая фантомных полётов (долгие
    разрывы остаются границами розыгрышей)."""
    if not raw:
        return []
    pts = sorted(raw, key=lambda p: p[0])
    out = [pts[0]]
    for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
        gap = f1 - f0
        if 1 < gap <= gap_max:
            for k in range(1, gap):
                t = k / gap
                out.append((f0 + k, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append((f1, x1, y1))
    return out


def analyze_video(path: str, detector: VballNetDetector | None = None,
                  max_frames: int | None = None,
                  progress_cb=None) -> VideoAnalysis:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = detector or get_vballnet_detector()

    # --- проход 1: детекция мяча окном VballNet ------------------------------
    raw: list[tuple[int, float, float]] = []
    frames_read = 0
    detected = 0
    chunk: list[np.ndarray] = []
    CHUNK = 60  # батч кадров для скользящего 9-окна модели

    def _flush(chunk: list[np.ndarray], start_idx: int):
        nonlocal detected
        results = det.feed(chunk, start_idx=start_idx)
        for r in results:
            if r is not None:
                raw.append((r.frame, r.x, r.y))
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        chunk.append(frame)
        frames_read += 1
        if len(chunk) >= CHUNK:
            _flush(chunk, frames_read - len(chunk))
            chunk = []
        if max_frames and frames_read >= max_frames:
            break
    if chunk:
        _flush(chunk, frames_read - len(chunk))
    cap.release()

    frames_read = max(frames_read, 1)
    detected = len(raw)

    # --- проход 2: трек → фильтрация ложных детекций → клип выше сетки --------
    # Порядок важен: сначала убираем изолированные выбросы heatmap-детекции
    # (блики/текстуры «телепортируют» трек и рвут баллистические фиты), затем
    # отбрасываем точки НИЖЕ верхней ленты сетки (243 см) — нас интересуют
    # только части траекторий ВЫШЕ сетки. Оба шага — app.services.trajectory_filter;
    # раньше функции были написаны, но в пайплайн не подключены (баг).
    ball_diam_px = max(8.0, height * 0.035)   # диаметр ≈ 3.5% высоты кадра

    track_all = _interp_track(raw, gap_max=settings.rally_gap_frames)

    eps = settings.outlier_eps_px
    if not eps or eps <= 0:                   # 0 → авто из config-множителей
        eps = default_outlier_eps(fps, height,
                                  max_speed_px_s=settings.outlier_eps_max_speed_px_s,
                                  ball_diam_mult=settings.outlier_eps_ball_diam_mult,
                                  ball_diam_px=ball_diam_px)
    track, dropped = remove_outliers(track_all, eps)
    # интерполяция разрывов ПОВТОРНО: remove_outliers вырывает точки-выбросы,
    # на их месте остаются дыры (разрыв ровно в 1 кадр), из-за которых
    # find_parabolic_segments отказывался брать окно (запрет diff > gap_break).
    track = _interp_track(track, gap_max=settings.rally_gap_frames)

    net: NetLevel | None = None
    if settings.only_above_net:
        # геометрию/масштаб оцениваем по ПОЛНОМУ треку (до клипа): нижняя
        # огибающая y — это как раз точки «у земли», которые клип как раз и
        # отсекает; после клипа масштаб оценить было бы невозможно.
        geo = auto_court_geometry(
            track_all, height,
            ground_frac_lo=settings.net_auto_ground_lo_frac,
            ground_frac_hi=settings.net_auto_ground_hi_frac,
            player_height_frac=settings.net_auto_player_height_m)
        net = estimate_net_level(height, width, geo,
                                 net_height_m=settings.net_top_height_m,
                                 manual_y_px=float(settings.net_top_px),
                                 scale_min=settings.net_scale_min_px_per_m,
                                 scale_max=settings.net_scale_max_px_per_m)
    track, clip_stats = clip_track_above_net(track, net)
    # финальная чистка: интерполяция после клипа соединяет куски через дыру
    # «под сеткой» прямыми отрезками вниз; убираем такие точки ещё раз (клип
    # идемпотентен) — в анализ/рендер уходит трек СТРОГО выше уровня сетки.
    track, _clip2 = clip_track_above_net(track, net)
    clip_stats["removed_below"] += _clip2["removed_below"]

    # --- проход 3: розыгрыши → параболические пасы -----------------------------
    set_gravity_px(fps, settings.gravity_fit_ratio)  # размерность px/s^2 для фита
    rallies, pass_segs = detect_passes(
        track, fps, width, height,
        rally_gap_frames=settings.rally_gap_frames,
        par_min_frames=settings.par_min_frames,
        par_max_frames=settings.par_max_frames,
        par_max_rmse_frac=settings.par_max_rmse_frac,
        gravity_px_s2=fps * fps * settings.gravity_fit_ratio,
        grav_tol_rel=settings.grav_tol_rel,
        min_flight_frames=settings.min_flight_frames,
        min_horizontal_disp_frac=settings.min_horizontal_disp_frac,
    )

    analysis = VideoAnalysis(fps=fps, width=width, height=height,
                             n_frames=frames_read)
    analysis.detection_rate = round(detected / frames_read, 3)
    analysis.ball_track_points = [{"frame": int(f), "x": round(x, 1), "y": round(y, 1)}
                                  for f, x, y in track]
    analysis.rallies = rallies
    analysis.removed_outliers = {
        "eps_px": round(eps, 1),
        "n_removed": len(dropped),
        "n_track_before": len(track_all),
        "points": [[int(f), round(x, 1), round(y, 1)] for f, x, y in dropped][:200],
    }
    analysis.net_level = net
    analysis.net_clip_stats = clip_stats
    analysis.ball_radius_px = ball_diam_px / 2.0

    # привязка пасов к розыгрышам
    def rally_of(fr: int) -> int:
        for i, r in enumerate(rallies):
            if r.start_frame <= fr <= r.end_frame:
                return i
        return -1

    for ps in pass_segs:
        pts = [(int(f), float(x), float(y)) for f, x, y in ps.segment.points]
        flight = None
        if settings.use_physics_fit and len(pts) >= 4:
            flight = estimate_flight(pts, fps, ball_diam_px,
                                     settings.drag_coefficient, use_physics=True)
        analysis.passes.append(PassEvent(
            release_frame=ps.release_frame, catch_frame=ps.catch_frame,
            direction=ps.direction, to_net_ratio=round(ps.segment.to_net_ratio, 3),
            rmse_px=round(ps.segment.rmse_px, 2), points_px=pts, flight=flight,
            rally_index=rally_of(ps.release_frame)))

    # --- диагностика для render_mode="full": фиты БЕЗ фильтров ---------------
    if settings.render_mode == "full":
        from app.services.pass_detector import split_rallies
        diag_rallies = []
        all_fits: list[dict] = []
        for ri, rp in enumerate(split_rallies(track, settings.rally_gap_frames)):
            fits = raw_fit_diagnostics(
                rp, fps,
                min_frames=settings.par_min_frames,
                max_frames=settings.par_max_frames,
                gap_break=settings.rally_gap_frames,
                gravity_px_s2=fps * fps * settings.gravity_fit_ratio,
                grav_tol_rel=settings.grav_tol_rel,
                max_rmse_frac=settings.par_max_rmse_frac,
                frame_small=min(width, height), frame_width=width,
                min_horizontal_disp_frac=settings.min_horizontal_disp_frac)
            for k, ft in enumerate(fits):
                ft["rally_index"] = ri
                ft["fit_index"] = len(all_fits)
                all_fits.append(ft)
            diag_rallies.append({"index": ri,
                                 "start_frame": rp[0][0], "end_frame": rp[-1][0],
                                 "n_points": len(rp),
                                 "n_raw_fits": len(fits),
                                 "n_would_pass": sum(f["would_pass_filters"]
                                                     for f in fits)})
        n_would = sum(f["would_pass_filters"] for f in all_fits)
        analysis.fit_diagnostics = {
            "render_mode": "full",
            "note": ("LSQ-фиты баллистики без фильтров rmse/gravity/disp; "
                     "failed_checks показывает, какой фильтр отсек бы участок"),
            "total_raw_fits": len(all_fits),
            "total_would_pass": n_would,
            "coverage_px": _fit_coverage_px(all_fits, track),
            "rallies": diag_rallies,
            "fits": [{**f, "points": [[int(a), round(b, 1), round(c, 1)]
                                      for a, b, c in f["points"]]}
                     for f in all_fits],
        }
    if progress_cb:
        progress_cb(frames_read, total)
    return analysis


def _fit_coverage_px(fits: list[dict], track: list[tuple[int, float, float]]) -> dict:
    """Какую часть трека мяча «объясняют» баллистические фиты без фильтров.

    Если coverage низкий — гипотеза верна: траектория между контактами вообще
    не описывается кусками парабол (шум детекции/окно фита), и дело не в по-
    рогах фильтрации пасов."""
    covered = set()
    for f in fits:
        for p in f["points"]:
            covered.add(int(p[0]))
    frames_in_track = {int(t[0]) for t in track}
    cov = len(covered & frames_in_track) / max(len(frames_in_track), 1)
    return {"track_frames": len(frames_in_track),
            "ballistic_frames": len(covered & frames_in_track),
            "coverage": round(cov, 3)}


def save_analysis(analysis: "VideoAnalysis", path: str) -> None:
    """Сериализация результата анализа (чтобы ререндер не гонял модель заново)."""
    with open(path, "wb") as f:
        pickle.dump(analysis, f)


def load_analysis(path: str) -> "VideoAnalysis":
    with open(path, "rb") as f:
        return pickle.load(f)
