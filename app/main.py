"""FastAPI-сервис «BallTrack» — аналог BallTime™.

Эндпоинты:
  POST /api/v1/videos            — загрузить видео, создать job, запустить анализ
  GET  /api/v1/jobs              — список задач
  GET  /api/v1/jobs/{id}         — статус задачи
  GET  /api/v1/passes/{job_id}   — все пасы (time-of-flight и метрики)
  GET  /api/v1/trajectories/{job_id} — точки траекторий мяча для графиков
  GET  /health                   — проверка живости

Анализ выполняется в фоновом воркере (пул потоков), статус — в SQLite.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.services import storage
from app.services.pipeline import (analyze_video, get_vballnet_detector,
                                   load_analysis, save_analysis)
from app.services.renderer import render_tracked_video

app = FastAPI(title="BallTrack — pass trajectory & time-of-flight", version="1.0.0")

_executor = ThreadPoolExecutor(max_workers=2)


def get_detector():
    """Ленивый singleton VballNet-детектора (загрузка ONNX дорогая)."""
    return get_vballnet_detector()


class JobOut(BaseModel):
    id: str
    status: str
    fps: float | None
    width: int | None
    height: int | None
    n_frames: int | None
    error: str | None
    created_at: str
    finished_at: str | None
    has_result_video: bool = False


def _job_out(j: dict) -> JobOut:
    j = dict(j)
    j["has_result_video"] = bool(j.pop("result_video_path", None))
    return JobOut(**j)


def _render_for_job(jid: str, path: str, force: bool = False,
                    mode: str | None = None) -> str:
    """Видео с траекториями: кэш в БД или повторный анализ при необходимости.

    mode: "passes"|"full" (см. settings.render_mode); для "full" диагностиче-
    ские фиты считаются внутри analyze_video, поэтому закэшированный analysis
    без них переанализируется заново."""
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    eff_mode = (mode or settings.render_mode or "passes").lower()
    cached = job.get("result_video_path")
    if (cached and os.path.exists(cached) and not force
            and eff_mode == "passes"):
        return cached
    dst_dir = os.path.join(settings.render_dir, jid)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"tracked_{eff_mode}.mp4")
    apath = os.path.join(dst_dir, "analysis.pkl")
    if eff_mode == "full":
        apath = os.path.join(dst_dir, "analysis_full.pkl")
    if os.path.exists(apath) and not force:
        analysis = load_analysis(apath)      # без повторного прогона модели
        if eff_mode == "full" and not analysis.fit_diagnostics:
            analysis = analyze_video(path, detector=get_detector())
            save_analysis(analysis, apath)
    else:
        analysis = analyze_video(path, detector=get_detector())
        save_analysis(analysis, apath)
    render_tracked_video(path, dst, analysis, mode=eff_mode)
    storage.set_result_video(jid, dst)
    return dst


def _run_job(jid: str, path: str) -> None:
    storage.set_status(jid, "running")
    try:
        analysis = analyze_video(path, detector=get_detector())
        passes = []
        for p in analysis.passes:
            d = {
                "release_frame": p.release_frame,
                "catch_frame": p.catch_frame,
                "passer_bbox": None,
                "catcher_bbox": None,
                "direction": p.direction,
                "rally_index": p.rally_index,
                "to_net_ratio": p.to_net_ratio,
                "rmse_px": p.rmse_px,
                "points_px": [[int(f), round(x, 1), round(y, 1)]
                              for f, x, y in p.points_px],
            }
            if p.flight:
                d.update({
                    "time_of_flight_s": p.flight.time_of_flight_s,
                    "apex_height_m": p.flight.apex_height_m,
                    "distance_m": p.flight.distance_m,
                    "initial_speed_mps": p.flight.initial_speed_mps,
                    "peak_speed_mps": p.flight.peak_speed_mps,
                    "method": p.flight.method,
                    "trajectory": p.flight.trajectory,
                })
            else:
                d.update({"time_of_flight_s": None, "method": "none", "trajectory": [],
                          "apex_height_m": None, "distance_m": None,
                          "initial_speed_mps": None, "peak_speed_mps": None})
            passes.append(d)
        storage.save_passes(jid, passes)
        storage.set_analysis_meta(
            jid, detection_rate=analysis.detection_rate,
            n_rallies=len(analysis.rallies),
            rallies=[{"index": i, "start_frame": r.start_frame,
                      "end_frame": r.end_frame,
                      "phases": [{"start": a, "end": b, "side": s}
                                 for a, b, s in r.phases]}
                     for i, r in enumerate(analysis.rallies)])
        # сразу формируем видео с отрисованными траекториями поверх исходного
        if settings.render_tracked_video:
            try:
                rdir = os.path.join(settings.render_dir, jid)
                os.makedirs(rdir, exist_ok=True)
                save_analysis(analysis, os.path.join(rdir, "analysis.pkl"))
                if settings.render_mode == "full":
                    # fit_diagnostics уже посчитаны в analyze_video (режим full)
                    save_analysis(analysis, os.path.join(rdir, "analysis_full.pkl"))
                dst = os.path.join(rdir, f"tracked_{settings.render_mode}.mp4")
                render_tracked_video(path, dst, analysis)
                storage.set_result_video(jid, dst)
            except Exception:  # noqa: BLE001 — рендер не валит job
                traceback.print_exc()
        storage.set_status(jid, "done", meta={
            "fps": analysis.fps, "width": analysis.width,
            "height": analysis.height, "n_frames": analysis.n_frames})
        print(f"[job {jid}] frames={analysis.n_frames} ball_detection_rate="
              f"{analysis.detection_rate:.0%} rallies={len(analysis.rallies)} "
              f"passes={len(analysis.passes)}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        storage.set_status(jid, "error", error=str(e))


@app.on_event("startup")
def _startup() -> None:
    storage.init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/v1/videos", response_model=JobOut)
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Нужен video/* файл")
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    jid = storage.create_job("")
    path = os.path.join(settings.upload_dir, f"{jid}{ext}")
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    storage.set_status(jid, "pending")
    with storage._conn() as c:
        c.execute("UPDATE jobs SET video_path=? WHERE id=?", (path, jid))
    _executor.submit(_run_job, jid, path)
    return _job_out(storage.get_job(jid))


@app.get("/api/v1/jobs", response_model=list[JobOut])
def jobs(limit: int = 50):
    return [_job_out(j) for j in storage.list_jobs(limit)]


@app.get("/api/v1/jobs/{jid}", response_model=JobOut)
def job(jid: str):
    j = storage.get_job(jid)
    if not j:
        raise HTTPException(404, "job не найден")
    return _job_out(j)


@app.get("/api/v1/passes/{jid}")
def passes(jid: str):
    if not storage.get_job(jid):
        raise HTTPException(404, "job не найден")
    return storage.get_passes(jid)


@app.get("/api/v1/jobs/{jid}/result-video")
def result_video(jid: str, refresh: bool = Query(False, description="перерендерить заново"),
                 mode: str | None = Query(
                     None,
                     description=("режим отрисовки: 'passes' — только траектории пасов "
                                  "(по умолчанию); 'full' — полная траектория мяча до "
                                  "поиска параболических участков + все баллистические "
                                  "фиты без фильтров (проверка гипотезы о фитах)"))):
    """Скачать видео, аналогичное загруженному, но с траекториями за мячом."""
    if mode is not None and mode.lower() not in ("passes", "full"):
        raise HTTPException(400, "mode должен быть 'passes' или 'full'")
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    path = job.get("video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(409, "исходное видео недоступно")
    if job["status"] == "error":
        raise HTTPException(409, f"анализ завершился ошибкой: {job.get('error')}")
    if job["status"] != "done" and not (job.get("result_video_path")
                                        and os.path.exists(job["result_video_path"])):
        raise HTTPException(409, "анализ ещё не завершён — дождитесь status=done")
    dst = _render_for_job(jid, path, force=refresh, mode=mode)
    return FileResponse(dst, media_type="video/mp4",
                        filename=os.path.basename(dst))


@app.get("/api/v1/jobs/{jid}/debug-fits")
def debug_fits(jid: str):
    """Диагностика гипотезы «баллистический фит не находится» в JSON.

    Возвращает analysis.fit_diagnostics: сырые LSQ-фиты parabola по всему треку
    БЕЗ фильтров rmse/gravity/disp с фактическими значениями каждого критерия и
    полем failed_checks (какой фильтр отсёк бы участок как пас). Если coverage
    низкий — траектория между контактами вообще не ложится на куски парабол;
    если coverage высокий, а passes мало — проблема в порогах/направлении."""
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    path = job.get("video_path")
    if not path or not os.path.exists(path):
        raise HTTPException(409, "исходное видео недоступно")
    rdir = os.path.join(settings.render_dir, jid)
    apath = os.path.join(rdir, "analysis_full.pkl")
    analysis = None
    if os.path.exists(apath):
        analysis = load_analysis(apath)
    if analysis is None or not analysis.fit_diagnostics:
        old_mode = settings.render_mode
        try:
            settings.render_mode = "full"          # включает расчёт диагностики
            analysis = analyze_video(path, detector=get_detector())
            os.makedirs(rdir, exist_ok=True)
            save_analysis(analysis, apath)
        finally:
            settings.render_mode = old_mode
    return {"job_id": jid,
            "detection_rate": analysis.detection_rate,
            "n_passes": len(analysis.passes),
            **analysis.fit_diagnostics}


@app.get("/api/v1/jobs/{jid}/video")
def job_video(jid: str):
    """Отдать готовое видео с траекториями (без повторного рендера)."""
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    p = job.get("result_video_path")
    if not p or not os.path.exists(p):
        raise HTTPException(409, "видео с траекториями ещё не сформировано")
    return FileResponse(p, media_type="video/mp4", filename=f"{jid}_tracked.mp4")


@app.post("/api/v1/jobs/{jid}/render")
def render_now(jid: str, mode: str | None = Query(
        None, description="'passes' — только пасы; 'full' — полная траектория "
                          "+ фиты без фильтров")):
    """Принудительно (пере)сформировать видео с траекториями и вернуть путь."""
    if mode is not None and mode.lower() not in ("passes", "full"):
        raise HTTPException(400, "mode должен быть 'passes' или 'full'")
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    if job["status"] != "done":
        raise HTTPException(409, "дождитесь status=done")
    dst = _render_for_job(jid, job["video_path"], force=True, mode=mode)
    return {"job_id": jid, "result_video_path": dst}


@app.get("/api/v1/trajectories/{jid}")
def trajectories(jid: str):
    """Точки траекторий по каждому пасу (px + метры физмодели)."""
    if not storage.get_job(jid):
        raise HTTPException(404, "job не найден")
    out = []
    for p in storage.get_passes(jid):
        out.append({
            "release_frame": p["release_frame"],
            "catch_frame": p["catch_frame"],
            "direction": p.get("direction"),
            "rally_index": p.get("rally_index"),
            "to_net_ratio": p.get("to_net_ratio"),
            "rmse_px": p.get("rmse_px"),
            "time_of_flight_s": p["time_of_flight_s"],
            "method": p["method"],
            "trajectory": p["trajectory"],
            "points_px": p.get("points_px") or [],
        })
    return {"job_id": jid, "flights": out}


@app.get("/api/v1/rallies/{jid}")
def rallies(jid: str):
    """Розыгрыши, найденные при анализе (кадровые границы и фазы по сторонам)."""
    job = storage.get_job(jid)
    if not job:
        raise HTTPException(404, "job не найден")
    return {"job_id": jid, "rallies": json.loads(job.get("rallies_json") or "[]")}
