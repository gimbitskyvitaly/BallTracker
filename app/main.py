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

import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.services import storage
from app.services.detector import BallDetector
from app.services.pipeline import analyze_video

app = FastAPI(title="BallTrack — pass trajectory & time-of-flight", version="1.0.0")

_executor = ThreadPoolExecutor(max_workers=2)
_detector: BallDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> BallDetector:
    """Ленивый singleton: загрузка YOLO дорогая."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = BallDetector()
    return _detector


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


def _run_job(jid: str, path: str) -> None:
    storage.set_status(jid, "running")
    try:
        analysis = analyze_video(path, detector=get_detector())
        passes = []
        for p in analysis.passes:
            d = {
                "release_frame": p.release_frame,
                "catch_frame": p.catch_frame,
                "passer_bbox": p.passer_bbox,
                "catcher_bbox": p.catcher_bbox,
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
        storage.set_status(jid, "done", meta={
            "fps": analysis.fps, "width": analysis.width,
            "height": analysis.height, "n_frames": analysis.n_frames})
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
    return JobOut(**storage.get_job(jid))


@app.get("/api/v1/jobs", response_model=list[JobOut])
def jobs(limit: int = 50):
    return [JobOut(**j) for j in storage.list_jobs(limit)]


@app.get("/api/v1/jobs/{jid}", response_model=JobOut)
def job(jid: str):
    j = storage.get_job(jid)
    if not j:
        raise HTTPException(404, "job не найден")
    return JobOut(**j)


@app.get("/api/v1/passes/{jid}")
def passes(jid: str):
    if not storage.get_job(jid):
        raise HTTPException(404, "job не найден")
    return storage.get_passes(jid)


@app.get("/api/v1/trajectories/{jid}")
def trajectories(jid: str):
    """Точки траекторий по каждому полёту + сырой трек мяча."""
    if not storage.get_job(jid):
        raise HTTPException(404, "job не найден")
    out = []
    for p in storage.get_passes(jid):
        out.append({
            "release_frame": p["release_frame"],
            "catch_frame": p["catch_frame"],
            "time_of_flight_s": p["time_of_flight_s"],
            "method": p["method"],
            "trajectory": p["trajectory"],
        })
    return {"job_id": jid, "flights": out}
