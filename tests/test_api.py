"""Интеграционные тесты API (FastAPI TestClient) с мок-детектором.

Реальный YOLO на синтетике не детектирует круг-мяч, поэтому в pipeline
подставляется mock-класс BallDetector, возвращающий детерминированные
детекции мяча и двух «игроков». Так проверяются все слои: загрузка видео →
фон job → трекинг → события паса → физика → SQLite → JSON API.
"""
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BT_DB_PATH", "data/test_balltime.db")
os.environ.setdefault("BT_UPLOAD_DIR", "data/test_uploads")

from app.config import settings  # noqa: E402
from app.services.detector import Detection  # noqa: E402


# --- мок-детектор: аналитическая парабола мяча + два rect-«игрока» ---------
FPS = 30.0
HOLD1 = int(FPS * 1.0)
FLIGHT = int(FPS * 2.5)
X0, Y0 = 140.0, 300.0
X1, Y1 = 500.0, 320.0
APEX = 140.0
G = settings.gravity_ratio * FPS * FPS


def ball_pos(i: int):
    if i < HOLD1:
        return X0, Y0
    if i < HOLD1 + FLIGHT:
        t = (i - HOLD1) / FLIGHT
        x = X0 + (X1 - X0) * t
        y = (1 - t) ** 2 * Y0 + 2 * (1 - t) * t * APEX + t ** 2 * Y1
        return x, y
    return X1, Y1


class MockDetector:
    def __init__(self):
        self.frame = 0

    def detect(self, frame):
        self.frame += 1
        i = self.frame - 1
        bx, by = ball_pos(i)
        r = 14.0
        balls = [Detection(bbox=np.array([bx - r, by - r, bx + r, by + r]), conf=0.9, cls=32)]
        persons = [
            Detection(bbox=np.array([X0 - 30, Y0 - 90, X0 + 30, Y0 + 90]), conf=0.9, cls=0),
            Detection(bbox=np.array([X1 - 30, Y1 - 90, X1 + 30, Y1 + 90]), conf=0.9, cls=0),
        ]
        return balls, persons


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.main as main
    main._detector = MockDetector()   # подменяем singleton до первого запроса
    dbf = str(tmp_path_factory.mktemp("db") / "t.db")
    settings.db_path = dbf
    settings.upload_dir = str(tmp_path_factory.mktemp("up"))
    from app.services import storage
    storage.init_db()
    with TestClient(main.app) as c:
        yield c


def _wait_done(client, jid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/v1/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise TimeoutError(jid)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_full_pipeline(client):
    video = "data/test_pass.mp4"
    assert os.path.exists(video), "сначала выполните python tools/make_test_video.py"
    with open(video, "rb") as f:
        r = client.post("/api/v1/videos", files={"file": ("pass.mp4", f, "video/mp4")})
    assert r.status_code == 200
    jid = r.json()["id"]
    j = _wait_done(client, jid)
    assert j["status"] == "done", j.get("error")
    assert j["fps"] == pytest.approx(FPS, abs=1)

    passes = client.get(f"/api/v1/passes/{jid}").json()
    assert len(passes) >= 1, "должно быть обнаружено минимум одно событие паса"
    p = passes[0]
    # ожидаемое время полёта ~2.5 c (релиз на ~кадре 30, приёмка ~105)
    assert 1.8 <= p["time_of_flight_s"] <= 3.2, p
    assert p["method"] in ("physics_fit", "tracked_direct")
    assert p["apex_height_m"] > 0.5
    assert p["distance_m"] > 1.0
    assert p["initial_speed_mps"] > 0.5
    traj = client.get(f"/api/v1/trajectories/{jid}").json()
    assert len(traj["flights"][0]["trajectory"]) > 10

    jobs = client.get("/api/v1/jobs").json()
    assert any(x["id"] == jid for x in jobs)


def test_rejects_non_video(client):
    r = client.post("/api/v1/videos",
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_job_not_found(client):
    assert client.get("/api/v1/jobs/deadbeef").status_code == 404
