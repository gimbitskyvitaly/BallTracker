"""Интеграционные тесты API (FastAPI TestClient) с mock-детектором VballNet.

Реальная ONNX-модель на синтетике не нужна (и тяжела для CI): в pipeline
подставляется заглушка, возвращающая детерминированные позиции мяча —
баллистические полёты К сетке (пасы) и «ведение» мяча (не пас). Так
проверяются все слои: загрузка видео → job → трек/розыгрыши → события
пасов → физика → SQLite → JSON API → рендер видео с траекториями.
"""
import math
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BT_DB_PATH", "data/test_balltime.db")
os.environ.setdefault("BT_UPLOAD_DIR", "data/test_uploads")

from app.config import settings  # noqa: E402
from app.services.vballnet import BallPoint  # noqa: E402

FPS = 30.0
W, H = 640, 480
G_PX_S2 = FPS * FPS * 0.55

# Сцена: розыгрыш 1 — пас ВЛЕВО к сетке (кадры 0..26), затем «ведение» (не пас);
#         розыгрыш 2 — пас ВПРАВО к сетке (кадры 90..115).
PASS1 = [(i, 520.0 - 280 * i / FPS,
          320.0 - 320 * i / FPS + 0.5 * G_PX_S2 * (i / FPS) ** 2)
         for i in range(27)]
CARRY = [(30 + i, 150.0 + 2.0 * i, 380.0 + 0.5 * i) for i in range(55)]
# y считается от начала полёта (i-90): иначе парабола улетает за кадр и
# трек обрывается интерполяцией на границе видео.
PASS2 = [(i, 120.0 + 280 * (i - 90) / FPS,
          320.0 - 320 * (i - 90) / FPS + 0.5 * G_PX_S2 * ((i - 90) / FPS) ** 2)
         for i in range(90, 116)]
SCENE = {f: (x, y) for f, x, y in PASS1 + CARRY + PASS2}


class MockVballNet:
    """Заглушка VballNetDetector.feed: окно кадров -> точки мяча по расписанию."""

    def __init__(self, scene=SCENE):
        self.scene = scene

    def feed(self, frames, start_idx=0):
        out = []
        for i in range(len(frames)):
            fr = start_idx + i                # 0-based номера кадров сцены
            if fr in self.scene:
                x, y = self.scene[fr]
                out.append(BallPoint(frame=fr, x=x, y=y, conf=0.9))
            else:
                out.append(None)
        return out


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.main as main
    main.get_detector = lambda: MockVballNet()   # mock до первого запроса
    dbf = str(tmp_path_factory.mktemp("db") / "t.db")
    settings.db_path = dbf
    settings.upload_dir = str(tmp_path_factory.mktemp("up"))
    settings.render_dir = str(tmp_path_factory.mktemp("renders"))
    from app.services import storage
    storage.init_db()
    with TestClient(main.app) as c:
        yield c


def _wait_done(client, jid, timeout=90):
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
    # ровно два паса: ведение мяча (без баллистики) пасом считаться не должно
    assert len(passes) == 2, f"ожидалось 2 паса, получено {len(passes)}"
    p = passes[0]
    tof_expected = (p["catch_frame"] - p["release_frame"]) / FPS
    assert abs(p["time_of_flight_s"] - tof_expected) < 0.1
    assert p["method"] in ("physics_fit", "tracked_direct")
    assert p["apex_height_m"] > 0.3
    assert p["distance_m"] > 1.0
    assert p["initial_speed_mps"] > 0.5
    # направление: первый пас летит влево к сетке, второй — вправо
    assert p["direction"] == "left"
    assert passes[1]["direction"] == "right"
    assert p["to_net_ratio"] < 0.85 and passes[1]["to_net_ratio"] < 0.85

    traj = client.get(f"/api/v1/trajectories/{jid}").json()
    assert len(traj["flights"][0]["trajectory"]) > 10
    assert len(traj["flights"][0]["points_px"]) >= 20

    rallies = client.get(f"/api/v1/rallies/{jid}").json()["rallies"]
    assert len(rallies) >= 1
    # оба паса лежат внутри своих розыгрышей
    for fl in traj["flights"]:
        assert any(r["start_frame"] <= fl["release_frame"] <= r["end_frame"]
                   for r in rallies)

    jobs = client.get("/api/v1/jobs").json()
    assert any(x["id"] == jid for x in jobs)

    # видео с отрисованными траекториями сформировано и отдаётся
    detail = client.get(f"/api/v1/jobs/{jid}").json()
    assert detail["has_result_video"]
    v = client.get(f"/api/v1/jobs/{jid}/video")
    assert v.status_code == 200
    assert len(v.content) > 1000


def test_rejects_non_video(client):
    r = client.post("/api/v1/videos",
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_job_not_found(client):
    assert client.get("/api/v1/jobs/deadbeef").status_code == 404
    assert client.get("/api/v1/rallies/deadbeef").status_code == 404
