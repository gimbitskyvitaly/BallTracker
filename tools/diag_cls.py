"""Симуляция сегментов/классификатора на реальном видео без детектора."""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from app.config import settings
from app.services.pipeline import _is_pass, _drop_outliers, _median_filter

vid = "videos/VID_20260925_163505.mp4"
a_pts = None
# переиспользуем трек из предыдущего прогона? пересчитаем быстро через analyze_video нельзя (долго).
# Вместо этого: читаем ball_track_points из БД последнего job.
import sqlite3, json
con = sqlite3.connect("data/balltime.db")
rows = con.execute("select id from videos order by created_at desc limit 5").fetchall() if False else []
try:
    rows = [r[0] for r in con.execute("select job_id from passes order by rowid desc limit 5")]
except Exception as e:
    print("db err", e)
print("recent jobs:", rows)
con.close()
