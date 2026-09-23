"""Генерация синтетического тестового видео: два «игрока» и мяч, летящий
по параболической траектории между ними (pass). Нужно для E2E-теста без
реальных спортивных записей.

Запуск:  python tools/make_test_video.py [out.mp4]
"""
import os
import sys

import cv2
import numpy as np


def make(path: str = "data/test_pass.mp4", fps: int = 30, seconds: float = 6.0):
    W, H = 640, 480
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = int(fps * seconds)
    hold1 = int(fps * 1.0)        # держим мяч
    flight = int(fps * 2.5)       # полёт (pass)
    x0, y0 = 140, 300             # позиция мяча у первого игрока
    x1, y1 = 500, 320             # у второго
    apex = 140                    # вершина траектории (y вниз => выше = меньше)
    for i in range(n):
        frame = np.full((H, W, 3), 40, np.uint8)
        if i < hold1:
            bx, by = x0, y0
        elif i < hold1 + flight:
            t = (i - hold1) / flight
            bx = x0 + (x1 - x0) * t
            # квадратичная кривая Безье через apex — парабola
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * apex + t ** 2 * y1
        else:
            bx, by = x1, y1
        cv2.rectangle(frame, (x0 - 30, y0 - 90), (x0 + 30, y0 + 90), (200, 80, 80), -1)
        cv2.rectangle(frame, (x1 - 30, y1 - 90), (x1 + 30, y1 + 90), (80, 200, 80), -1)
        cv2.circle(frame, (int(bx), int(by)), 14, (0, 220, 255), -1)
        cv2.circle(frame, (int(bx), int(by)), 14, (0, 0, 0), 2)
        vw.write(frame)
    vw.release()
    print("written:", path, os.path.getsize(path), "bytes,", n, "frames")


if __name__ == "__main__":
    make(*(sys.argv[1:2] or []))
