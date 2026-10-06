"""Synthetic test-video generator emulating a back-line volleyball camera.

Draws a net (top cable + antennas), a ball flying along ballistic parabolic
arcs, injects single-frame detection "teleports" (outliers) and long fast
flights (attack/serve) so the outlier filter and the pass classifier can be
verified end-to-end without a real match recording.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Flight:
    """One ballistic flight of the ball between two court points."""

    start_frame: int
    duration: int               # frames
    p0: Tuple[float, float]     # release point (px)
    p1: Tuple[float, float]     # landing point (px)
    apex_lift: float            # px above the straight chord at mid-flight


def render_flights(
    width: int,
    height: int,
    fps: int,
    net_top_y: float,
    net_left_x: float,
    net_right_x: float,
    flights: List[Flight],
    outlier_frames: List[Tuple[int, Tuple[float, float]]],
    out_path: str,
) -> None:
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total = max(f.start_frame + f.duration for f in flights) + 30

    positions: dict[int, Tuple[float, float]] = {}
    for fl in flights:
        for k in range(fl.duration):
            t = k / max(fl.duration - 1, 1)
            x = fl.p0[0] + (fl.p1[0] - fl.p0[0]) * t
            y = fl.p0[1] + (fl.p1[1] - fl.p0[1]) * t - 4 * fl.apex_lift * t * (1 - t)
            positions[fl.start_frame + k] = (x, y)
    for frame, pos in outlier_frames:
        positions[frame] = pos  # injected teleport

    for frame_no in range(total):
        canvas = np.full((height, width, 3), 60, dtype=np.uint8)
        # court floor
        cv2.rectangle(canvas, (0, int(height * 0.55)), (width, height), (90, 120, 40), -1)
        # net top cable + antennas
        cv2.line(canvas, (int(net_left_x), int(net_top_y)),
                 (int(net_right_x), int(net_top_y)), (255, 255, 255), 3)
        for ax in (net_left_x, net_right_x):
            cv2.line(canvas, (int(ax), int(net_top_y)), (int(ax), int(height * 0.72)),
                     (200, 200, 200), 2)
        pos = positions.get(frame_no)
        if pos is not None:
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(canvas, (x, y), 8, (0, 215, 255), -1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def build_default_scene(width: int = 1280, height: int = 720) -> Tuple[List[Flight], List[Tuple[int, Tuple[float, float]]]]:
    net_top_y = 0.36 * height
    net_left_x = 0.21 * width
    net_right_x = 0.79 * width

    flights: List[Flight] = []
    # 1) PASS to the right antenna: released near the left-back zone, arcs
    #    above the net cable, descends softly near the right antenna.
    flights.append(Flight(30, 45, (0.30 * width, 0.75 * height),
                          (net_right_x - 0.05 * width, 0.55 * height),
                          apex_lift=0.35 * height))
    # 2) Attack/serve: fast flat-ish flight, no over-net apex near an antenna.
    flights.append(Flight(120, 12, (0.10 * width, 0.30 * height),
                          (0.90 * width, 0.80 * height), apex_lift=0.02 * height))
    # 3) PASS to the left antenna.
    flights.append(Flight(180, 50, (0.68 * width, 0.78 * height),
                          (net_left_x + 0.04 * width, 0.52 * height),
                          apex_lift=0.30 * height))
    # 4) Random noise detections with teleports (outliers get filtered).
    rng = np.random.default_rng(7)
    base = (0.5 * width, 0.6 * height)
    for k in range(40):
        jitter = (base[0] + rng.normal(0, 4), base[1] + rng.normal(0, 4))
        flights.append(Flight(260 + k, 1, jitter, jitter, 0.0))
    outlier_frames = [
        # teleport inside trajectory 1: far away for one frame, then returns
        (52, (0.05 * width, 0.10 * height)),
        (53, (0.95 * width, 0.90 * height)),
    ]
    return flights, outlier_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic volleyball test video")
    parser.add_argument("--output", default="examples/match.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    flights, outliers = build_default_scene(args.width, args.height)
    render_flights(
        width=args.width,
        height=args.height,
        fps=args.fps,
        net_top_y=0.36 * args.height,
        net_left_x=0.21 * args.width,
        net_right_x=0.79 * args.width,
        flights=flights,
        outlier_frames=outliers,
        out_path=args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
