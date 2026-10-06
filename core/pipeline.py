"""End-to-end processing pipeline.

Mirrors the upstream repo layout (https://github.com/asigatchov/fast-volleyball-tracking-inference):

* step 1 - sequential ONNX ball detection over the whole video -> ``ball.csv``
* step 2 - greedy nearest-neighbour tracking -> raw trajectories
* step 3 - outlier removal (teleport rule) -> clean trajectories
* step 4 - pass classification (hyperbola + over net + to a net edge)
* step 5 - JSON report + annotated video rendering
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from core.detector import BallDetector
from core.models import NetGeometry, TrackPoint, Trajectory
from core.outliers import remove_outliers
from core.pass_classifier import classify_pass
from core.tracker import BallTrajectoryTracker
from core.visualizer import render_video

LOG = logging.getLogger(__name__)


def build_net_geometry(settings: Settings, width: int, height: int) -> NetGeometry:
    """Scale the normalized net geometry from .env to real pixel coordinates."""
    net = settings.net
    return NetGeometry(
        enabled=net.enabled,
        top_y=net.top_y * height,
        left_x=net.left_x * width,
        right_x=net.right_x * width,
        clearance_px=net.clearance_norm * height,
    )


def detect_ball_positions(
    video_path: str, detector: BallDetector, settings: Settings
) -> Dict[int, Tuple[float, float]]:
    """Stream the video through the sequential model; frame -> (x, y) in px."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    positions: Dict[int, Tuple[float, float]] = {}
    batch: List[Tuple[int, np.ndarray]] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    frame_idx = 0
    emitted = 0

    def flush(items: List[Tuple[int, np.ndarray]]) -> None:
        nonlocal emitted
        if not items:
            return
        frames = [f for _, f in items]
        preds = detector.detect_batch(frames, settings.confidence_threshold)
        # The sequential model decodes its whole receptive field every call;
        # only the newest `len(frames)` predictions correspond to these frames.
        for (idx, _), pos in zip(items, preds[-len(items):]):
            emitted += 1
            if pos is not None:
                positions[idx] = pos
        if emitted % 900 == 0:
            LOG.info("Detected through frame %s / %s", idx, total_frames)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append((frame_idx, frame))
        frame_idx += 1
        if len(batch) >= detector.batch_size:
            flush(batch)
            batch = []
    flush(batch)
    cap.release()
    LOG.info("Processed %s frames, ball visible in %s", frame_idx, len(positions))
    return positions


def write_ball_csv(positions: Dict[int, Tuple[float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "x", "y", "visible"])
        for frame in sorted(positions):
            x, y = positions[frame]
            writer.writerow([frame, round(x, 2), round(y, 2), 1])


def build_trajectories(
    positions: Dict[int, Tuple[float, float]],
    total_frames: int,
    tracker: BallTrajectoryTracker,
    settings: Settings,
) -> List[Trajectory]:
    """Feed every frame into the tracker and collect finished trajectories."""
    for frame in range(total_frames):
        det = positions.get(frame)
        tracker.update(frame, det)
    raw_tracks = tracker.finalize()

    trajectories: List[Trajectory] = []
    for tid, points in enumerate(raw_tracks):
        if len(points) < settings.min_track_points:
            continue
        traj = Trajectory(
            track_id=tid,
            start_frame=points[0].frame,
            end_frame=points[-1].frame,
            points=list(points),
        )
        trajectories.append(traj)
    return trajectories


def process(settings: Settings) -> Dict[str, Path]:
    """Run the full service pipeline; returns paths of produced artifacts."""
    video_path = settings.video_path
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = settings.resolved_output_dir
    stem = Path(video_path).stem
    run_dir = out_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    fps = settings.fps if settings.fps > 0 else video_fps
    LOG.info(
        "Video %s: %dx%d, %d frames, %.2f fps",
        Path(video_path).name, width, height, total_frames, fps,
    )

    # ---- 1. detection ------------------------------------------------------
    detector = BallDetector(str(settings.resolved_model_path), device=settings.device)
    positions = detect_ball_positions(video_path, detector, settings)
    ball_csv = run_dir / "ball.csv"
    write_ball_csv(positions, ball_csv)

    # ---- 2. tracking -------------------------------------------------------
    tracker = BallTrajectoryTracker(
        max_distance=settings.max_distance,
        max_disappeared=settings.max_disappeared,
        frame_width=width,
        fps=fps,
    )
    trajectories = build_trajectories(positions, total_frames, tracker, settings)
    LOG.info("Built %d raw trajectories", len(trajectories))

    # ---- 3. outlier removal -------------------------------------------------
    eps = settings.outlier.eps_ratio * width
    for traj in trajectories:
        clean, removed = remove_outliers(
            traj.points, eps, settings.outlier.max_iterations
        )
        traj.points = clean
        traj.removed_outliers = [(f, (x, y)) for f, x, y in removed]
        if clean:
            traj.start_frame = clean[0].frame
            traj.end_frame = clean[-1].frame
    trajectories = [t for t in trajectories if len(t.points) >= settings.min_track_points]

    # ---- 4. pass classification ---------------------------------------------
    net = build_net_geometry(settings, width, height)
    for traj in trajectories:
        classify_pass(traj, fps, net, settings.pass_cfg, width, height)
    passes = [t for t in trajectories if t.is_pass]
    LOG.info("Classified %d/%d trajectories as passes", len(passes), len(trajectories))

    # ---- 5. artifacts ---------------------------------------------------------
    report = {
        "video": str(video_path),
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "draw_all_trajectories": settings.draw_all_trajectories,
        "net_geometry": {
            "top_y_px": net.top_y,
            "left_antenna_px": net.left_x,
            "right_antenna_px": net.right_x,
            "clearance_px": net.clearance_px,
        },
        "num_trajectories": len(trajectories),
        "num_passes": len(passes),
        "passes": [t.pass_event.to_dict() for t in passes if t.pass_event],
        "trajectories": [t.to_dict() for t in trajectories],
    }
    report_path = run_dir / "passes.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    LOG.info("Report written to %s", report_path)

    artifacts: Dict[str, Path] = {"ball_csv": ball_csv, "report": report_path}

    if settings.write_video:
        rendered = run_dir / "result.mp4"
        render_video(
            video_path=video_path,
            output_path=rendered,
            trajectories=trajectories,
            settings=settings,
            net=net,
            fps=fps,
        )
        artifacts["video"] = rendered
        LOG.info("Rendered video written to %s", rendered)

    return artifacts
