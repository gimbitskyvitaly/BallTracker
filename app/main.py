"""Service entrypoint: volleyball pass-trajectory tracker.

Configuration comes exclusively from the ``.env`` file next to the project
root (see ``config/settings.py``); CLI flags exist only to override individual
values for convenience.

Usage:
    python -m app.main                       # everything from .env
    python -m app.main --video path.mp4      # override VIDEO_PATH
    DRAW_ALL_TRAJECTORIES=true python -m app.main
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when run as `python app/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from core.pipeline import process  # noqa: E402

LOG = logging.getLogger("pass_tracker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volleyball setting-pass trajectory tracker (back-line camera)"
    )
    parser.add_argument("--env-file", help="path to a custom .env file")
    parser.add_argument("--video", dest="video_path", help="override VIDEO_PATH")
    parser.add_argument("--model", dest="model_path", help="override MODEL_PATH")
    parser.add_argument("--output-dir", dest="output_dir", help="override OUTPUT_DIR")
    parser.add_argument(
        "--draw-all",
        dest="draw_all_trajectories",
        action="store_true",
        default=None,
        help="draw ALL ball trajectories, not only pass trajectories "
             "(overrides DRAW_ALL_TRAJECTORIES from .env)",
    )
    parser.add_argument(
        "--passes-only",
        dest="draw_all_trajectories",
        action="store_false",
        help="draw only pass trajectories (overrides DRAW_ALL_TRAJECTORIES=false)",
    )
    parser.set_defaults(draw_all_trajectories=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings(args.env_file)

    overrides = {}
    if args.video_path:
        overrides["video_path"] = args.video_path
    if args.model_path:
        overrides["model_path"] = args.model_path
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.draw_all_trajectories is not None:
        overrides["draw_all_trajectories"] = args.draw_all_trajectories
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)

    mode = "ALL trajectories" if settings.draw_all_trajectories else "PASS trajectories only"
    LOG.info("Input video: %s", settings.video_path)
    LOG.info("Model: %s", settings.resolved_model_path)
    LOG.info("Rendering mode: %s", mode)

    try:
        artifacts = process(settings)
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001
        LOG.exception("Pipeline failed")
        return 1

    LOG.info("Done. Artifacts:")
    for name, path in artifacts.items():
        LOG.info("  %-9s %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
