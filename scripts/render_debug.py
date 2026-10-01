"""CLI-проверка флага render_mode: полная траектория vs только пасы.

Примеры:
  python scripts/render_debug.py video.mp4 --mode full  --out data/debug_full.mp4
  python scripts/render_debug.py video.mp4 --mode passes --out data/debug_passes.mp4
  BT_RENDER_MODE=full uvicorn app.main:app   # то же самое через env для всего сервиса

Режим "full" рисует ВСЮ траекторию мяча до поиска параболических участков и
все баллистические LSQ-фиты БЕЗ фильтров (rmse/gravity/disp) с подписями,
какой фильтр завалил участок — это позволяет проверить гипотезу «пас почти не
детектится, потому что баллистическая траектория не находится».
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["passes", "full"], default="passes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    from app.config import settings
    settings.render_mode = args.mode          # флаг включает диагностику в analyze_video
    from app.services.pipeline import analyze_video
    from app.services.renderer import render_tracked_video

    out = args.out or args.video.rsplit(".", 1)[0] + f"_tracked_{args.mode}.mp4"
    print(f"[render_debug] mode={args.mode} анализ {args.video} ...")
    analysis = analyze_video(args.video, max_frames=args.max_frames)
    d = analysis.fit_diagnostics
    if d:
        cov = d["coverage_px"]
        print(f"[render_debug] трек: {len(analysis.ball_track_points)} точек "
              f"(det_rate={analysis.detection_rate:.0%}), розыгрышей={len(analysis.rallies)}, "
              f"пасов={len(analysis.passes)}")
        print(f"[render_debug] фиты без фильтров: total={d['total_raw_fits']}, "
              f"прошли бы все фильтры={d['total_would_pass']}, "
              f"покрытие трека баллистикой={cov['coverage']:.0%} "
              f"({cov['ballistic_frames']}/{cov['track_frames']} кадров)")
        for ft in d["fits"]:
            print(f"  fit #{ft['fit_index']} frames[{ft['start_frame']}..{ft['end_frame']}] "
                  f"dir={ft['direction']:5s} grav_err={ft['grav_err_rel']:.2f} "
                  f"rmse_frac={ft['rmse_frac']:.3f} disp_frac={ft['disp_frac']:.3f} "
                  f"-> {'PASS' if ft['would_pass_filters'] else 'FAIL:' + ','.join(ft['failed_checks'])}")
    render_tracked_video(args.video, out, analysis, mode=args.mode)
    print(f"[render_debug] готово: {out}")


if __name__ == "__main__":
    main()
