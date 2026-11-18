"""
CLI entry point for Phase 1 — local video file in storage.

    python run.py --input match_clip.mp4 --output match_clip_overlay.mp4

Calibration currently must be supplied in code (see calibration.py);
wire up your own CalibrationConfig with real keypoints from your clip
before running against footage you care about — the example in
calibration.example_config_for_frame() is illustrative only and will
NOT project correctly onto an arbitrary video.
"""
import argparse

from calibration import example_config_for_frame
from config import PipelineConfig
from pipeline import Phase1Pipeline


def main():
    parser = argparse.ArgumentParser(description="Phase 1 formation overlay pipeline")
    parser.add_argument("--input", required=True, help="Path to a video file on local storage")
    parser.add_argument("--output", required=True, help="Path to write the overlaid video")
    parser.add_argument("--min-gap-m", type=float, default=7.0)
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_video_path=args.input,
        output_video_path=args.output,
        calibration=example_config_for_frame(),  # replace with real keypoints
    )
    cfg.formation.min_gap_m = args.min_gap_m

    pipeline = Phase1Pipeline(cfg)
    pipeline.run()
    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()
