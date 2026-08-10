import os
import sys
import argparse
from tracker import process_cricket_video, generate_sample_video


def main():
    parser = argparse.ArgumentParser(description="Cricket Ball AI Tracking CLI")
    parser.add_argument("--input", "-i", type=str, default="data/raw/cricket_video.mp4", help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="outputs/tracked_output.mp4", help="Path to save output video")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="YOLO Confidence Threshold")
    parser.add_argument("--pitch-pixels", type=float, default=500.0, help="Pitch length calibration in pixels")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[INFO] Input video '{input_path}' not found. Generating sample cricket video...")
        input_path = generate_sample_video(input_path)

    print(f"[INFO] Processing video: {input_path}")
    metrics = process_cricket_video(
        video_path=input_path,
        output_path=args.output,
        conf_thresh=args.conf,
        pitch_pixel_len=args.pitch_pixels
    )

    print("=" * 60)
    print("  CRICKET BALL TRACKING & DRS REVIEW REPORT")
    print("=" * 60)
    print(f" Output Video    : {metrics['output_path']}")
    print(f" Telemetry CSV   : {metrics.get('csv_path', 'N/A')}")
    print(f" Total Frames    : {metrics['total_frames']}")
    print(f" Tracked Points  : {metrics['trajectory_points']}")
    print(f" Peak Speed      : {metrics['max_speed_kmh']} km/h")
    print(f" Average Speed   : {metrics['avg_speed_kmh']} km/h")
    if metrics.get('bounce_point'):
        print(f" Bounce Location : X={metrics['bounce_point'][0]}, Y={metrics['bounce_point'][1]}")
    if metrics.get('impact_point'):
        print(f" Impact Location : X={metrics['impact_point'][0]}, Y={metrics['impact_point'][1]}")
    print("-" * 60)
    print(f" PITCHING        : {metrics.get('drs_pitching', 'N/A')}")
    print(f" IMPACT          : {metrics.get('drs_impact', 'N/A')}")
    print(f" WICKETS         : {metrics.get('drs_wickets', 'N/A')}")
    print(f" DECISION        : {metrics.get('drs_decision', 'N/A')} ({metrics.get('drs_hit_detail', '')})")
    print("=" * 60)


if __name__ == "__main__":
    main()
