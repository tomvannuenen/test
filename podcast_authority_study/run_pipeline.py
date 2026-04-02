"""
Podcast Authority Study - Pipeline Orchestrator
=================================================
Master script to run the full pipeline or individual stages.

Usage:
    python run_pipeline.py --stage all --yt-api-key KEY --anthropic-api-key KEY
    python run_pipeline.py --stage 1 --yt-api-key KEY
    python run_pipeline.py --stage 3 --whisper-fallback
    python run_pipeline.py --stage 5 --anthropic-api-key KEY --sample-rate 0.3

Environment variables (alternative to CLI args):
    YOUTUBE_API_KEY
    ANTHROPIC_API_KEY
"""

import argparse
import os
import subprocess
import sys


def run_stage(stage_num: int, args: argparse.Namespace):
    """Run a pipeline stage as a subprocess."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if stage_num == 1:
        cmd = [
            sys.executable, "-m", "stage1_corpus.define_corpus",
            "--api-key", args.yt_api_key or os.environ.get("YOUTUBE_API_KEY", ""),
        ]
        if args.skip_validation:
            cmd.append("--skip-validation")

    elif stage_num == 2:
        cmd = [
            sys.executable, "-m", "stage2_metadata.pull_metadata",
            "--api-key", args.yt_api_key or os.environ.get("YOUTUBE_API_KEY", ""),
        ]
        if args.channel:
            cmd.extend(["--channel", args.channel])

    elif stage_num == 3:
        cmd = [sys.executable, "-m", "stage3_transcripts.get_transcripts"]
        if args.whisper_fallback:
            cmd.append("--whisper-fallback")
            cmd.extend(["--whisper-model", args.whisper_model])
        if args.channel:
            cmd.extend(["--channel", args.channel])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        cmd.append("--skip-existing")

    elif stage_num == 4:
        cmd = [
            sys.executable, "-m", "stage4_cleaning.clean_and_segment",
            "--window-seconds", str(args.window_seconds),
        ]
        if args.channel:
            cmd.extend(["--channel", args.channel])

    elif stage_num == 5:
        cmd = [
            sys.executable, "-m", "stage5_analysis.llm_coder",
            "--api-key", args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            "--model", args.model,
            "--sample-rate", str(args.sample_rate),
        ]
        if args.channel:
            cmd.extend(["--channel", args.channel])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        cmd.append("--skip-existing")

    elif stage_num == 6:  # Aggregation (5b)
        cmd = [sys.executable, "-m", "stage5_analysis.aggregate"]

    elif stage_num == 7:  # Validation export (5c)
        cmd = [
            sys.executable, "-m", "stage5_analysis.validate_coding",
            "export-sample", "--n", str(args.validation_n),
        ]

    else:
        print(f"Unknown stage: {stage_num}")
        return False

    print(f"\n{'=' * 60}")
    print(f"STAGE {stage_num}: {' '.join(cmd[:4])}...")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=base_dir)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Podcast Authority Study Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  Define corpus (validate channels against YouTube API)
  2  Pull video metadata (titles, descriptions, dates, durations)
  3  Get transcripts (captions + optional Whisper fallback)
  4  Clean and segment (merge fragments, create 5-min windows)
  5  LLM coding (apply coding frame via Claude API)
  6  Aggregate (produce summary statistics and comparisons)
  7  Validation export (sample segments for human coding)

Examples:
  python run_pipeline.py --stage 1 --yt-api-key KEY --skip-validation
  python run_pipeline.py --stage 2,3 --yt-api-key KEY
  python run_pipeline.py --stage 5 --anthropic-api-key KEY --sample-rate 0.3
  python run_pipeline.py --stage all --yt-api-key KEY --anthropic-api-key KEY
        """,
    )

    parser.add_argument("--stage", required=True,
                        help="Stage number(s) to run: 1-7, comma-separated, or 'all'")
    parser.add_argument("--yt-api-key", help="YouTube Data API v3 key")
    parser.add_argument("--anthropic-api-key", help="Anthropic API key")
    parser.add_argument("--channel", help="Process only this channel name")
    parser.add_argument("--limit", type=int, help="Max items to process per stage")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Stage 1: skip YouTube API validation")
    parser.add_argument("--whisper-fallback", action="store_true",
                        help="Stage 3: use Whisper when captions unavailable")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--window-seconds", type=int, default=300,
                        help="Stage 4: window size in seconds (default: 300)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Stage 5: Claude model to use")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Stage 5: fraction of windows to code (default: 1.0)")
    parser.add_argument("--validation-n", type=int, default=50,
                        help="Stage 7: number of segments for validation sample")

    args = parser.parse_args()

    # Determine which stages to run
    if args.stage == "all":
        stages = [1, 2, 3, 4, 5, 6, 7]
    else:
        stages = [int(s.strip()) for s in args.stage.split(",")]

    print("Podcast Authority Study Pipeline")
    print(f"Stages to run: {stages}")

    for stage in stages:
        success = run_stage(stage, args)
        if not success:
            print(f"\nERROR: Stage {stage} failed. Stopping pipeline.")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
