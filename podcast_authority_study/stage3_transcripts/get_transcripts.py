"""
Stage 3: Get Transcript Text
==============================
For each video in the metadata file, attempts to retrieve transcripts
using multiple fallback strategies:
  1. YouTube captions API (creator-uploaded or auto-generated)
  2. youtube-transcript-api library (community/auto captions)
  3. Audio download + Whisper ASR (self-transcribed)

Each transcript is tagged with its source for analytical transparency.

Usage:
    python -m stage3_transcripts.get_transcripts [--whisper-fallback]

Output:
    data/transcripts/{video_id}.json
    data/transcript_index.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Try imports with graceful fallbacks
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YT_TRANSCRIPT = True
except ImportError:
    HAS_YT_TRANSCRIPT = False
    print("WARNING: youtube-transcript-api not installed. Install for caption retrieval.")

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def get_transcript_yt_api(video_id: str) -> tuple[list[dict] | None, str]:
    """Try to get transcript via youtube-transcript-api."""
    if not HAS_YT_TRANSCRIPT:
        return None, "unavailable"

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created captions
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
            segments = transcript.fetch()
            return segments, "creator_caption"
        except Exception:
            pass

        # Fall back to auto-generated
        try:
            transcript = transcript_list.find_generated_transcript(["en"])
            segments = transcript.fetch()
            return segments, "auto_caption"
        except Exception:
            pass

    except Exception as e:
        pass

    return None, "unavailable"


def transcribe_with_whisper(video_id: str, audio_dir: str, model_name: str = "base") -> tuple[list[dict] | None, str]:
    """Download audio and transcribe with Whisper."""
    if not HAS_WHISPER:
        return None, "whisper_unavailable"

    audio_path = os.path.join(audio_dir, f"{video_id}.mp3")

    # Download audio if not already cached
    if not os.path.exists(audio_path):
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "5",  # Lower quality is fine for speech
                    "-o", audio_path,
                    f"https://www.youtube.com/watch?v={video_id}",
                ],
                capture_output=True,
                timeout=600,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Audio download failed for {video_id}: {e}")
            return None, "download_failed"

    # Transcribe
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language="en")

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "duration": seg["end"] - seg["start"],
            })

        return segments, "self_asr"

    except Exception as e:
        print(f"  Whisper transcription failed for {video_id}: {e}")
        return None, "asr_failed"


def save_transcript(video_id: str, segments: list[dict], source: str,
                    metadata: dict, output_dir: str):
    """Save transcript as structured JSON."""
    doc = {
        "video_id": video_id,
        "channel_name": metadata.get("channel_name", ""),
        "persona_type": metadata.get("persona_type", ""),
        "title": metadata.get("title", ""),
        "publish_date": metadata.get("publish_date", ""),
        "duration_seconds": int(metadata.get("duration_seconds", 0)),
        "transcript_source": source,
        "segment_count": len(segments),
        "total_words": sum(len(s.get("text", "").split()) for s in segments),
        "segments": segments,
    }

    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return doc


def load_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Get transcripts")
    parser.add_argument("--metadata", default="data/video_metadata.csv")
    parser.add_argument("--output-dir", default="data/transcripts")
    parser.add_argument("--index-output", default="data/transcript_index.csv")
    parser.add_argument("--whisper-fallback", action="store_true",
                        help="Use Whisper ASR when captions unavailable")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--audio-dir", default="data/audio_cache")
    parser.add_argument("--channel", help="Process only this channel (for testing)")
    parser.add_argument("--limit", type=int, help="Max videos to process (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have transcripts")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)

    videos = load_metadata(args.metadata)
    if args.channel:
        videos = [v for v in videos if v["channel_name"] == args.channel]
    if args.limit:
        videos = videos[: args.limit]

    print(f"Processing {len(videos)} videos")

    index_rows = []
    stats = {"creator_caption": 0, "auto_caption": 0, "self_asr": 0, "unavailable": 0}

    for video in tqdm(videos, desc="Transcripts"):
        vid = video["video_id"]

        # Skip if already processed
        if args.skip_existing and os.path.exists(os.path.join(args.output_dir, f"{vid}.json")):
            continue

        # Strategy 1: YouTube captions
        segments, source = get_transcript_yt_api(vid)

        # Strategy 2: Whisper fallback
        if segments is None and args.whisper_fallback:
            print(f"  Trying Whisper for {vid} ({video.get('title', '')[:50]})")
            segments, source = transcribe_with_whisper(vid, args.audio_dir, args.whisper_model)

        if segments is None:
            source = "unavailable"
            stats["unavailable"] += 1
            index_rows.append({
                "video_id": vid,
                "channel_name": video.get("channel_name", ""),
                "transcript_source": source,
                "segment_count": 0,
                "total_words": 0,
            })
            continue

        stats[source] = stats.get(source, 0) + 1

        doc = save_transcript(vid, segments, source, video, args.output_dir)

        index_rows.append({
            "video_id": vid,
            "channel_name": video.get("channel_name", ""),
            "transcript_source": source,
            "segment_count": doc["segment_count"],
            "total_words": doc["total_words"],
        })

        time.sleep(0.2)  # Be polite to YouTube

    # Write index
    with open(args.index_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "channel_name", "transcript_source", "segment_count", "total_words"
        ])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n{'=' * 60}")
    print(f"Transcript retrieval complete")
    print(f"  Creator captions: {stats['creator_caption']}")
    print(f"  Auto captions:    {stats['auto_caption']}")
    print(f"  Self ASR:         {stats['self_asr']}")
    print(f"  Unavailable:      {stats['unavailable']}")
    print(f"Index: {args.index_output}")


if __name__ == "__main__":
    main()
