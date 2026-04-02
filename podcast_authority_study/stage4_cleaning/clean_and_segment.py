"""
Stage 4: Clean and Segment Transcripts
========================================
Takes raw transcript JSON files and produces analysis-ready segments:
  - Merges short caption fragments into coherent utterances
  - Splits into time-windowed segments (~5 minutes each)
  - Attempts basic speaker diarization from caption cues
  - Normalizes text (whitespace, encoding, filler words)

Usage:
    python -m stage4_cleaning.clean_and_segment

Output:
    data/segments/{video_id}_segments.json
    data/segment_index.csv
"""

import argparse
import csv
import json
import os
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm


# --- Text Cleaning ---

def normalize_text(text: str) -> str:
    """Basic text normalization for transcript segments."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove YouTube auto-caption artifacts
    text = re.sub(r"\[Music\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Applause\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Laughter\]", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text


def detect_speaker_change(text: str) -> str | None:
    """Try to detect speaker labels in caption text."""
    # Common patterns: "Speaker:", "HOST:", ">>", "- "
    match = re.match(r"^(?:>>|-)?\s*([A-Z][A-Z\s.]+):\s", text)
    if match:
        return match.group(1).strip()
    return None


# --- Segment Merging ---

def merge_fragments(segments: list[dict], min_gap: float = 1.0) -> list[dict]:
    """Merge very short consecutive caption fragments into utterances."""
    if not segments:
        return []

    merged = []
    current = {
        "text": segments[0].get("text", ""),
        "start": segments[0].get("start", 0),
        "end": segments[0].get("start", 0) + segments[0].get("duration", 0),
    }

    for seg in segments[1:]:
        seg_start = seg.get("start", 0)
        seg_end = seg_start + seg.get("duration", 0)
        seg_text = seg.get("text", "")

        # If gap is small and no speaker change, merge
        gap = seg_start - current["end"]
        speaker = detect_speaker_change(seg_text)

        if gap < min_gap and speaker is None and len(current["text"]) < 500:
            current["text"] += " " + seg_text
            current["end"] = seg_end
        else:
            current["text"] = normalize_text(current["text"])
            if current["text"]:
                merged.append(current)
            current = {"text": seg_text, "start": seg_start, "end": seg_end}
            if speaker:
                current["speaker"] = speaker

    # Don't forget the last one
    current["text"] = normalize_text(current["text"])
    if current["text"]:
        merged.append(current)

    return merged


# --- Time Windowing ---

def create_time_windows(utterances: list[dict], window_seconds: int = 300) -> list[dict]:
    """Group utterances into fixed time windows for analysis."""
    if not utterances:
        return []

    windows = []
    current_window = {
        "window_index": 0,
        "start_time": 0,
        "end_time": window_seconds,
        "utterances": [],
        "text": "",
        "word_count": 0,
    }

    for utt in utterances:
        utt_start = utt.get("start", 0)

        # Check if we need a new window
        while utt_start >= current_window["end_time"]:
            # Finalize current window
            current_window["text"] = " ".join(
                u["text"] for u in current_window["utterances"]
            )
            current_window["word_count"] = len(current_window["text"].split())
            if current_window["utterances"]:
                windows.append(current_window)

            # Start new window
            next_idx = current_window["window_index"] + 1
            current_window = {
                "window_index": next_idx,
                "start_time": next_idx * window_seconds,
                "end_time": (next_idx + 1) * window_seconds,
                "utterances": [],
                "text": "",
                "word_count": 0,
            }

        current_window["utterances"].append(utt)

    # Finalize last window
    current_window["text"] = " ".join(u["text"] for u in current_window["utterances"])
    current_window["word_count"] = len(current_window["text"].split())
    if current_window["utterances"]:
        windows.append(current_window)

    return windows


def process_transcript(transcript_path: str, window_seconds: int = 300) -> dict | None:
    """Process a single transcript file into analysis-ready segments."""
    with open(transcript_path) as f:
        doc = json.load(f)

    segments = doc.get("segments", [])
    if not segments:
        return None

    # Step 1: Merge fragments into utterances
    utterances = merge_fragments(segments)

    # Step 2: Create time windows
    windows = create_time_windows(utterances, window_seconds)

    # Build output document
    output = {
        "video_id": doc["video_id"],
        "channel_name": doc.get("channel_name", ""),
        "persona_type": doc.get("persona_type", ""),
        "title": doc.get("title", ""),
        "publish_date": doc.get("publish_date", ""),
        "transcript_source": doc.get("transcript_source", ""),
        "total_utterances": len(utterances),
        "total_windows": len(windows),
        "total_words": sum(w["word_count"] for w in windows),
        "window_seconds": window_seconds,
        "windows": [
            {
                "window_index": w["window_index"],
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "word_count": w["word_count"],
                "text": w["text"],
            }
            for w in windows
        ],
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Clean and segment transcripts")
    parser.add_argument("--input-dir", default="data/transcripts")
    parser.add_argument("--output-dir", default="data/segments")
    parser.add_argument("--index-output", default="data/segment_index.csv")
    parser.add_argument("--window-seconds", type=int, default=300,
                        help="Time window size in seconds (default: 300 = 5 min)")
    parser.add_argument("--min-words", type=int, default=50,
                        help="Minimum words per window to keep (default: 50)")
    parser.add_argument("--channel", help="Process only this channel (for testing)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Find all transcript files
    transcript_files = sorted(Path(args.input_dir).glob("*.json"))
    print(f"Found {len(transcript_files)} transcript files")

    index_rows = []
    total_windows = 0

    for tf in tqdm(transcript_files, desc="Segmenting"):
        result = process_transcript(str(tf), args.window_seconds)
        if result is None:
            continue

        if args.channel and result["channel_name"] != args.channel:
            continue

        # Filter out windows that are too short
        result["windows"] = [
            w for w in result["windows"] if w["word_count"] >= args.min_words
        ]
        result["total_windows"] = len(result["windows"])

        if not result["windows"]:
            continue

        # Save segmented file
        output_path = os.path.join(args.output_dir, f"{result['video_id']}_segments.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        total_windows += result["total_windows"]

        index_rows.append({
            "video_id": result["video_id"],
            "channel_name": result["channel_name"],
            "persona_type": result["persona_type"],
            "total_windows": result["total_windows"],
            "total_words": result["total_words"],
            "transcript_source": result["transcript_source"],
        })

    # Write index
    with open(args.index_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "channel_name", "persona_type",
            "total_windows", "total_words", "transcript_source",
        ])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n{'=' * 60}")
    print(f"Segmentation complete")
    print(f"  Videos processed: {len(index_rows)}")
    print(f"  Total windows:    {total_windows}")
    print(f"  Window size:      {args.window_seconds}s ({args.window_seconds // 60} min)")
    print(f"Index: {args.index_output}")


if __name__ == "__main__":
    main()
