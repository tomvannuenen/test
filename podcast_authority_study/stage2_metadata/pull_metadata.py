"""
Stage 2: Pull Video Metadata
=============================
For each channel in the corpus manifest, retrieves all video metadata
within the configured date range.

Collects: video ID, title, description, publish date, duration,
          caption availability, view count, like count.

Usage:
    python -m stage2_metadata.pull_metadata --api-key YOUR_KEY

Output:
    data/video_metadata.csv
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def parse_duration(iso_duration: str) -> int:
    """Convert ISO 8601 duration (PT1H2M3S) to seconds."""
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or ""
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def get_uploads_playlist_videos(youtube, playlist_id: str, start_date: str, end_date: str) -> list[str]:
    """Get all video IDs from an uploads playlist within date range."""
    video_ids = []
    next_page_token = None
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    while True:
        try:
            resp = youtube.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            ).execute()
        except HttpError as e:
            print(f"  API error fetching playlist {playlist_id}: {e}")
            break

        for item in resp.get("items", []):
            pub_date_str = item["snippet"]["publishedAt"][:10]
            pub_dt = datetime.fromisoformat(pub_date_str)

            if start_dt <= pub_dt <= end_dt:
                video_ids.append(item["contentDetails"]["videoId"])
            elif pub_dt < start_dt:
                # Playlist is reverse-chronological; if we've gone past start, stop
                return video_ids

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)  # Rate limiting

    return video_ids


def get_video_details(youtube, video_ids: list[str]) -> list[dict]:
    """Get detailed metadata for a batch of video IDs."""
    details = []

    # YouTube API allows max 50 IDs per request
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            resp = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            print(f"  API error fetching video details: {e}")
            continue

        for item in resp.get("items", []):
            snippet = item["snippet"]
            content = item["contentDetails"]
            stats = item.get("statistics", {})

            details.append({
                "video_id": item["id"],
                "title": snippet["title"],
                "description": snippet.get("description", "")[:2000],
                "publish_date": snippet["publishedAt"][:10],
                "publish_datetime": snippet["publishedAt"],
                "duration_seconds": parse_duration(content.get("duration", "")),
                "caption_available": content.get("caption", "false") == "true",
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
            })

        time.sleep(0.1)

    return details


def load_manifest(manifest_path: str = "data/corpus_manifest.csv") -> list[dict]:
    with open(manifest_path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Pull video metadata")
    parser.add_argument("--api-key", required=True, help="YouTube Data API v3 key")
    parser.add_argument("--manifest", default="data/corpus_manifest.csv")
    parser.add_argument("--config", default="config/channels.yaml")
    parser.add_argument("--output", default="data/video_metadata.csv")
    parser.add_argument("--channel", help="Process only this channel name (for testing)")
    parser.add_argument("--min-duration", type=int, default=600,
                        help="Minimum video duration in seconds (default: 600 = 10 min)")
    args = parser.parse_args()

    config = load_channel_config(args.config)
    date_range = config["date_range"]
    manifest = load_manifest(args.manifest)

    youtube = build("youtube", "v3", developerKey=args.api_key)

    all_videos = []
    fieldnames = [
        "channel_name", "persona_type", "video_id", "title", "description",
        "publish_date", "publish_datetime", "duration_seconds",
        "caption_available", "view_count", "like_count", "comment_count",
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for ch in manifest:
        if args.channel and ch["name"] != args.channel:
            continue

        playlist_id = ch.get("uploads_playlist_id")
        if not playlist_id or playlist_id == "None":
            print(f"SKIP: {ch['name']} - no uploads playlist ID")
            continue

        print(f"\nProcessing: {ch['name']} ({ch['persona_type']})")

        video_ids = get_uploads_playlist_videos(
            youtube, playlist_id, date_range["start"], date_range["end"]
        )
        print(f"  Found {len(video_ids)} videos in date range")

        if not video_ids:
            continue

        details = get_video_details(youtube, video_ids)

        # Filter by minimum duration (skip clips, shorts)
        long_form = [v for v in details if v["duration_seconds"] >= args.min_duration]
        print(f"  {len(long_form)} videos >= {args.min_duration}s (filtered {len(details) - len(long_form)} short clips)")

        for v in long_form:
            v["channel_name"] = ch["name"]
            v["persona_type"] = ch["persona_type"]
            all_videos.append(v)

    # Write output
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in all_videos:
            writer.writerow(v)

    print(f"\n{'=' * 60}")
    print(f"Total videos collected: {len(all_videos)}")
    print(f"Output: {args.output}")

    # Summary by channel
    from collections import Counter
    by_channel = Counter(v["channel_name"] for v in all_videos)
    print("\nVideos per channel:")
    for name, count in sorted(by_channel.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


def load_channel_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
