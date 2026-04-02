"""
Stage 1: Define the Corpus
===========================
Loads channel list from config, validates channel IDs against YouTube API,
and produces a canonical corpus manifest with influence metadata.

Usage:
    python -m stage1_corpus.define_corpus --api-key YOUR_KEY

Output:
    data/corpus_manifest.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False


def load_channel_config(config_path: str = "config/channels.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_channels(youtube, channels: list[dict]) -> list[dict]:
    """Validate channel IDs and pull live metadata from YouTube API."""
    validated = []

    for ch in channels:
        channel_id = ch["youtube_channel_id"]
        try:
            resp = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id,
            ).execute()

            if not resp.get("items"):
                print(f"WARNING: No channel found for ID {channel_id} ({ch['name']})")
                ch["api_status"] = "not_found"
                ch["actual_subscriber_count"] = None
                ch["uploads_playlist_id"] = None
            else:
                item = resp["items"][0]
                stats = item["statistics"]
                ch["api_status"] = "found"
                ch["actual_subscriber_count"] = int(stats.get("subscriberCount", 0))
                ch["total_video_count"] = int(stats.get("videoCount", 0))
                ch["uploads_playlist_id"] = item["contentDetails"]["relatedPlaylists"]["uploads"]
                ch["channel_title"] = item["snippet"]["title"]
                ch["channel_description"] = item["snippet"].get("description", "")[:500]
                print(f"  OK: {ch['name']} -> {ch['actual_subscriber_count']:,} subscribers")

        except HttpError as e:
            print(f"ERROR: API error for {ch['name']}: {e}")
            ch["api_status"] = "error"
            ch["actual_subscriber_count"] = None
            ch["uploads_playlist_id"] = None

        validated.append(ch)

    return validated


def write_manifest(channels: list[dict], output_path: str = "data/corpus_manifest.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name", "youtube_channel_id", "persona_type", "subscriber_estimate",
        "actual_subscriber_count", "total_video_count", "uploads_playlist_id",
        "channel_title", "api_status", "justification", "platform_note",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ch in channels:
            writer.writerow(ch)

    print(f"\nManifest written to {output_path} ({len(channels)} channels)")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Define and validate corpus")
    parser.add_argument("--api-key", required=True, help="YouTube Data API v3 key")
    parser.add_argument("--config", default="config/channels.yaml", help="Channel config YAML")
    parser.add_argument("--output", default="data/corpus_manifest.csv", help="Output manifest path")
    parser.add_argument("--skip-validation", action="store_true", help="Skip API validation (offline mode)")
    args = parser.parse_args()

    config = load_channel_config(args.config)
    channels = config["channels"]
    print(f"Loaded {len(channels)} channels from {args.config}")

    # Print corpus summary by persona type
    from collections import Counter
    type_counts = Counter(ch["persona_type"] for ch in channels)
    print("\nCorpus composition by persona type:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")

    if args.skip_validation:
        print("\nSkipping API validation (offline mode)")
        for ch in channels:
            ch["api_status"] = "skipped"
            ch["actual_subscriber_count"] = ch.get("subscriber_estimate")
            ch["uploads_playlist_id"] = None
    else:
        if not HAS_GOOGLE_API:
            print("ERROR: google-api-python-client not installed. Use --skip-validation or pip install google-api-python-client")
            sys.exit(1)
        print("\nValidating channels against YouTube API...")
        youtube = build("youtube", "v3", developerKey=args.api_key)
        channels = validate_channels(youtube, channels)

    write_manifest(channels, args.output)

    # Summary stats
    found = sum(1 for ch in channels if ch.get("api_status") == "found")
    not_found = sum(1 for ch in channels if ch.get("api_status") == "not_found")
    errors = sum(1 for ch in channels if ch.get("api_status") == "error")
    if not args.skip_validation:
        print(f"\nValidation summary: {found} found, {not_found} not found, {errors} errors")


if __name__ == "__main__":
    main()
