#!/usr/bin/env python3
"""
Filter Taiko no Tatsujin songs from the past ~10 years for "angel" and "dream"
related terms, including Japanese equivalents.

Source: taikowiki/taiko-song-database (GitHub)
Coverage: Songs playable on Nijiiro arcade; may be missing some older/removed/console-only items.
"""

import json
import csv
import re
from datetime import datetime, timezone

# --- Configuration ---
DATABASE_FILE = "database_raw.json"
CUTOFF_DATE_MS = int(datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
OUTPUT_CSV = "taiko_angel_dream_songs.csv"
OUTPUT_TXT = "taiko_angel_dream_songs.txt"

# Search terms (case-insensitive for Latin; exact for CJK)
ANGEL_TERMS = [
    # English / romaji
    r"angel",
    r"tenshi",
    r"enjeru",
    # Japanese
    "天使",
    "エンジェル",
    "えんじぇる",
    "てんし",
]

DREAM_TERMS = [
    # English / romaji
    r"dream",
    r"yume",
    # Japanese
    "夢",
    "ドリーム",
    "どりーむ",
    "ゆめ",
]

ALL_TERMS = ANGEL_TERMS + DREAM_TERMS

def build_pattern(terms):
    """Build a single compiled regex that matches any of the terms (case-insensitive)."""
    escaped = [re.escape(t) for t in terms]
    return re.compile("|".join(escaped), re.IGNORECASE)

ANGEL_PATTERN = build_pattern(ANGEL_TERMS)
DREAM_PATTERN = build_pattern(DREAM_TERMS)
ANY_PATTERN = build_pattern(ALL_TERMS)

def searchable_text(song):
    """Concatenate all text fields we want to search."""
    fields = [
        song.get("title"),
        song.get("titleEn"),
        song.get("titleKo"),
        song.get("romaji"),
        song.get("aliasEn"),
        song.get("aliasKo"),
    ]
    artists = song.get("artists") or []
    if isinstance(artists, list):
        fields.extend(artists)
    elif isinstance(artists, str):
        fields.append(artists)
    return " ".join(f for f in fields if f)

def match_categories(text):
    """Return which keyword categories matched."""
    cats = []
    if ANGEL_PATTERN.search(text):
        cats.append("angel")
    if DREAM_PATTERN.search(text):
        cats.append("dream")
    return cats

def ts_to_date(ts_ms):
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

def fmt_genre(g):
    if isinstance(g, list):
        return ", ".join(g)
    return str(g) if g else ""

def fmt_versions(v):
    if isinstance(v, list):
        return ", ".join(v)
    return str(v) if v else ""

def fmt_artists(a):
    if isinstance(a, list):
        return " / ".join(a)
    return str(a) if a else ""

# --- Main ---
with open(DATABASE_FILE, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total songs in database: {len(data)}")

# Filter: added in the past ~10 years AND matching angel/dream terms
results = []
for song in data:
    added = song.get("addedDate")
    # Include songs with addedDate >= 2016-01-01, or songs with no date (unknown)
    if added is not None and added < CUTOFF_DATE_MS:
        continue

    text = searchable_text(song)
    cats = match_categories(text)
    if not cats:
        continue

    results.append({
        "addedDate": ts_to_date(added),
        "title": song.get("title", ""),
        "titleEn": song.get("titleEn", "") or "",
        "romaji": song.get("romaji", "") or "",
        "artists": fmt_artists(song.get("artists")),
        "genre": fmt_genre(song.get("genre")),
        "versions": fmt_versions(song.get("version")),
        "matchedTerms": ", ".join(cats),
        "isDeleted": "yes" if song.get("isDeleted") else "no",
    })

results.sort(key=lambda r: (r["addedDate"] or "9999", r["title"]))

print(f"Songs matching 'angel'/'dream' (added >= 2016-01-01): {len(results)}")
angel_count = sum(1 for r in results if "angel" in r["matchedTerms"])
dream_count = sum(1 for r in results if "dream" in r["matchedTerms"])
both_count = sum(1 for r in results if "angel" in r["matchedTerms"] and "dream" in r["matchedTerms"])
print(f"  angel: {angel_count}  |  dream: {dream_count}  |  both: {both_count}")

# Write CSV
fieldnames = ["addedDate", "title", "titleEn", "romaji", "artists", "genre", "versions", "matchedTerms", "isDeleted"]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)
print(f"\nCSV written to {OUTPUT_CSV}")

# Write human-readable text
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("Taiko no Tatsujin Songs — 'Angel' / 'Dream' Filter (added >= 2016-01-01)\n")
    f.write(f"Source: taikowiki/taiko-song-database  |  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    f.write(f"Total matches: {len(results)}  (angel: {angel_count}, dream: {dream_count}, both: {both_count})\n")
    f.write("=" * 90 + "\n\n")

    for i, r in enumerate(results, 1):
        f.write(f"[{i:3d}] {r['title']}")
        if r["titleEn"]:
            f.write(f"  ({r['titleEn']})")
        f.write("\n")
        f.write(f"      Added: {r['addedDate'] or '?'}  |  Match: {r['matchedTerms']}  |  Deleted: {r['isDeleted']}\n")
        if r["romaji"]:
            f.write(f"      Romaji: {r['romaji']}\n")
        f.write(f"      Artists: {r['artists']}\n")
        f.write(f"      Genre: {r['genre']}  |  Versions: {r['versions']}\n")
        f.write("\n")

print(f"TXT written to {OUTPUT_TXT}")
