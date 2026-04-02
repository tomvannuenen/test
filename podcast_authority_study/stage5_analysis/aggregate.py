"""
Stage 5b: Aggregate Coded Data
================================
Reads all coded transcript files and produces summary statistics
for comparative analysis across channels and persona types.

Usage:
    python -m stage5_analysis.aggregate

Output:
    data/analysis/code_frequencies.csv
    data/analysis/channel_profiles.csv
    data/analysis/persona_comparison.csv
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_coded_files(coded_dir: str) -> list[dict]:
    """Load all coded JSON files."""
    docs = []
    for f in sorted(Path(coded_dir).glob("*_coded.json")):
        with open(f) as fh:
            docs.append(json.load(fh))
    return docs


def extract_code_frequencies(docs: list[dict]) -> dict:
    """Count code frequencies across all dimensions, grouped by channel and persona."""
    # Structure: {dimension: {code: {channel: count, persona: count, total: count}}}
    by_channel = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    by_persona = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals = defaultdict(lambda: defaultdict(int))
    window_counts = defaultdict(int)
    persona_window_counts = defaultdict(int)

    dimensions = [
        "self_positioning", "enemy_construction", "epistemic_style",
        "power_model", "authorization_move",
    ]

    for doc in docs:
        channel = doc.get("channel_name", "unknown")
        persona = doc.get("persona_type", "unknown")

        for window in doc.get("coded_windows", []):
            coding = window.get("coding", {})
            window_counts[channel] += 1
            persona_window_counts[persona] += 1

            for dim in dimensions:
                codes = coding.get(dim, [])
                if isinstance(codes, list):
                    for code in codes:
                        by_channel[dim][code][channel] += 1
                        by_persona[dim][code][persona] += 1
                        totals[dim][code] += 1

    return {
        "by_channel": by_channel,
        "by_persona": by_persona,
        "totals": totals,
        "window_counts": dict(window_counts),
        "persona_window_counts": dict(persona_window_counts),
    }


def write_code_frequencies(freq_data: dict, output_path: str):
    """Write overall code frequency table."""
    rows = []
    for dim, codes in sorted(freq_data["totals"].items()):
        for code, count in sorted(codes.items(), key=lambda x: -x[1]):
            rows.append({
                "dimension": dim,
                "code": code,
                "total_count": count,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "code", "total_count"])
        writer.writeheader()
        writer.writerows(rows)


def write_channel_profiles(freq_data: dict, output_path: str):
    """Write per-channel code profiles (normalized by window count)."""
    rows = []
    channels = sorted(freq_data["window_counts"].keys())

    for channel in channels:
        n_windows = freq_data["window_counts"][channel]
        row = {"channel": channel, "total_windows": n_windows}

        for dim, codes in sorted(freq_data["by_channel"].items()):
            for code, channel_counts in codes.items():
                count = channel_counts.get(channel, 0)
                # Normalized frequency (per window)
                row[f"{dim}__{code}"] = round(count / n_windows, 3) if n_windows else 0

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_persona_comparison(freq_data: dict, output_path: str):
    """Write persona-type comparison table."""
    rows = []
    personas = sorted(freq_data["persona_window_counts"].keys())

    for persona in personas:
        n_windows = freq_data["persona_window_counts"][persona]
        row = {"persona_type": persona, "total_windows": n_windows}

        for dim, codes in sorted(freq_data["by_persona"].items()):
            for code, persona_counts in codes.items():
                count = persona_counts.get(persona, 0)
                row[f"{dim}__{code}"] = round(count / n_windows, 3) if n_windows else 0

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Stage 5b: Aggregate coded data")
    parser.add_argument("--coded-dir", default="data/coded")
    parser.add_argument("--output-dir", default="data/analysis")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    docs = load_coded_files(args.coded_dir)
    print(f"Loaded {len(docs)} coded files")

    if not docs:
        print("No coded files found. Run stage5_analysis.llm_coder first.")
        return

    freq_data = extract_code_frequencies(docs)

    write_code_frequencies(
        freq_data, os.path.join(args.output_dir, "code_frequencies.csv")
    )
    write_channel_profiles(
        freq_data, os.path.join(args.output_dir, "channel_profiles.csv")
    )
    write_persona_comparison(
        freq_data, os.path.join(args.output_dir, "persona_comparison.csv")
    )

    print(f"\nAnalysis outputs written to {args.output_dir}/")
    print(f"  code_frequencies.csv  - overall code counts")
    print(f"  channel_profiles.csv  - per-channel normalized profiles")
    print(f"  persona_comparison.csv - persona type comparison")

    # Print quick summary
    print(f"\n{'=' * 60}")
    total_windows = sum(freq_data["window_counts"].values())
    print(f"Total windows coded: {total_windows}")
    print(f"Channels: {len(freq_data['window_counts'])}")
    print(f"Persona types: {len(freq_data['persona_window_counts'])}")

    print("\nTop codes by dimension:")
    for dim, codes in sorted(freq_data["totals"].items()):
        top = sorted(codes.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{c}({n})" for c, n in top)
        print(f"  {dim}: {top_str}")


if __name__ == "__main__":
    main()
