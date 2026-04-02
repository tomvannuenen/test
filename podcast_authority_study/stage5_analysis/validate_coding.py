"""
Stage 5c: Validate LLM Coding
================================
Tools for intercoder reliability between LLM codings and human codings.

Workflow:
  1. Export a random sample of segments for human coding
  2. Import human-coded CSV
  3. Compute agreement metrics (Cohen's kappa, percent agreement)

Usage:
    python -m stage5_analysis.validate_coding export-sample --n 50
    python -m stage5_analysis.validate_coding compute-agreement --human-file data/human_coded.csv
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import yaml


def load_coding_frame(frame_path: str = "config/coding_frame.yaml") -> dict:
    with open(frame_path) as f:
        return yaml.safe_load(f)


def export_sample(coded_dir: str, segments_dir: str, n: int, output_path: str, seed: int = 42):
    """Export a random sample of segments for human coding."""
    random.seed(seed)

    # Collect all coded windows with their text
    all_windows = []

    for coded_file in Path(coded_dir).glob("*_coded.json"):
        with open(coded_file) as f:
            coded_doc = json.load(f)

        video_id = coded_doc["video_id"]
        seg_file = Path(segments_dir) / f"{video_id}_segments.json"

        if not seg_file.exists():
            continue

        with open(seg_file) as f:
            seg_doc = json.load(f)

        # Build lookup for segment text
        text_by_idx = {w["window_index"]: w["text"] for w in seg_doc.get("windows", [])}

        for cw in coded_doc.get("coded_windows", []):
            idx = cw["window_index"]
            text = text_by_idx.get(idx, "")
            if not text:
                continue

            all_windows.append({
                "video_id": video_id,
                "channel_name": coded_doc.get("channel_name", ""),
                "persona_type": coded_doc.get("persona_type", ""),
                "window_index": idx,
                "text": text[:3000],  # Truncate for spreadsheet readability
                "llm_self_positioning": "|".join(cw.get("coding", {}).get("self_positioning", [])),
                "llm_enemy_construction": "|".join(cw.get("coding", {}).get("enemy_construction", [])),
                "llm_epistemic_style": "|".join(cw.get("coding", {}).get("epistemic_style", [])),
                "llm_power_model": "|".join(cw.get("coding", {}).get("power_model", [])),
                "llm_authorization_move": "|".join(cw.get("coding", {}).get("authorization_move", [])),
                "llm_confidence": cw.get("coding", {}).get("confidence", ""),
                # Empty columns for human coder
                "human_self_positioning": "",
                "human_enemy_construction": "",
                "human_epistemic_style": "",
                "human_power_model": "",
                "human_authorization_move": "",
                "human_confidence": "",
                "human_notes": "",
            })

    if len(all_windows) < n:
        print(f"WARNING: Only {len(all_windows)} windows available, requested {n}")
        sample = all_windows
    else:
        sample = random.sample(all_windows, n)

    # Stratify summary
    from collections import Counter
    personas = Counter(w["persona_type"] for w in sample)
    print(f"Sample composition: {dict(personas)}")

    fieldnames = list(sample[0].keys()) if sample else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)

    print(f"Exported {len(sample)} windows to {output_path}")


def compute_agreement(human_file: str, output_path: str):
    """Compute intercoder agreement between LLM and human codings."""
    with open(human_file) as f:
        rows = list(csv.DictReader(f))

    dimensions = [
        "self_positioning", "enemy_construction", "epistemic_style",
        "power_model", "authorization_move",
    ]

    results = []

    for dim in dimensions:
        llm_key = f"llm_{dim}"
        human_key = f"human_{dim}"

        agreements = 0
        total = 0
        llm_codes_all = []
        human_codes_all = []

        for row in rows:
            llm_codes = set(row.get(llm_key, "").split("|")) - {""}
            human_codes = set(row.get(human_key, "").split("|")) - {""}

            if not human_codes:
                continue  # Human hasn't coded this one

            total += 1

            # Jaccard similarity for set-valued codes
            if llm_codes == human_codes:
                agreements += 1

            # For kappa, we need binary per-code agreement
            all_codes = llm_codes | human_codes
            for code in all_codes:
                llm_codes_all.append(1 if code in llm_codes else 0)
                human_codes_all.append(1 if code in human_codes else 0)

        pct_exact = (agreements / total * 100) if total else 0

        # Cohen's kappa (binary)
        kappa = compute_cohens_kappa(llm_codes_all, human_codes_all) if llm_codes_all else None

        results.append({
            "dimension": dim,
            "n_coded": total,
            "exact_agreement_pct": round(pct_exact, 1),
            "cohens_kappa": round(kappa, 3) if kappa is not None else "N/A",
        })

        print(f"{dim}: {pct_exact:.1f}% exact agreement, kappa={kappa:.3f}" if kappa else f"{dim}: {pct_exact:.1f}% exact agreement")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "n_coded", "exact_agreement_pct", "cohens_kappa"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAgreement results: {output_path}")


def compute_cohens_kappa(rater1: list[int], rater2: list[int]) -> float | None:
    """Compute Cohen's kappa for two binary raters."""
    if not rater1 or len(rater1) != len(rater2):
        return None

    n = len(rater1)
    # Observed agreement
    agree = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agree / n

    # Expected agreement
    p1_yes = sum(rater1) / n
    p2_yes = sum(rater2) / n
    p_e = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def main():
    parser = argparse.ArgumentParser(description="Stage 5c: Validate coding")
    subparsers = parser.add_subparsers(dest="command")

    # Export sample
    exp = subparsers.add_parser("export-sample")
    exp.add_argument("--coded-dir", default="data/coded")
    exp.add_argument("--segments-dir", default="data/segments")
    exp.add_argument("--n", type=int, default=50)
    exp.add_argument("--output", default="data/validation_sample.csv")
    exp.add_argument("--seed", type=int, default=42)

    # Compute agreement
    agr = subparsers.add_parser("compute-agreement")
    agr.add_argument("--human-file", required=True)
    agr.add_argument("--output", default="data/analysis/intercoder_agreement.csv")

    args = parser.parse_args()

    if args.command == "export-sample":
        export_sample(args.coded_dir, args.segments_dir, args.n, args.output, args.seed)
    elif args.command == "compute-agreement":
        compute_agreement(args.human_file, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
