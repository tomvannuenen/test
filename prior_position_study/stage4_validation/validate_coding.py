"""
Stage 4: Validate the Automated Coding
========================================
Workflow:
  1. export-sample     -- draw a random 50 for hand-coding
  2. (human codes the exported CSV's blank columns from CVs / faculty pages)
  3. compute-agreement -- percent agreement + Cohen's kappa on prior_rank_class

The brief sets the bar at ~85% agreement on `prior_rank_class`. Below that,
the country title mapping needs work before any base rate is trustworthy, and
compute-agreement says so explicitly in its verdict line.

Sampling note
-------------
The default sample is a SIMPLE RANDOM one, as the brief specifies, because
that is what yields an unbiased estimate of the overall agreement rate.
`--stratify confidence` oversamples low-confidence codings instead; that is a
diagnostic for locating which national systems are misfiring, and its
agreement rate is biased downward by construction. Do not report it as the
headline number.

Usage:
    python -m stage4_validation.validate_coding export-sample --n 50
    python -m stage4_validation.validate_coding export-sample --n 50 --stratify confidence
    python -m stage4_validation.validate_coding compute-agreement \
        --human-file data/validation_sample_coded.csv
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA_DIR

CODED_PATH = DATA_DIR / "coded_trajectories.csv"
SAMPLE_PATH = DATA_DIR / "validation_sample.csv"

# Columns the human fills in. Left blank on export so the coder is not
# anchored by the machine's answer.
HUMAN_COLUMNS = [
    "human_prior_rank_class",
    "human_prior_country",
    "human_prior_institution",
    "human_prior_role_title",
    "human_notes",
]

# Context shown to the human coder.
CONTEXT_COLUMNS = [
    "person_id",
    "display_name",
    "field",
    "current_institution",
    "appointment_year",
    "prior_institution",
    "prior_role_title",
    "prior_start_year",
    "prior_end_year",
]

RANK_CLASSES = ["ladder", "non-ladder-academic", "postdoc", "industry", "other"]


def read_coded(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"No coded file at {path}. Run stage 3 first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def export_sample(coded_path: Path, n: int, output: Path, seed: int, stratify: str):
    rows = [r for r in read_coded(coded_path) if r.get("prior_rank_class")]
    if not rows:
        raise SystemExit("No coded rows with a prior_rank_class to sample from.")

    rng = random.Random(seed)

    if stratify == "confidence":
        buckets = defaultdict(list)
        for r in rows:
            buckets[r.get("coding_confidence") or "none"].append(r)
        # Weight uncertain codings up: they are where the mapping fails.
        weights = {"none": 0.35, "low": 0.35, "medium": 0.20, "high": 0.10}
        sample = []
        for bucket, weight in weights.items():
            pool = buckets.get(bucket, [])
            take = min(len(pool), round(n * weight))
            sample.extend(rng.sample(pool, take))
        # Top up from anywhere if a bucket was short.
        if len(sample) < n:
            remaining = [r for r in rows if r not in sample]
            rng.shuffle(remaining)
            sample.extend(remaining[: n - len(sample)])
        sample = sample[:n]
    else:
        sample = rng.sample(rows, min(n, len(rows)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONTEXT_COLUMNS + HUMAN_COLUMNS)
        w.writeheader()
        for r in sample:
            row = {c: r.get(c, "") for c in CONTEXT_COLUMNS}
            row.update({c: "" for c in HUMAN_COLUMNS})
            w.writerow(row)

    print(f"Exported {len(sample)} rows for hand-coding -> {output}")
    print(f"  sampling: {'stratified by confidence (diagnostic)' if stratify else 'simple random'}")
    print(f"  seed: {seed}")
    print(f"\nFill in {HUMAN_COLUMNS[0]} using one of: {', '.join(RANK_CLASSES)}")
    print("Machine codings are deliberately withheld from this file to avoid anchoring.")


def cohens_kappa(a: list, b: list) -> float:
    """Unweighted Cohen's kappa; implemented directly to keep deps light."""
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def compute_agreement(coded_path: Path, human_path: Path, threshold: float):
    if not human_path.exists():
        raise SystemExit(f"No hand-coded file at {human_path}.")

    machine = {r["person_id"]: r for r in read_coded(coded_path)}
    with open(human_path) as f:
        human_rows = list(csv.DictReader(f))

    pairs, skipped = [], 0
    for h in human_rows:
        hv = (h.get("human_prior_rank_class") or "").strip().lower()
        m = machine.get(h["person_id"])
        if not hv or not m:
            skipped += 1
            continue
        if hv not in RANK_CLASSES:
            print(f"  [warn] unrecognized human code {hv!r} for {h['person_id']}")
            skipped += 1
            continue
        # Exclude rows already hand-coded upstream: comparing a hand coding
        # to itself would inflate agreement.
        if m.get("coding_source") == "manual":
            skipped += 1
            continue
        pairs.append((m["prior_rank_class"], hv, m, h))

    if not pairs:
        raise SystemExit("No comparable pairs. Has the sample been hand-coded?")

    mach = [p[0] for p in pairs]
    hum = [p[1] for p in pairs]
    agree = sum(1 for x, y in zip(mach, hum) if x == y)
    rate = agree / len(pairs)
    kappa = cohens_kappa(mach, hum)

    print("=" * 68)
    print("VALIDATION: prior_rank_class")
    print("=" * 68)
    print(f"n compared        : {len(pairs)}   (skipped {skipped})")
    print(f"percent agreement : {rate:.1%}  ({agree}/{len(pairs)})")
    print(f"Cohen's kappa     : {kappa:.3f}")
    print()

    print("Per-class recall (human label as truth):")
    by_class = defaultdict(lambda: [0, 0])
    for m, h, _, _ in pairs:
        by_class[h][1] += 1
        if m == h:
            by_class[h][0] += 1
    for cls in RANK_CLASSES:
        hit, total = by_class.get(cls, [0, 0])
        if total:
            print(f"  {cls:<22} {hit}/{total}  ({hit / total:.0%})")
    print()

    disagreements = [(m, h, mr, hr) for m, h, mr, hr in pairs if m != h]
    if disagreements:
        print(f"Disagreements ({len(disagreements)}):")
        for m, h, mr, hr in disagreements:
            print(
                f"  {mr.get('display_name', '?'):<28} "
                f"machine={m:<20} human={h:<20}\n"
                f"      title={mr.get('prior_role_title', '')!r} "
                f"country={mr.get('prior_country', '')} "
                f"system={mr.get('coding_system', '')}"
            )
        print()

        # Where the mapping fails, by national system.
        by_system = Counter(mr.get("coding_system", "?") for _, _, mr, _ in disagreements)
        print("Disagreements by coding system:")
        for system, count in by_system.most_common():
            print(f"  {system:<24} {count}")
        print()

    print("=" * 68)
    if rate >= threshold:
        print(f"PASS: {rate:.1%} >= {threshold:.0%}. Base rates can be reported.")
    else:
        print(
            f"FAIL: {rate:.1%} < {threshold:.0%}. Per the brief, the title mapping "
            "needs work before\n      any base rate is trustworthy. Use the "
            "by-system breakdown above to target it."
        )
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export-sample")
    e.add_argument("--coded", default=str(CODED_PATH))
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--output", default=str(SAMPLE_PATH))
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--stratify", choices=["confidence"],
                   help="diagnostic mode; biases the agreement rate downward")

    c = sub.add_parser("compute-agreement")
    c.add_argument("--coded", default=str(CODED_PATH))
    c.add_argument("--human-file", required=True)
    c.add_argument("--threshold", type=float, default=0.85)

    args = ap.parse_args()

    if args.command == "export-sample":
        export_sample(Path(args.coded), args.n, Path(args.output), args.seed, args.stratify)
    else:
        compute_agreement(Path(args.coded), Path(args.human_file), args.threshold)


if __name__ == "__main__":
    main()
