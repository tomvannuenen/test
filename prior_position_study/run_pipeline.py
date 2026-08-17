"""
Prior-Position Study - Pipeline Orchestrator
==============================================
Runs the full pipeline or individual stages.

Usage:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage 1 --field communication
    python run_pipeline.py --stage 3
    python run_pipeline.py --stage 4 --human-file data/validation_sample_coded.csv
    python run_pipeline.py --selftest         # offline, fixture-driven

Stages:
    1  build the sampling frame from OpenAlex          [needs network]
    2  enrich with ORCID employments + pub counts      [needs network]
    3  code trajectories -> tidy CSV                   [offline]
    4  export validation sample / compute agreement    [offline]
    5  base-rate tables                                [offline]

Stages 1-2 require outbound access to api.openalex.org and pub.orcid.org.
Where egress policy blocks those hosts the stage exits with a clear message
rather than retrying; run it from an environment where they are permitted.

Environment variables:
    ORCID_ACCESS_TOKEN   optional; only needed for ORCID's search endpoint
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run(cmd: list) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=BASE_DIR)


def run_stage(stage: str, args) -> int:
    py = sys.executable

    if stage == "1":
        cmd = [py, "-m", "stage1_frame.build_frame"]
        if args.field:
            cmd += ["--field", args.field]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.skip_verify:
            cmd += ["--skip-verify"]

    elif stage == "2":
        cmd = [py, "-m", "stage2_orcid.fetch_employments"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

    elif stage == "3":
        cmd = [py, "-m", "stage3_coding.code_trajectories"]
        if args.input:
            cmd += ["--input", args.input]
        if args.output:
            cmd += ["--output", args.output]
        if args.merge_manual:
            cmd += ["--merge-manual", args.merge_manual]

    elif stage == "4":
        if args.human_file:
            cmd = [py, "-m", "stage4_validation.validate_coding",
                   "compute-agreement", "--human-file", args.human_file]
        else:
            cmd = [py, "-m", "stage4_validation.validate_coding",
                   "export-sample", "--n", str(args.n)]
            if args.stratify:
                cmd += ["--stratify", args.stratify]

    elif stage == "5":
        cmd = [py, "-m", "stage5_analysis.base_rates"]
        if args.exclude_flagged:
            cmd += ["--exclude-flagged"]

    else:
        raise SystemExit(f"Unknown stage {stage}")

    return run(cmd)


def simulate_human_coding(sample_path: Path, coded_path: Path, out_path: Path,
                          disagreement_rate: float = 0.12, seed: int = 3):
    """
    Fill a validation sample with machine codings, flipping a fraction of them.

    Used ONLY by --selftest, to prove the agreement machinery reports sensible
    numbers. It is not a substitute for hand-coding: it manufactures
    disagreements at a rate chosen for the test, and tells you nothing about
    whether the title mapping is right.
    """
    rng = random.Random(seed)
    classes = ["ladder", "non-ladder-academic", "postdoc", "industry", "other"]
    machine = {r["person_id"]: r for r in csv.DictReader(open(coded_path))}

    rows = list(csv.DictReader(open(sample_path)))
    for r in rows:
        m = machine.get(r["person_id"], {})
        truth = m.get("prior_rank_class", "")
        if truth and rng.random() < disagreement_rate:
            truth = rng.choice([c for c in classes if c != truth])
            r["human_notes"] = "simulated disagreement"
        r["human_prior_rank_class"] = truth
        r["human_prior_country"] = m.get("prior_country", "")
        r["human_prior_institution"] = m.get("prior_institution", "")
        r["human_prior_role_title"] = m.get("prior_role_title", "")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  simulated hand-coding -> {out_path}")


def selftest(args) -> int:
    """
    Exercise every offline stage against fixtures. No network required.

    Proves the coding, validation and analysis logic runs end to end and that
    the known coding hazards resolve correctly. It cannot validate the
    OpenAlex/ORCID request layer, which is only reachable with network access.
    """
    py = sys.executable
    data = BASE_DIR / "data"
    fixture = data / "enriched_fixture.jsonl"
    coded = data / "coded_fixture.csv"
    sample = data / "validation_sample_fixture.csv"
    human = data / "validation_sample_fixture_coded.csv"

    steps = [
        ("unit tests: title mapping + trajectories",
         [py, "-m", "tests.test_coding"]),
        ("build fixtures",
         [py, "-m", "tests.make_fixtures", "--output", str(fixture)]),
        ("stage 3: code trajectories",
         [py, "-m", "stage3_coding.code_trajectories",
          "--input", str(fixture), "--output", str(coded)]),
        ("stage 4a: export validation sample",
         [py, "-m", "stage4_validation.validate_coding", "export-sample",
          "--coded", str(coded), "--n", "50", "--output", str(sample)]),
    ]

    for label, cmd in steps:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        if run(cmd) != 0:
            print(f"\nSELFTEST FAILED at: {label}")
            return 1

    print(f"\n{'=' * 70}\nsimulate hand-coding (test scaffolding only)\n{'=' * 70}")
    simulate_human_coding(sample, coded, human)

    for label, cmd in [
        ("stage 4b: compute agreement",
         [py, "-m", "stage4_validation.validate_coding", "compute-agreement",
          "--coded", str(coded), "--human-file", str(human)]),
        ("stage 5: base-rate tables",
         [py, "-m", "stage5_analysis.base_rates", "--coded", str(coded),
          "--tables-dir", str(data / "tables_fixture")]),
    ]:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        if run(cmd) != 0:
            print(f"\nSELFTEST FAILED at: {label}")
            return 1

    print(
        "\n"
        + "=" * 70
        + "\nSELFTEST PASSED - all offline stages ran end to end.\n"
        + "=" * 70
        + "\nNote: every number produced above comes from SYNTHETIC fixture data\n"
        "and is substantively meaningless. Stages 1-2 (OpenAlex/ORCID) are not\n"
        "covered by this test because they require network access.\n"
    )
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "1", "2", "3", "4", "5"])
    ap.add_argument("--selftest", action="store_true",
                    help="run all offline stages against fixtures")
    ap.add_argument("--field", help="stage 1: restrict to one field")
    ap.add_argument("--limit", type=int, help="stages 1-2: cap records processed")
    ap.add_argument("--skip-verify", action="store_true",
                    help="stage 1: skip subfield ID verification")
    ap.add_argument("--input", help="stage 3: input jsonl")
    ap.add_argument("--output", help="stage 3: output csv")
    ap.add_argument("--merge-manual", help="stage 3: hand-coded CSV to overlay")
    ap.add_argument("--n", type=int, default=50, help="stage 4: validation sample size")
    ap.add_argument("--stratify", choices=["confidence"],
                    help="stage 4: diagnostic sampling mode")
    ap.add_argument("--human-file", help="stage 4: hand-coded file -> compute agreement")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="stage 5: drop rows flagged for manual review")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest(args))

    stages = ["1", "2", "3", "4", "5"] if args.stage == "all" else [args.stage]
    for stage in stages:
        rc = run_stage(stage, args)
        if rc != 0:
            print(f"\nStage {stage} exited {rc}; stopping.")
            raise SystemExit(rc)
    print("\nDone.")


if __name__ == "__main__":
    main()
