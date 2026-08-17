"""
Stage 5: Base-Rate Tables
===========================
Produces the four descriptive outputs the brief asks for:

  1. Path A / Path B base rates as a share of all appointments
  2. The same, conditioned on pubs_pre_move (split at 10 and 20)
  3. Which non-US countries feed US ladder-rank hiring, Hong Kong called out
  4. Composition of prior positions overall

Everything is descriptive. Proportions carry Wilson 95% intervals because
several cells will be small, and a bare percentage on n=7 invites more
confidence than the data supports. No causal quantity is estimated anywhere.

Denominators
------------
The base-rate denominator is people with a CODED prior position, not the whole
frame. People whose prior position could not be established are reported
separately as a coverage figure rather than folded into the denominator, where
they would silently deflate every rate.

Usage:
    python -m stage5_analysis.base_rates
    python -m stage5_analysis.base_rates --exclude-flagged
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from common import DATA_DIR

CODED_PATH = DATA_DIR / "coded_trajectories.csv"
TABLES_DIR = DATA_DIR / "tables"

PUB_BINS = [0, 10, 20, math.inf]
PUB_LABELS = ["<10", "10-19", "20+"]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval; behaves sensibly at 0 and at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pct_row(label: str, k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        "group": label,
        "n": n,
        "count": k,
        "share": round(k / n, 4) if n else None,
        "ci_low": round(lo, 4) if n else None,
        "ci_high": round(hi, 4) if n else None,
    }


def load(path: Path, exclude_flagged: bool) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No coded file at {path}. Run stage 3 first.")
    df = pd.read_csv(path, dtype=str).fillna("")
    total = len(df)

    coded = df[df["prior_rank_class"] != ""].copy()
    if exclude_flagged:
        before = len(coded)
        coded = coded[coded["needs_manual_review"] != "1"]
        print(f"  excluded {before - len(coded)} rows flagged for manual review")

    for col in ("pubs_pre_move", "appointment_year", "gap_years",
                "prior_duration_years", "phd_year"):
        coded[col] = pd.to_numeric(coded[col], errors="coerce")
    for col in ("path_a", "path_b_strict", "path_b_broad"):
        coded[col] = pd.to_numeric(coded[col], errors="coerce").fillna(0).astype(int)

    coded["prior_country"] = coded["prior_country"].str.upper()
    print(f"  frame rows: {total}; with a coded prior position: {len(coded)} "
          f"({len(coded) / total:.1%})" if total else "")
    return coded


def table_composition(df: pd.DataFrame) -> pd.DataFrame:
    """What did people hold immediately before, overall?"""
    n = len(df)
    rows = []
    for cls, count in df["prior_rank_class"].value_counts().items():
        rows.append(pct_row(cls, int(count), n))
    us = df["prior_country"] == "US"
    rows.append(pct_row("-- prior position in US", int(us.sum()), n))
    rows.append(pct_row("-- prior position outside US", int((~us & (df["prior_country"] != "")).sum()), n))
    return pd.DataFrame(rows)


def table_base_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Headline Path A / Path B rates, overall and by field."""
    rows = []
    n = len(df)
    rows.append(pct_row("Path A (non-US ladder -> US ladder) | ALL", int(df["path_a"].sum()), n))
    rows.append(pct_row("Path B strict (US non-ladder) | ALL", int(df["path_b_strict"].sum()), n))
    rows.append(pct_row("Path B broad (incl. extended postdoc) | ALL", int(df["path_b_broad"].sum()), n))

    std_postdoc = (
        (df["prior_country"] == "US")
        & (df["prior_rank_class"] == "postdoc")
        & (df["path_b_broad"] == 0)
    )
    rows.append(pct_row("Reference: standard US postdoc | ALL", int(std_postdoc.sum()), n))

    for field, sub in df.groupby("field"):
        m = len(sub)
        rows.append(pct_row(f"Path A | {field}", int(sub["path_a"].sum()), m))
        rows.append(pct_row(f"Path B strict | {field}", int(sub["path_b_strict"].sum()), m))
        rows.append(pct_row(f"Path B broad | {field}", int(sub["path_b_broad"].sum()), m))
    return pd.DataFrame(rows)


def table_by_pubs(df: pd.DataFrame) -> pd.DataFrame:
    """
    The crux: does output close the gap for the non-ladder path?

    Read DOWN each path within a publication band, and ACROSS bands within a
    path. This is a cross-tabulation, not a controlled comparison -- people
    with more pre-move publications differ on many things besides path.
    """
    d = df[df["pubs_pre_move"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["pub_band"] = pd.cut(d["pubs_pre_move"], bins=PUB_BINS, labels=PUB_LABELS,
                           right=False, include_lowest=True)
    rows = []
    for band in PUB_LABELS:
        sub = d[d["pub_band"] == band]
        m = len(sub)
        rows.append(pct_row(f"Path A | pubs {band}", int(sub["path_a"].sum()), m))
        rows.append(pct_row(f"Path B strict | pubs {band}", int(sub["path_b_strict"].sum()), m))
        rows.append(pct_row(f"Path B broad | pubs {band}", int(sub["path_b_broad"].sum()), m))
    missing = len(df) - len(d)
    if missing:
        rows.append({"group": f"(pubs_pre_move missing for {missing} people)",
                     "n": missing, "count": None, "share": None,
                     "ci_low": None, "ci_high": None})
    return pd.DataFrame(rows)


def table_feeder_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Which non-US systems actually feed US ladder-rank hiring."""
    ladder_abroad = df[(df["prior_rank_class"] == "ladder")
                       & (df["prior_country"] != "US")
                       & (df["prior_country"] != "")]
    n_all = len(df)
    rows = []
    for country, count in ladder_abroad["prior_country"].value_counts().items():
        row = pct_row(country, int(count), n_all)
        row["share_of_path_a"] = (
            round(count / len(ladder_abroad), 4) if len(ladder_abroad) else None
        )
        rows.append(row)
    out = pd.DataFrame(rows)

    hk = int((ladder_abroad["prior_country"] == "HK").sum())
    print(f"\n  Hong Kong specifically: {hk} of {len(ladder_abroad)} Path A moves, "
          f"{hk}/{n_all} = {hk / n_all:.1%} of all appointments" if n_all else "")
    if hk and hk < 10:
        print(f"  [caution] n={hk} is too small to characterize the HK pipeline; "
              "report the count, not a rate.")
    return out


def table_gap_years(df: pd.DataFrame) -> pd.DataFrame:
    """Time served in the prior post, by path."""
    rows = []
    for label, mask in (
        ("Path A", df["path_a"] == 1),
        ("Path B strict", df["path_b_strict"] == 1),
        ("Path B broad", df["path_b_broad"] == 1),
        ("All others", (df["path_a"] == 0) & (df["path_b_broad"] == 0)),
    ):
        sub = df[mask]["gap_years"].dropna()
        rows.append(
            {
                "group": label,
                "n": len(sub),
                "median_gap_years": sub.median() if len(sub) else None,
                "mean_gap_years": round(sub.mean(), 2) if len(sub) else None,
                "p25": sub.quantile(0.25) if len(sub) else None,
                "p75": sub.quantile(0.75) if len(sub) else None,
            }
        )
    return pd.DataFrame(rows)


def show(title: str, df: pd.DataFrame, path: Path = None):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    if df is None or df.empty:
        print("(no data)")
        return
    printable = df.copy()
    for col in ("share", "ci_low", "ci_high", "share_of_path_a"):
        if col in printable:
            printable[col] = printable[col].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) else ""
            )
    print(printable.to_string(index=False))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coded", default=str(CODED_PATH))
    ap.add_argument("--tables-dir", default=str(TABLES_DIR))
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop rows flagged needs_manual_review")
    args = ap.parse_args()

    tables = Path(args.tables_dir)
    print("Loading coded trajectories...")
    df = load(Path(args.coded), args.exclude_flagged)
    if df.empty:
        raise SystemExit("No coded prior positions to analyze.")

    show("1. COMPOSITION OF PRIOR POSITIONS",
         table_composition(df), tables / "composition.csv")
    show("2. BASE RATES: PATH A vs PATH B",
         table_base_rates(df), tables / "base_rates.csv")
    show("3. CONDITIONED ON PRE-MOVE PUBLICATIONS",
         table_by_pubs(df), tables / "base_rates_by_pubs.csv")
    show("4. FEEDER COUNTRIES INTO US LADDER-RANK POSTS",
         table_feeder_countries(df), tables / "feeder_countries.csv")
    show("5. YEARS IN PRIOR POSITION",
         table_gap_years(df), tables / "gap_years.csv")

    print(f"\nTables written to {tables}/")
    print(
        "\nReminder: these are base rates over people who ALREADY hold a US "
        "ladder-rank job.\nEveryone who attempted either path and did not land "
        "one is invisible here, so no\nfigure below is a probability of success."
    )


if __name__ == "__main__":
    main()
