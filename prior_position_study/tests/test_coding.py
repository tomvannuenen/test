"""
Tests: Title Mapping and Trajectory Reconstruction
====================================================
Asserts the handcrafted coding hazards in tests/make_fixtures.py resolve the
way the country title table says they should. These are the cases where naive
English-string matching gets the answer wrong, so a regression here means the
base rates are wrong.

Usage:
    python -m tests.test_coding
"""

import sys

from common import load_config
from stage3_coding.code_trajectories import TitleCoder, code_person, same_org
from tests.make_fixtures import handcrafted

PATH_FLAGS = ("path_a", "path_b_strict", "path_b_broad")


def check(results, label, condition, detail=""):
    results.append((label, bool(condition), detail))


def test_title_coder(results):
    """Direct title -> rank class assertions, independent of trajectories."""
    coder = TitleCoder(load_config("title_mapping.yaml"))

    cases = [
        # (title, country, org, expected_class, why)
        ("Lecturer", "GB", "University of Manchester", "ladder",
         "UK Lecturer is the ladder entry grade"),
        ("Lecturer", "US", "Boston University", "non-ladder-academic",
         "US Lecturer is teaching-track"),
        ("Senior Lecturer", "GB", "KCL", "ladder", "UK Senior Lecturer ~ US Associate"),
        ("Senior Lecturer", "SG", "NTU", "non-ladder-academic",
         "Singapore educator track inverts the UK reading"),
        ("Research Assistant Professor", "HK", "CityU", "non-ladder-academic",
         "HK RAP is a fixed-term non-ladder grade"),
        ("Assistant Professor", "HK", "HKUST", "ladder", "HK adopted the US ladder"),
        ("Research Fellow", "SG", "NUS", "postdoc",
         "Singapore Research Fellow is the standard postdoc title"),
        ("Assistant Professor (Teaching)", "US", "Yale", "non-ladder-academic",
         "parenthesised track marker"),
        ("Research Assistant Professor", "US", "Pitt", "non-ladder-academic",
         "US research-professor track is non-tenure-track"),
        ("Adjunkt", "DK", "KU", "ladder", "Danish adjunkt = assistant professor"),
        ("Adjunkt", "SE", "Umea", "non-ladder-academic",
         "Swedish adjunkt is a teaching post"),
        ("Universitetslektor", "SE", "Lund", "ladder",
         "Swedish universitetslektor is permanent ladder"),
        ("Profesor Asociado", "ES", "UPF", "non-ladder-academic",
         "Spanish asociado is an adjunct, not a US associate professor"),
        ("Charge de Recherche", "FR", "CNRS", "ladder",
         "permanent CNRS research post"),
        ("Maitre de Conferences", "FR", "Sorbonne", "ladder", "French ladder grade"),
        ("Universitair Hoofddocent", "NL", "UvA", "ladder", "NL associate professor"),
        ("Senior Lecturer", "AU", "University of Melbourne", "ladder",
         "Australian Level C is ladder"),
        ("Associate Lecturer", "AU", "University of Melbourne", "non-ladder-academic",
         "Australian Level A is the entry/teaching grade"),
        ("Research Fellow", "AU", "University of Sydney", "postdoc",
         "ANZ Research Fellow is the postdoc grade"),
        ("Research Professor", "KR", "SNU", "non-ladder-academic",
         "Korean research professor is contract"),
        ("Senior Researcher", "US", "Microsoft Research", "industry",
         "employer override beats the academic-sounding title"),
        ("Research Scientist", "US", "Google LLC", "industry", "employer override"),
        ("Visiting Assistant Professor", "US", "NYU", "non-ladder-academic",
         "universal visiting rule precedes the professorial rule"),
        ("Professor Emeritus", "US", "Michigan", "other", "emeritus is not an active post"),
        ("PhD Student", "US", "Berkeley", "other", "still in training"),
        ("Postdoctoral Scholar", "US", "Stanford", "postdoc", "plain postdoc"),
        ("Ricercatore a tempo determinato di tipo B", "IT", "Bologna", "ladder",
         "RTD-b is the Italian tenure track"),
        ("Assegnista di Ricerca", "IT", "Milano", "postdoc", "Italian postdoc"),
        ("Juniorprofessor", "DE", "TUM", "ladder", "W1 junior professorship"),
        ("Forsteamanuensis", "NO", "UiO", "ladder", "Norwegian associate professor"),
    ]

    for title, country, org, expected, why in cases:
        got = coder.code(title, country, org)
        check(
            results,
            f"title: {title!r} [{country}] -> {expected}",
            got["rank_class"] == expected,
            f"got {got['rank_class']} via system={got['system']} rule={got['rule']!r} ({why})",
        )

    # Duration heuristic on an ambiguous US research title.
    short = coder.code("Research Fellow", "US", "Johns Hopkins University", duration_years=2)
    long = coder.code("Research Fellow", "US", "Johns Hopkins University", duration_years=7)
    check(results, "duration heuristic: 2y Research Fellow -> postdoc",
          short["rank_class"] == "postdoc", f"got {short['rank_class']}")
    check(results, "duration heuristic: 7y Research Fellow -> non-ladder-academic",
          long["rank_class"] == "non-ladder-academic", f"got {long['rank_class']}")

    # An unmapped country must fall through to generic, not crash.
    fallback = coder.code("Assistant Professor", "ZZ", "Somewhere")
    check(results, "unmapped country falls back to generic",
          fallback["system"] == "generic" and fallback["confidence"] == "low",
          f"got system={fallback['system']} conf={fallback['confidence']}")


def test_same_org(results):
    check(results, "same_org: UC Berkeley variants match",
          same_org("University of California, Berkeley", "UC Berkeley"))
    check(results, "same_org: distinct institutions do not match",
          not same_org("University of Michigan", "Michigan State University"))
    check(results, "same_org: empty names do not match",
          not same_org("", "University of Michigan"))


def test_trajectories(results):
    """End-to-end: fixture person -> coded row."""
    cfg = load_config("fields.yaml")
    mapping = load_config("title_mapping.yaml")
    coder = TitleCoder(mapping)

    for case in handcrafted():
        row = code_person(case["person"], coder, cfg, mapping)
        name = case["person"]["display_name"]

        check(
            results,
            f"trajectory: {name} -> {case['expected_prior_rank_class']}",
            row["prior_rank_class"] == case["expected_prior_rank_class"],
            f"got {row['prior_rank_class']!r} "
            f"(prior={row['prior_role_title']!r} @ {row['prior_institution']!r} "
            f"[{row['prior_country']}])",
        )

        expected_paths = case["expected_paths"]
        actual_paths = {f for f in PATH_FLAGS if row.get(f) == "1"}
        check(
            results,
            f"paths: {name} -> {sorted(expected_paths) or 'none'}",
            actual_paths == expected_paths,
            f"got {sorted(actual_paths) or 'none'}",
        )

    # Targeted structural assertions.
    by_name = {}
    for case in handcrafted():
        row = code_person(case["person"], coder, cfg, mapping)
        by_name[case["person"]["display_name"]] = row

    promo = by_name["Internal Promotion Case"]
    check(results, "internal promotion is not treated as the prior position",
          promo["prior_institution"] == "University of Amsterdam",
          f"got {promo['prior_institution']!r}")
    check(results, "internal promotion resolves appointment to the original hire",
          str(promo["appointment_year"]) == "2018",
          f"got {promo['appointment_year']}")

    open_ended = by_name["Open-Ended Prior Case"]
    check(results, "missing end date still yields a duration",
          open_ended["prior_duration_years"] == 5,
          f"got {open_ended['prior_duration_years']!r}")
    check(results, "gap_years = appointment year - prior start year",
          open_ended["gap_years"] == 5, f"got {open_ended['gap_years']!r}")

    untitled = by_name["Untitled Employment Case"]
    check(results, "untitled prior position is flagged for review",
          untitled["needs_manual_review"] == "1",
          f"got {untitled['needs_manual_review']!r}")

    visiting = by_name["Visiting Overlap Case"]
    check(results, "visiting-overlap row is flagged rather than silently coded",
          visiting["needs_manual_review"] == "1",
          f"got {visiting['needs_manual_review']!r}")

    ext = by_name["US Extended Postdoc Case"]
    check(results, "extended postdoc counts to Path B broad only",
          ext["path_b_broad"] == "1" and ext["path_b_strict"] == "0",
          f"broad={ext['path_b_broad']} strict={ext['path_b_strict']}")

    std = by_name["US Standard Postdoc Case"]
    check(results, "standard postdoc counts to neither Path B variant",
          std["path_b_broad"] == "0" and std["path_b_strict"] == "0",
          f"broad={std['path_b_broad']} strict={std['path_b_strict']}")

    phd_case = by_name["UK Lecturer Case"]
    check(results, "PhD country and year extracted from educations",
          phd_case["phd_country"] == "GB" and str(phd_case["phd_year"]) == "2015",
          f"got {phd_case['phd_country']}/{phd_case['phd_year']}")


def main():
    results = []
    test_title_coder(results)
    test_same_org(results)
    test_trajectories(results)

    failures = [r for r in results if not r[1]]
    for label, ok, detail in results:
        if not ok:
            print(f"FAIL  {label}\n      {detail}")

    print(f"\n{len(results) - len(failures)}/{len(results)} assertions passed")
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print("All coding tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
