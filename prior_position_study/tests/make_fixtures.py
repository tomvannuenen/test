"""
Build Test Fixtures
=====================
Generates a synthetic `enriched.jsonl` in the exact shape stage 2 emits, so
stages 3-5 can be exercised without network access.

Two parts:
  * HANDCRAFTED cases -- one per known coding hazard, with the expected
    prior_rank_class asserted in tests/test_coding.py.
  * SYNTHETIC bulk    -- ~320 people drawn from plausible distributions, so
    the stage 5 tables have enough rows to render.

The synthetic bulk exists ONLY to exercise the code paths. Its numbers are
drawn from arbitrary parameters and mean nothing substantively.

Usage:
    python -m tests.make_fixtures --output data/enriched_fixture.jsonl
"""

import argparse
import random
from pathlib import Path

from common import write_jsonl


def person(pid, name, field, current_inst, appt_year, employments,
           educations=None, pubs=12):
    return {
        "openalex_id": f"https://openalex.org/{pid}",
        "orcid": f"https://orcid.org/0000-0000-0000-{pid[-4:]}",
        "display_name": name,
        "field": field,
        "keyword_hit": True,
        "works_count": pubs + 5,
        "current_institution": {
            "institution_id": "https://openalex.org/I000",
            "name": current_inst,
            "country": "US",
            "type": "education",
            "ror": None,
            "first_year": appt_year,
            "last_year": 2024,
            "years": list(range(appt_year, 2025)),
        },
        "prior_spells_openalex": [],
        "all_spells_openalex": [],
        "appointment_year_proxy": appt_year,
        "pubs_pre_move": pubs,
        "orcid_employments": employments,
        "orcid_educations": educations or [],
    }


def emp(org, country, title, start, end=None, dept=None):
    return {
        "org_name": org,
        "org_country": country,
        "org_city": None,
        "role_title": title,
        "department": dept,
        "start_year": start,
        "start_month": 8,
        "end_year": end,
        "end_month": 7 if end else None,
        "source": "orcid",
    }


def phd(org, country, year):
    return [emp(org, country, "PhD", year - 5, year, dept="Doctor of Philosophy")]


# =============================================================================
# Handcrafted cases: (id, expected_prior_rank_class, expected_path, person)
# =============================================================================


def handcrafted():
    cases = []

    def add(pid, expected_class, expected_paths, p):
        cases.append(
            {"person_id": f"https://openalex.org/{pid}",
             "expected_prior_rank_class": expected_class,
             "expected_paths": expected_paths,
             "person": p}
        )

    # --- The UK/US Lecturer inversion --------------------------------------
    add("A1001", "ladder", {"path_a"},
        person("A1001", "UK Lecturer Case", "communication",
               "University of Michigan", 2019,
               [emp("University of Manchester", "GB", "Lecturer in Media", 2015, 2019),
                emp("University of Michigan", "US", "Assistant Professor", 2019)],
               phd("University of Leeds", "GB", 2015)))

    add("A1002", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1002", "US Lecturer Case", "communication",
               "Ohio State University", 2020,
               [emp("Boston University", "US", "Lecturer", 2016, 2020),
                emp("Ohio State University", "US", "Assistant Professor", 2020)],
               phd("Boston University", "US", 2016)))

    # --- Hong Kong: RAP is NOT ladder --------------------------------------
    add("A1003", "non-ladder-academic", set(),
        person("A1003", "HK Research Asst Prof Case", "human_computer_interaction",
               "University of Washington", 2021,
               [emp("City University of Hong Kong", "HK",
                    "Research Assistant Professor", 2018, 2021),
                emp("University of Washington", "US", "Assistant Professor", 2021)],
               phd("Chinese University of Hong Kong", "HK", 2018)))

    add("A1004", "ladder", {"path_a"},
        person("A1004", "HK Assistant Prof Case", "computational_social_science",
               "Cornell University", 2022,
               [emp("Hong Kong University of Science and Technology", "HK",
                    "Assistant Professor", 2017, 2022),
                emp("Cornell University", "US", "Assistant Professor", 2022)],
               phd("Stanford University", "US", 2017)))

    # --- Singapore: Research Fellow = postdoc, Lecturer = non-ladder -------
    add("A1005", "postdoc", set(),
        person("A1005", "SG Research Fellow Case", "information_science",
               "Rutgers University", 2018,
               [emp("National University of Singapore", "SG", "Research Fellow", 2016, 2018),
                emp("Rutgers University", "US", "Assistant Professor", 2018)],
               phd("Nanyang Technological University", "SG", 2016)))

    add("A1006", "non-ladder-academic", set(),
        person("A1006", "SG Lecturer Case", "digital_humanities",
               "New York University", 2023,
               [emp("Nanyang Technological University", "SG", "Senior Lecturer", 2018, 2023),
                emp("New York University", "US", "Assistant Professor", 2023)],
               phd("University of Toronto", "CA", 2017)))

    # --- Continental Europe -------------------------------------------------
    add("A1007", "ladder", {"path_a"},
        person("A1007", "NL UHD Case", "science_technology_studies",
               "Georgia Institute of Technology", 2020,
               [emp("Universiteit van Amsterdam", "NL", "Universitair Hoofddocent", 2016, 2020),
                emp("Georgia Institute of Technology", "US", "Associate Professor", 2020)],
               phd("Universiteit Twente", "NL", 2012)))

    add("A1008", "postdoc", set(),
        person("A1008", "DE Wiss Mitarbeiter Case", "science_technology_studies",
               "University of Wisconsin-Madison", 2017,
               [emp("Technische Universitat Munchen", "DE",
                    "Wissenschaftlicher Mitarbeiter", 2014, 2017),
                emp("University of Wisconsin-Madison", "US", "Assistant Professor", 2017)],
               phd("Universitat Bielefeld", "DE", 2014)))

    # Swedish universitetslektor is a permanent ladder post.
    add("A1009", "ladder", {"path_a"},
        person("A1009", "SE Lektor Case", "information_science",
               "University of Illinois", 2021,
               [emp("Lunds Universitet", "SE", "Universitetslektor", 2016, 2021),
                emp("University of Illinois", "US", "Associate Professor", 2021)],
               phd("Goteborgs Universitet", "SE", 2013)))

    # Swedish adjunkt is NOT (contrast with the Danish adjunkt below).
    add("A1010", "non-ladder-academic", set(),
        person("A1010", "SE Adjunkt Case", "communication",
               "Michigan State University", 2019,
               [emp("Umea Universitet", "SE", "Adjunkt", 2015, 2019),
                emp("Michigan State University", "US", "Assistant Professor", 2019)],
               phd("Umea Universitet", "SE", 2015)))

    add("A1011", "ladder", {"path_a"},
        person("A1011", "DK Adjunkt Case", "computational_social_science",
               "Northwestern University", 2022,
               [emp("Kobenhavns Universitet", "DK", "Adjunkt", 2018, 2022),
                emp("Northwestern University", "US", "Assistant Professor", 2022)],
               phd("Aarhus Universitet", "DK", 2018)))

    # Spanish "profesor asociado" is an adjunct, not a US associate professor.
    add("A1012", "non-ladder-academic", set(),
        person("A1012", "ES Asociado Case", "digital_humanities",
               "University of Texas at Austin", 2018,
               [emp("Universitat Pompeu Fabra", "ES", "Profesor Asociado", 2014, 2018),
                emp("University of Texas at Austin", "US", "Assistant Professor", 2018)],
               phd("Universitat Autonoma de Barcelona", "ES", 2013)))

    # CNRS permanent researcher -> coded ladder.
    add("A1013", "ladder", {"path_a"},
        person("A1013", "FR CNRS Case", "computational_social_science",
               "Columbia University", 2023,
               [emp("Centre National de la Recherche Scientifique", "FR",
                    "Charge de Recherche", 2017, 2023),
                emp("Columbia University", "US", "Associate Professor", 2023)],
               phd("Sorbonne Universite", "FR", 2016)))

    # --- US non-ladder variants --------------------------------------------
    add("A1014", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1014", "US Adjunct Case", "communication",
               "University of Oregon", 2017,
               [emp("Emerson College", "US", "Adjunct Professor", 2014, 2017),
                emp("University of Oregon", "US", "Assistant Professor", 2017)],
               phd("Temple University", "US", 2013)))

    add("A1015", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1015", "US Project Scientist Case", "human_computer_interaction",
               "University of Maryland", 2019,
               [emp("University of California, San Diego", "US", "Project Scientist", 2015, 2019),
                emp("University of Maryland", "US", "Assistant Professor", 2019)],
               phd("University of California, Irvine", "US", 2014)))

    # US "Research Assistant Professor" is non-ladder too.
    add("A1016", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1016", "US Research Asst Prof Case", "information_science",
               "Indiana University", 2020,
               [emp("University of Pittsburgh", "US", "Research Assistant Professor", 2016, 2020),
                emp("Indiana University", "US", "Assistant Professor", 2020)],
               phd("University of Pittsburgh", "US", 2015)))

    # Standard 2-year US postdoc: the modal path, NOT Path B.
    add("A1017", "postdoc", set(),
        person("A1017", "US Standard Postdoc Case", "computational_social_science",
               "Duke University", 2021,
               [emp("Princeton University", "US", "Postdoctoral Research Associate", 2019, 2021),
                emp("Duke University", "US", "Assistant Professor", 2021)],
               phd("Harvard University", "US", 2019)))

    # 6-year US postdoc: "extended postdoc", so Path B broad but not strict.
    add("A1018", "postdoc", {"path_b_broad"},
        person("A1018", "US Extended Postdoc Case", "science_technology_studies",
               "University of Virginia", 2023,
               [emp("Massachusetts Institute of Technology", "US",
                    "Postdoctoral Associate", 2017, 2023),
                emp("University of Virginia", "US", "Assistant Professor", 2023)],
               phd("Massachusetts Institute of Technology", "US", 2017)))

    # --- Industry -----------------------------------------------------------
    add("A1019", "industry", set(),
        person("A1019", "Industry Case", "human_computer_interaction",
               "Carnegie Mellon University", 2022,
               [emp("Microsoft Research", "US", "Senior Researcher", 2018, 2022),
                emp("Carnegie Mellon University", "US", "Assistant Professor", 2022)],
               phd("University of Washington", "US", 2018)))

    # --- Structural edge cases ---------------------------------------------
    # Internal promotion must NOT be read as the prior position.
    add("A1020", "ladder", {"path_a"},
        person("A1020", "Internal Promotion Case", "communication",
               "University of Pennsylvania", 2018,
               [emp("University of Amsterdam", "NL", "Universitair Docent", 2014, 2018),
                emp("University of Pennsylvania", "US", "Assistant Professor", 2018, 2024),
                emp("University of Pennsylvania", "US", "Associate Professor", 2024)],
               phd("University of Amsterdam", "NL", 2013)))

    # Visiting position overlaps the real prior post. "Visiting" is non-ladder
    # and starts later, so by the rules it wins as "immediately prior" and the
    # row scores as Path B -- even though the substantive prior post was a UK
    # Senior Lectureship (Path A). The rules-correct answer is substantively
    # misleading, so the row MUST be flagged for review; see the masked-position
    # check in find_prior_position.
    add("A1021", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1021", "Visiting Overlap Case", "digital_humanities",
               "Yale University", 2020,
               [emp("University of Bristol", "GB", "Senior Lecturer", 2013, 2019),
                emp("Stanford University", "US", "Visiting Scholar", 2019, 2020),
                emp("Yale University", "US", "Assistant Professor", 2020)],
               phd("University of Cambridge", "GB", 2012)))

    # Missing end date on the prior post: duration falls back to appt year.
    add("A1022", "ladder", {"path_a"},
        person("A1022", "Open-Ended Prior Case", "information_science",
               "University of North Carolina", 2021,
               [emp("University of Sheffield", "GB", "Senior Lecturer", 2016),
                emp("University of North Carolina", "US", "Associate Professor", 2021)],
               phd("University of Sheffield", "GB", 2011)))

    # Ambiguous "Research Fellow" resolved by duration: 7 years -> staff.
    add("A1023", "non-ladder-academic", {"path_b_strict", "path_b_broad"},
        person("A1023", "Long Research Fellow Case", "computational_social_science",
               "Arizona State University", 2023,
               [emp("Johns Hopkins University", "US", "Research Fellow", 2016, 2023),
                emp("Arizona State University", "US", "Assistant Professor", 2023)],
               phd("Johns Hopkins University", "US", 2015)))

    # No title at all -> "other" and flagged.
    add("A1024", "other", set(),
        person("A1024", "Untitled Employment Case", "communication",
               "University of Iowa", 2019,
               [emp("Some Organization", "US", None, 2016, 2019),
                emp("University of Iowa", "US", "Assistant Professor", 2019)],
               phd("University of Iowa", "US", 2015)))

    return cases


# =============================================================================
# Synthetic bulk
# =============================================================================

FIELDS = [
    "computational_social_science", "communication", "information_science",
    "science_technology_studies", "digital_humanities", "human_computer_interaction",
]

NON_US_LADDER = [
    ("University of Manchester", "GB", "Lecturer"),
    ("King's College London", "GB", "Senior Lecturer"),
    ("University of Hong Kong", "HK", "Assistant Professor"),
    ("City University of Hong Kong", "HK", "Assistant Professor"),
    ("National University of Singapore", "SG", "Assistant Professor"),
    ("Universiteit Utrecht", "NL", "Universitair Docent"),
    ("University of Melbourne", "AU", "Senior Lecturer"),
    ("Universitat Zurich", "CH", "Assistant Professor"),
    ("Kobenhavns Universitet", "DK", "Adjunkt"),
]

US_NON_LADDER = [
    ("Boston University", "US", "Lecturer"),
    ("New York University", "US", "Adjunct Professor"),
    ("University of California, Davis", "US", "Project Scientist"),
    ("Northeastern University", "US", "Research Scientist"),
    ("University of Michigan", "US", "Academic Coordinator"),
    ("University of Chicago", "US", "Research Assistant Professor"),
]

US_POSTDOC = [
    ("Princeton University", "US", "Postdoctoral Research Associate"),
    ("Stanford University", "US", "Postdoctoral Scholar"),
    ("Massachusetts Institute of Technology", "US", "Postdoctoral Associate"),
]

INDUSTRY = [
    ("Microsoft Research", "US", "Senior Researcher"),
    ("Google LLC", "US", "Research Scientist"),
    ("Spotify", "SE", "Research Scientist"),
]

US_UNIVERSITIES = [
    "University of Michigan", "Cornell University", "University of Washington",
    "Northwestern University", "University of Texas at Austin", "Rutgers University",
    "Indiana University", "University of Maryland", "Ohio State University",
    "University of Colorado Boulder", "Syracuse University", "Drexel University",
]


def synthetic(n: int, seed: int = 42) -> list:
    """
    Draw plausible-looking people. Mixture weights are arbitrary; they exist
    to populate every cell of the stage 5 tables, not to estimate anything.
    """
    rng = random.Random(seed)
    people = []
    for i in range(n):
        pid = f"S{2000 + i}"
        field = rng.choice(FIELDS)
        appt = rng.randint(2016, 2024)
        us_inst = rng.choice(US_UNIVERSITIES)

        roll = rng.random()
        if roll < 0.16:
            org, country, title = rng.choice(NON_US_LADDER)
            dur = rng.randint(2, 7)
        elif roll < 0.34:
            org, country, title = rng.choice(US_NON_LADDER)
            dur = rng.randint(1, 6)
        elif roll < 0.80:
            org, country, title = rng.choice(US_POSTDOC)
            dur = rng.randint(1, 7)
        else:
            org, country, title = rng.choice(INDUSTRY)
            dur = rng.randint(2, 6)

        prior_start = appt - dur
        phd_year = prior_start - rng.randint(0, 2)
        # Publication counts loosely track seniority at the time of the move.
        pubs = max(0, int(rng.gauss(4 + 2.2 * dur, 6)))

        employments = [
            emp(org, country, title, prior_start, appt),
            emp(us_inst, "US", rng.choice(["Assistant Professor", "Associate Professor"]), appt),
        ]
        people.append(
            person(pid, f"Synthetic Person {i}", field, us_inst, appt,
                   employments, phd(rng.choice(US_UNIVERSITIES), "US", phd_year), pubs)
        )
    return people


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/enriched_fixture.jsonl")
    ap.add_argument("--n-synthetic", type=int, default=320)
    args = ap.parse_args()

    rows = [c["person"] for c in handcrafted()] + synthetic(args.n_synthetic)
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} fixture people -> {args.output}")
    print(f"  {len(handcrafted())} handcrafted coding-hazard cases")
    print(f"  {args.n_synthetic} synthetic bulk rows (meaningless numbers, code-path only)")


if __name__ == "__main__":
    main()
