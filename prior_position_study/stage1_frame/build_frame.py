"""
Stage 1: Build the Sampling Frame (OpenAlex)
=============================================
Harvests candidate US ladder-rank faculty in the six target fields and
reconstructs a first-pass affiliation timeline for each.

What OpenAlex can and cannot do here
------------------------------------
OpenAlex records the INSTITUTION an author published from in a given year. It
does not record job titles, so this stage cannot distinguish ladder from
non-ladder on its own -- that is stage 2's job. What it gives us is the frame
plus a proxy appointment year.

The proxy is the first year the author appears with their current US
institution. It is biased LATE relative to the true start date, because papers
in the pipeline at the time of a move still carry the old affiliation, and
biased EARLY in the rarer case of a courtesy or visiting affiliation predating
the hire. Stage 3 prefers an ORCID start date whenever one is available and
falls back on this proxy otherwise; `appointment_year_source` records which
was used for every row.

Usage:
    python -m stage1_frame.build_frame
    python -m stage1_frame.build_frame --field communication --limit 200
    python -m stage1_frame.build_frame --skip-verify
"""

import argparse
import json
import random
from pathlib import Path

from common import (
    BASE_DIR,
    DATA_DIR,
    CachedSession,
    EgressBlocked,
    load_config,
    paginate_openalex,
    write_jsonl,
)

FRAME_PATH = DATA_DIR / "frame.jsonl"


def make_session(cfg: dict) -> CachedSession:
    oa = cfg["openalex"]
    return CachedSession(
        namespace="openalex",
        requests_per_second=oa.get("requests_per_second", 8),
        max_retries=oa.get("max_retries", 5),
        backoff_seconds=oa.get("backoff_seconds"),
        headers={"User-Agent": f"prior-position-study (mailto:{oa['mailto']})"},
    )


def verify_subfields(session: CachedSession, cfg: dict) -> None:
    """
    Confirm every asserted subfield ID resolves to the expected display name.

    Guards against silently harvesting the wrong literature if an ID in
    fields.yaml is stale or mistyped.
    """
    base = cfg["openalex"]["base_url"]
    problems = []
    for field_key, field in cfg["fields"].items():
        for sf in field["subfields"]:
            doc = session.get_json(f"{base}/subfields/{sf['id']}")
            if not doc:
                problems.append(f"{field_key}: subfield {sf['id']} did not resolve")
                continue
            actual = doc.get("display_name", "")
            if actual.strip().lower() != sf["name"].strip().lower():
                problems.append(
                    f"{field_key}: subfield {sf['id']} is '{actual}', "
                    f"config asserts '{sf['name']}'"
                )
    if problems:
        raise SystemExit(
            "Subfield verification failed. Fix config/fields.yaml before harvesting:\n  "
            + "\n  ".join(problems)
        )
    print("  subfield IDs verified against OpenAlex")


def load_institution_allowlist(cfg: dict) -> set:
    """Optional curated list of PhD-granting institutions, keyed by ROR."""
    path = BASE_DIR / cfg["population"]["institution_allowlist_csv"]
    if not path.exists():
        return set()
    import csv

    with open(path) as f:
        rows = list(csv.DictReader(f))
    rors = {r["ror"].strip().rstrip("/").split("/")[-1] for r in rows if r.get("ror")}
    print(f"  institution allowlist: {len(rors)} PhD-granting institutions")
    return rors


def institution_ok(inst: dict, cfg: dict, allowlist: set, session: CachedSession) -> bool:
    """Is this a US PhD-granting academic employer?"""
    pop = cfg["population"]
    if (inst.get("country_code") or "").upper() != pop["target_country"]:
        return False
    if inst.get("type") and inst["type"] not in pop["institution_types"]:
        return False

    ror = (inst.get("ror") or "").rstrip("/").split("/")[-1]
    if allowlist:
        return ror in allowlist

    # Fall back to the research-output proxy for doctoral status.
    detail = session.get_json(inst["id"].replace("https://openalex.org/", cfg["openalex"]["base_url"] + "/institutions/"))
    if not detail:
        return False
    return detail.get("works_count", 0) >= pop["min_institution_works_count"]


def affiliation_timeline(author: dict) -> list:
    """
    Normalize an OpenAlex author's affiliations into sorted spells.

    Returns [{institution_id, name, country, type, first_year, last_year}]
    sorted by first_year ascending.
    """
    spells = []
    for aff in author.get("affiliations", []):
        inst = aff.get("institution") or {}
        years = sorted(y for y in (aff.get("years") or []) if isinstance(y, int))
        if not inst.get("id") or not years:
            continue
        spells.append(
            {
                "institution_id": inst["id"],
                "name": inst.get("display_name"),
                "country": (inst.get("country_code") or "").upper() or None,
                "type": inst.get("type"),
                "ror": inst.get("ror"),
                "first_year": years[0],
                "last_year": years[-1],
                "years": years,
            }
        )
    return sorted(spells, key=lambda s: (s["first_year"], s["last_year"]))


def pick_current_institution(author: dict, spells: list, cfg: dict) -> dict:
    """
    The author's current institution: prefer OpenAlex's own last_known
    affiliation, falling back to the spell with the latest final year.
    """
    known = author.get("last_known_institutions") or []
    if known and known[0].get("id"):
        for s in spells:
            if s["institution_id"] == known[0]["id"]:
                return s
        inst = known[0]
        return {
            "institution_id": inst["id"],
            "name": inst.get("display_name"),
            "country": (inst.get("country_code") or "").upper() or None,
            "type": inst.get("type"),
            "ror": inst.get("ror"),
            "first_year": None,
            "last_year": None,
            "years": [],
        }
    return max(spells, key=lambda s: s["last_year"]) if spells else {}


def harvest_field(session: CachedSession, cfg: dict, field_key: str, field: dict,
                  allowlist: set, limit: int) -> list:
    """Query OpenAlex authors for one field and keep in-frame candidates."""
    base = cfg["openalex"]["base_url"]
    pop = cfg["population"]
    per_page = cfg["openalex"]["per_page"]
    subfield_ids = "|".join(f"subfields/{sf['id']}" for sf in field["subfields"])

    filters = ",".join(
        [
            f"last_known_institutions.country_code:{pop['target_country'].lower()}",
            f"topics.subfield.id:{subfield_ids}",
            "works_count:>4",
            "has_orcid:true",
        ]
    )
    url = (
        f"{base}/authors?filter={filters}"
        f"&per-page={per_page}&mailto={cfg['openalex']['mailto']}"
    )

    include = [k.lower() for k in field["topic_keywords"]["include"]]
    exclude = [k.lower() for k in field["topic_keywords"]["exclude"]]

    candidates, seen_inst = [], {}
    for author in paginate_openalex(session, url, max_results=limit):
        topics_blob = " ".join(
            (t.get("display_name") or "") for t in (author.get("topics") or [])
        ).lower()
        if any(x in topics_blob for x in exclude):
            continue
        # Subfield match already qualifies; keywords only need to fire for
        # authors whose subfield tag is generic.
        keyword_hit = any(k in topics_blob for k in include)

        spells = affiliation_timeline(author)
        current = pick_current_institution(author, spells, cfg)
        if not current or not current.get("institution_id"):
            continue

        inst_id = current["institution_id"]
        if inst_id not in seen_inst:
            seen_inst[inst_id] = institution_ok(current, cfg, allowlist, session)
        if not seen_inst[inst_id]:
            continue

        appt_year = current.get("first_year")
        if appt_year is None or not (
            pop["appointment_year_min"] <= appt_year <= pop["appointment_year_max"]
        ):
            continue

        prior_spells = [
            s for s in spells
            if s["institution_id"] != inst_id and s["first_year"] <= appt_year
        ]
        if not prior_spells:
            # No observable prior position; nothing to code.
            continue

        candidates.append(
            {
                "openalex_id": author["id"],
                "orcid": author.get("orcid"),
                "display_name": author.get("display_name"),
                "field": field_key,
                "keyword_hit": keyword_hit,
                "works_count": author.get("works_count"),
                "current_institution": current,
                "prior_spells_openalex": prior_spells,
                "all_spells_openalex": spells,
                "appointment_year_proxy": appt_year,
            }
        )
    return candidates


def stratify(candidates_by_field: dict, cfg: dict) -> list:
    """Cap each field at per_field_target so no discipline dominates."""
    rng = random.Random(cfg["sampling"]["random_seed"])
    target = cfg["sampling"]["per_field_target"]
    out = []
    for field_key, rows in candidates_by_field.items():
        # Prefer authors whose topics also matched the keyword screen.
        rows = sorted(rows, key=lambda r: (not r["keyword_hit"], -(r["works_count"] or 0)))
        if len(rows) > target:
            head = [r for r in rows if r["keyword_hit"]][:target]
            if len(head) < target:
                rest = [r for r in rows if not r["keyword_hit"]]
                rng.shuffle(rest)
                head += rest[: target - len(head)]
            rows = head
        print(f"  {field_key}: {len(rows)} retained")
        out.extend(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", help="restrict to a single field key")
    ap.add_argument("--limit", type=int, help="max authors harvested per field")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip subfield ID verification (not recommended)")
    ap.add_argument("--output", default=str(FRAME_PATH))
    args = ap.parse_args()

    cfg = load_config("fields.yaml")
    session = make_session(cfg)

    try:
        if not args.skip_verify:
            print("Verifying subfield IDs...")
            verify_subfields(session, cfg)

        allowlist = load_institution_allowlist(cfg)
        limit = args.limit or cfg["sampling"]["per_field_max_candidates"]

        fields = cfg["fields"]
        if args.field:
            if args.field not in fields:
                raise SystemExit(f"Unknown field '{args.field}'. Options: {list(fields)}")
            fields = {args.field: fields[args.field]}

        by_field = {}
        for key, field in fields.items():
            print(f"Harvesting {field['label']}...")
            by_field[key] = harvest_field(session, cfg, key, field, allowlist, limit)
            print(f"  {len(by_field[key])} candidates in frame")

        frame = stratify(by_field, cfg)
    except EgressBlocked as exc:
        raise SystemExit(
            f"\nNetwork blocked: {exc}\n"
            "OpenAlex is unreachable from this environment. Run this stage where "
            "api.openalex.org is permitted by egress policy."
        )

    write_jsonl(Path(args.output), frame)
    print(f"\nFrame: {len(frame)} candidates -> {args.output}")
    print(f"Cache: {session.stats}")


if __name__ == "__main__":
    main()
