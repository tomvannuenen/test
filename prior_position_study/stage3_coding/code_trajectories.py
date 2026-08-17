"""
Stage 3: Code Trajectories
============================
Turns enriched ORCID employment histories into the tidy coded CSV described in
the brief. Two problems get solved here.

1. TITLE RESOLUTION. Every role title is resolved against the academic system
   of the employing country, never against the bare English string. See
   config/title_mapping.yaml for the table and its rationale.

2. TRAJECTORY RECONSTRUCTION. Employment records overlap, carry missing end
   dates, and mix substantive posts with courtesy appointments. We identify
   the current US ladder appointment, then the position held immediately
   before it at a DIFFERENT organization (an assistant-to-associate promotion
   at the same place is not a prior position).

`gap_years` follows the brief's definition -- appointment year minus prior
position START year. It therefore measures time served in the prior post, not
an employment gap.

Usage:
    python -m stage3_coding.code_trajectories
    python -m stage3_coding.code_trajectories --merge-manual data/manual_coded.csv
"""

import argparse
import csv
import re
from pathlib import Path

from common import BASE_DIR, DATA_DIR, load_config, read_jsonl

ENRICHED_PATH = DATA_DIR / "enriched.jsonl"
CODED_PATH = DATA_DIR / "coded_trajectories.csv"

CSV_COLUMNS = [
    "person_id",
    "display_name",
    "field",
    "current_institution",
    "current_country",
    "appointment_year",
    "appointment_year_source",
    "prior_rank_class",
    "prior_country",
    "prior_institution",
    "prior_role_title",
    "prior_start_year",
    "prior_end_year",
    "prior_duration_years",
    "gap_years",
    "phd_country",
    "phd_year",
    "pubs_pre_move",
    "path_a",
    "path_b_strict",
    "path_b_broad",
    "coding_confidence",
    "coding_system",
    "coding_rule",
    "coding_note",
    "needs_manual_review",
    "coding_source",
]

AMBIGUOUS_RESEARCH_TITLE = re.compile(
    r"\bresearch (fellow|scientist|associate|scholar|staff|officer)\b"
    r"|\bwissenschaftliche",
    re.I,
)

DOCTORATE_PATTERNS = re.compile(
    r"\b(ph\.?d|d\.?phil|doctor of philosophy|doctorate|doctoral|dr\.? rer|"
    r"dr\.? phil|sc\.?d|promotie)\b",
    re.I,
)


# =============================================================================
# Title coder
# =============================================================================


class TitleCoder:
    """Resolves (role title, country, organization) -> rank class."""

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.country_systems = {
            k.upper(): v for k, v in mapping.get("country_systems", {}).items()
        }
        self.org_overrides = self._compile_org_overrides(
            mapping.get("organization_overrides", {})
        )
        self.universal = self._compile_rules(mapping.get("universal_prefix_rules", []))
        self.systems = {
            key: self._compile_rules(sys_def.get("rules", []))
            for key, sys_def in mapping.get("systems", {}).items()
        }
        heur = mapping.get("research_title_duration_heuristic", {})
        self.heuristic_enabled = heur.get("enabled", False)
        self.short_max = heur.get("short_years_max", 3)
        self.long_min = heur.get("long_years_min", 5)

    @staticmethod
    def _compile_rules(rules: list) -> list:
        out = []
        for r in rules:
            out.append(
                {
                    "regex": re.compile(r["pattern"], re.I),
                    "pattern": r["pattern"],
                    "rank_class": r["rank_class"],
                    "confidence": r.get("confidence", "low"),
                    "note": r.get("note", ""),
                }
            )
        return out

    @staticmethod
    def _compile_org_overrides(overrides: dict) -> list:
        out = []
        for key, spec in overrides.items():
            out.append(
                {
                    "key": key,
                    "rank_class": spec.get("rank_class", key),
                    "confidence": spec.get("confidence", "medium"),
                    "regexes": [re.compile(p, re.I) for p in spec.get("name_patterns", [])],
                }
            )
        return out

    def system_for(self, country: str) -> str:
        return self.country_systems.get((country or "").upper(), "generic")

    def code(self, title: str, country: str, org_name: str = "",
             duration_years=None) -> dict:
        """
        Resolve one position.

        Returns dict with rank_class, confidence, system, rule, note.
        """
        title = (title or "").strip()
        org_name = (org_name or "").strip()

        # 1. Non-academic employer overrides everything.
        for ov in self.org_overrides:
            if any(rx.search(org_name) for rx in ov["regexes"]):
                return {
                    "rank_class": ov["rank_class"],
                    "confidence": ov["confidence"],
                    "system": "organization_override",
                    "rule": ov["key"],
                    "note": f"Employer matched {ov['key']} override.",
                }

        if not title:
            return {
                "rank_class": "other",
                "confidence": "none",
                "system": self.system_for(country),
                "rule": "",
                "note": "No role title recorded.",
            }

        system_key = self.system_for(country)

        # 2. Universal statuses, then 3. system rules, then 4. generic.
        for stage_name, rules in (
            ("universal", self.universal),
            (system_key, self.systems.get(system_key, [])),
            ("generic", self.systems.get("generic", [])),
        ):
            for rule in rules:
                if rule["regex"].search(title):
                    result = {
                        "rank_class": rule["rank_class"],
                        "confidence": rule["confidence"],
                        "system": stage_name,
                        "rule": rule["pattern"],
                        "note": rule["note"],
                    }
                    return self._apply_duration_heuristic(result, title, duration_years)

        return {
            "rank_class": "other",
            "confidence": "none",
            "system": system_key,
            "rule": "",
            "note": f"Unmatched title: {title!r}",
        }

    def _apply_duration_heuristic(self, result: dict, title: str, duration_years):
        """
        Split ambiguous research titles by how long they were held.

        A two-year "Research Fellow" is a postdoc in all but name; an
        eight-year one is a staff scientist. Only fires on rules already
        flagged as uncertain.
        """
        if not self.heuristic_enabled or duration_years is None:
            return result
        if result["confidence"] not in ("low", "medium"):
            return result
        if result["rank_class"] not in ("postdoc", "non-ladder-academic"):
            return result
        if not AMBIGUOUS_RESEARCH_TITLE.search(title):
            return result

        if duration_years <= self.short_max:
            result["rank_class"] = "postdoc"
            result["note"] = (
                f"{result['note']} Duration heuristic: {duration_years}y -> postdoc."
            ).strip()
        elif duration_years >= self.long_min:
            result["rank_class"] = "non-ladder-academic"
            result["note"] = (
                f"{result['note']} Duration heuristic: {duration_years}y -> staff research."
            ).strip()
        return result


# =============================================================================
# Trajectory reconstruction
# =============================================================================


# Abbreviations that would otherwise defeat token comparison.
ORG_ALIASES = {
    r"\buc\b": "university of california",
    r"\bucla\b": "university of california los angeles",
    r"\bucsd\b": "university of california san diego",
    r"\bucsf\b": "university of california san francisco",
    r"\bmit\b": "massachusetts institute of technology",
    r"\bnyu\b": "new york university",
    r"\bcmu\b": "carnegie mellon university",
    r"\bucl\b": "university college london",
    r"\bkcl\b": "kings college london",
    r"\blse\b": "london school of economics",
    r"\bnus\b": "national university of singapore",
    r"\bntu\b": "nanyang technological university",
    r"\bhkust\b": "hong kong university of science and technology",
    r"\bcuhk\b": "chinese university of hong kong",
    r"\beth\b": "eidgenossische technische hochschule",
    r"\bcnrs\b": "centre national de la recherche scientifique",
}

# Tokens that mark genuinely different institutions sharing a place name --
# "University of Michigan" vs "Michigan State University" is the canonical
# collision. Their presence in the token difference blocks a subset match.
ORG_DISCRIMINATORS = {
    "state", "tech", "technological", "technology", "polytechnic",
    "agricultural", "medical", "health", "a&m",
}

ORG_STOPWORDS = {"the", "of", "at", "and", "in", "for", "a", "de", "der", "van"}


def normalize_org_tokens(name: str) -> set:
    """
    Normalize an organization name to a comparable token set.

    Unlike a bare string normalization, this keeps structural words such as
    "state" and "university", which is what makes the discriminator check
    below possible.
    """
    n = (name or "").lower()
    n = re.sub(r"[^a-z0-9& ]", " ", n)
    n = " ".join(n.split())
    for pattern, expansion in ORG_ALIASES.items():
        n = re.sub(pattern, expansion, n)
    return {t for t in n.split() if t and t not in ORG_STOPWORDS}


def normalize_org(name: str) -> str:
    """String form of the normalized token set, for display and equality."""
    return " ".join(sorted(normalize_org_tokens(name)))


def same_org(a: str, b: str) -> bool:
    """
    Are these two strings the same employer?

    Matches when one token set contains the other -- so "University of
    Illinois" matches "University of Illinois Urbana-Champaign" -- but only
    if the surplus tokens carry no discriminator, and only in one direction.
    If each name has tokens the other lacks (Berkeley vs Davis) they are
    different institutions.
    """
    ta, tb = normalize_org_tokens(a), normalize_org_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True

    only_a, only_b = ta - tb, tb - ta
    if only_a and only_b:
        # Both sides carry distinct content: different institutions.
        return False

    surplus = only_a or only_b
    if surplus & ORG_DISCRIMINATORS:
        return False
    # Require the shared portion to actually be substantive.
    return len(ta & tb) >= 1


def duration_of(emp: dict, fallback_end: int = None):
    start, end = emp.get("start_year"), emp.get("end_year")
    if not start:
        return None
    end = end or fallback_end
    if not end:
        return None
    return max(0, end - start)


def find_current_appointment(employments: list, current_inst_name: str,
                             coder: TitleCoder, pop: dict):
    """
    Locate the current US ladder appointment.

    Prefers the earliest LADDER-coded employment at the current institution,
    so an assistant-to-associate promotion resolves to the original hire.
    """
    matches = []
    for emp in employments:
        if not same_org(emp.get("org_name"), current_inst_name):
            continue
        if not emp.get("start_year"):
            continue
        coded = coder.code(
            emp.get("role_title"),
            emp.get("org_country") or pop["target_country"],
            emp.get("org_name"),
            duration_of(emp),
        )
        matches.append((emp, coded))

    ladder = [m for m in matches if m[1]["rank_class"] == "ladder"]
    if ladder:
        return min(ladder, key=lambda m: m[0]["start_year"])
    if matches:
        # Employment at the right place but the title did not resolve to
        # ladder. Return it so the row can be flagged rather than dropped.
        return min(matches, key=lambda m: m[0]["start_year"])
    return None, None


def find_prior_position(employments: list, current_org: str, appointment_year: int,
                        coder: TitleCoder):
    """
    The position held immediately before the current appointment.

    Candidates must start before the appointment year and sit at a different
    organization. Contiguous posts (still open, or ending within a year of the
    move) are preferred over ones that ended long beforehand; ties break to the
    latest start.
    """
    candidates = []
    for emp in employments:
        start = emp.get("start_year")
        if not start or start >= appointment_year:
            continue
        if same_org(emp.get("org_name"), current_org):
            continue
        end = emp.get("end_year")
        contiguous = end is None or end >= appointment_year - 1
        candidates.append((contiguous, start, end or 0, emp))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    emp = candidates[0][3]
    coded = coder.code(
        emp.get("role_title"),
        emp.get("org_country"),
        emp.get("org_name"),
        duration_of(emp, fallback_end=appointment_year),
    )

    # A one-year transitional post (a visiting year, a bridging fellowship)
    # can mask the substantive prior position. Surface the runner-up so the
    # row can be flagged rather than silently coded off the transition.
    masked = None
    selected_duration = duration_of(emp, fallback_end=appointment_year)
    if selected_duration is not None and selected_duration <= 1:
        for _, start, end, other in candidates[1:]:
            other_duration = duration_of(other, fallback_end=appointment_year)
            if other_duration is not None and other_duration >= 3:
                masked = other
                break

    return emp, coded, masked


def find_doctorate(educations: list):
    """Extract PhD country and completion year."""
    best = None
    for ed in educations:
        blob = f"{ed.get('role_title') or ''} {ed.get('department') or ''}"
        if not DOCTORATE_PATTERNS.search(blob):
            continue
        year = ed.get("end_year") or ed.get("start_year")
        if best is None or (year and (best.get("end_year") or 0) < year):
            best = ed
    if not best:
        return None, None
    return best.get("org_country"), best.get("end_year") or best.get("start_year")


# =============================================================================
# Row assembly
# =============================================================================


def code_person(row: dict, coder: TitleCoder, cfg: dict, mapping: dict) -> dict:
    pop = cfg["population"]
    employments = row.get("orcid_employments") or []
    current_inst = row.get("current_institution") or {}
    current_name = current_inst.get("name") or ""

    out = {c: "" for c in CSV_COLUMNS}
    out.update(
        {
            "person_id": row.get("openalex_id", ""),
            "display_name": row.get("display_name", ""),
            "field": row.get("field", ""),
            "current_institution": current_name,
            "current_country": current_inst.get("country") or pop["target_country"],
            "pubs_pre_move": row.get("pubs_pre_move", ""),
            "coding_source": "automated",
            "needs_manual_review": "0",
        }
    )

    review_reasons = []

    current_emp, current_coded = find_current_appointment(
        employments, current_name, coder, pop
    )
    if current_emp:
        appointment_year = current_emp["start_year"]
        out["appointment_year_source"] = "orcid"
        if current_coded["rank_class"] != "ladder":
            review_reasons.append(
                f"current post at {current_name} coded "
                f"{current_coded['rank_class']}, not ladder"
            )
    else:
        appointment_year = row.get("appointment_year_proxy")
        out["appointment_year_source"] = "openalex_affiliation_proxy"
        review_reasons.append("no ORCID employment matched the current institution")

    out["appointment_year"] = appointment_year or ""

    if not appointment_year:
        out["needs_manual_review"] = "1"
        out["coding_note"] = "; ".join(review_reasons) or "no appointment year"
        out["prior_rank_class"] = ""
        return out

    # Population filter, re-applied against the ORCID date when we have one.
    if not (pop["appointment_year_min"] <= appointment_year <= pop["appointment_year_max"]):
        out["needs_manual_review"] = "1"
        review_reasons.append(
            f"appointment year {appointment_year} outside "
            f"{pop['appointment_year_min']}-{pop['appointment_year_max']}"
        )

    prior_emp, prior_coded, masked_emp = find_prior_position(
        employments, current_name, appointment_year, coder
    )

    if not prior_emp:
        out["needs_manual_review"] = "1"
        review_reasons.append("no prior employment record found")
        out["coding_note"] = "; ".join(review_reasons)
        return out

    duration = duration_of(prior_emp, fallback_end=appointment_year)
    out.update(
        {
            "prior_rank_class": prior_coded["rank_class"],
            "prior_country": prior_emp.get("org_country") or "",
            "prior_institution": prior_emp.get("org_name") or "",
            "prior_role_title": prior_emp.get("role_title") or "",
            "prior_start_year": prior_emp.get("start_year") or "",
            "prior_end_year": prior_emp.get("end_year") or "",
            "prior_duration_years": duration if duration is not None else "",
            "gap_years": appointment_year - prior_emp["start_year"],
            "coding_confidence": prior_coded["confidence"],
            "coding_system": prior_coded["system"],
            "coding_rule": prior_coded["rule"],
        }
    )

    phd_country, phd_year = find_doctorate(row.get("orcid_educations") or [])
    out["phd_country"] = phd_country or ""
    out["phd_year"] = phd_year or ""

    # Path classification
    ext_min = mapping.get("extended_postdoc_min_years", 4)
    prior_country = (out["prior_country"] or "").upper()
    rank = out["prior_rank_class"]
    is_us = prior_country == pop["target_country"]

    path_a = rank == "ladder" and bool(prior_country) and not is_us
    path_b_strict = is_us and rank == "non-ladder-academic"
    path_b_broad = path_b_strict or (
        is_us and rank == "postdoc" and duration is not None and duration >= ext_min
    )
    out["path_a"] = "1" if path_a else "0"
    out["path_b_strict"] = "1" if path_b_strict else "0"
    out["path_b_broad"] = "1" if path_b_broad else "0"

    if prior_coded["confidence"] in ("low", "none"):
        review_reasons.append(f"low-confidence title coding: {prior_coded['note']}")
    if not prior_country:
        review_reasons.append("prior country missing from ORCID record")
    if masked_emp:
        review_reasons.append(
            f"selected prior post is transitional (<=1y); it may mask "
            f"{masked_emp.get('role_title')!r} at {masked_emp.get('org_name')} "
            f"[{masked_emp.get('org_country')}], "
            f"{masked_emp.get('start_year')}-{masked_emp.get('end_year') or ''}"
        )

    if review_reasons:
        out["needs_manual_review"] = "1"
    notes = [prior_coded["note"]] if prior_coded["note"] else []
    out["coding_note"] = "; ".join(notes + review_reasons)
    return out


def merge_manual(rows: list, manual_path: Path) -> list:
    """
    Overlay hand-coded rows onto the automated output.

    Hand codings win; `coding_source` is set to "manual" so the validation
    stage can exclude them from the agreement computation.
    """
    if not manual_path.exists():
        print(f"  no manual file at {manual_path}, skipping merge")
        return rows
    with open(manual_path) as f:
        manual = {r["person_id"]: r for r in csv.DictReader(f)}
    merged = 0
    for row in rows:
        m = manual.get(row["person_id"])
        if not m:
            continue
        for col in CSV_COLUMNS:
            if col in m and m[col] not in ("", None):
                row[col] = m[col]
        row["coding_source"] = "manual"
        row["needs_manual_review"] = "0"
        merged += 1
    print(f"  merged {merged} hand-coded rows")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(ENRICHED_PATH))
    ap.add_argument("--output", default=str(CODED_PATH))
    ap.add_argument("--merge-manual", help="CSV of hand-coded rows to overlay")
    args = ap.parse_args()

    cfg = load_config("fields.yaml")
    mapping = load_config("title_mapping.yaml")
    coder = TitleCoder(mapping)

    people = read_jsonl(Path(args.input))
    if not people:
        raise SystemExit(f"No enriched data at {args.input}. Run stage 2 first.")

    rows = [code_person(p, coder, cfg, mapping) for p in people]

    if args.merge_manual:
        rows = merge_manual(rows, Path(args.merge_manual))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    codeable = [r for r in rows if r["prior_rank_class"]]
    flagged = [r for r in rows if r["needs_manual_review"] == "1"]
    print(f"\nCoded {len(rows)} people -> {out_path}")
    print(f"  with a prior position coded: {len(codeable)}")
    print(f"  flagged for manual review:   {len(flagged)}")


if __name__ == "__main__":
    main()
