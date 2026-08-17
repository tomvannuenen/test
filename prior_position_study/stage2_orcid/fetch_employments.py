"""
Stage 2: Enrich the Frame (ORCID + publication counts)
========================================================
ORCID supplies what OpenAlex structurally cannot: ROLE TITLES with start and
end dates. This is the only source in the pipeline that can separate ladder
from non-ladder, so the study's coverage is bounded by ORCID employment
coverage. Expect well under half the frame to carry usable records; the
uncovered remainder is exported for manual CV lookup rather than dropped
silently.

Also computes `pubs_pre_move`: OpenAlex works published before the appointment
year, used for the conditioning analysis in stage 5.

Authentication
--------------
The public record endpoint is read without a token. The ORCID SEARCH endpoint
requires client credentials; set ORCID_CLIENT_ID / ORCID_CLIENT_SECRET to
enable the (optional) name-based fallback lookup for frame members whose
OpenAlex record carries no ORCID.

Usage:
    python -m stage2_orcid.fetch_employments
    python -m stage2_orcid.fetch_employments --limit 50
    python -m stage2_orcid.fetch_employments --export-manual-queue
"""

import argparse
import csv
import os
from pathlib import Path

from common import (
    DATA_DIR,
    CachedSession,
    EgressBlocked,
    load_config,
    read_jsonl,
    write_jsonl,
)

FRAME_PATH = DATA_DIR / "frame.jsonl"
ENRICHED_PATH = DATA_DIR / "enriched.jsonl"
MANUAL_QUEUE_PATH = DATA_DIR / "manual_lookup_queue.csv"


def make_sessions(cfg: dict):
    orcid_cfg, oa_cfg = cfg["orcid"], cfg["openalex"]
    orcid = CachedSession(
        namespace="orcid",
        requests_per_second=orcid_cfg.get("requests_per_second", 5),
        max_retries=orcid_cfg.get("max_retries", 5),
        backoff_seconds=orcid_cfg.get("backoff_seconds"),
        headers={"Accept": "application/json",
                 "User-Agent": "prior-position-study"},
    )
    token = os.environ.get("ORCID_ACCESS_TOKEN")
    if token:
        orcid.session.headers["Authorization"] = f"Bearer {token}"

    openalex = CachedSession(
        namespace="openalex",
        requests_per_second=oa_cfg.get("requests_per_second", 8),
        max_retries=oa_cfg.get("max_retries", 5),
        backoff_seconds=oa_cfg.get("backoff_seconds"),
        headers={"User-Agent": f"prior-position-study (mailto:{oa_cfg['mailto']})"},
    )
    return orcid, openalex


def normalize_orcid(value: str) -> str:
    """Accept a bare ID or a full https://orcid.org/... URL."""
    if not value:
        return ""
    return value.rstrip("/").split("/")[-1]


def _date_parts(node: dict, key: str):
    """Pull (year, month) out of an ORCID fuzzy-date node."""
    d = (node or {}).get(key) or {}
    if not d:
        return None, None

    def _val(sub):
        v = (d.get(sub) or {}).get("value") if isinstance(d.get(sub), dict) else d.get(sub)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return _val("year"), _val("month")


def parse_affiliations(doc: dict, kind: str) -> list:
    """
    Flatten an ORCID employments/educations document into records.

    v3.0 nests summaries inside affiliation-groups; each group may hold
    several sources asserting the same post, so we take the first summary per
    group and deduplicate downstream.
    """
    out = []
    summary_key = f"{kind}-summary"
    for group in (doc or {}).get("affiliation-group", []) or []:
        for summary_wrapper in group.get("summaries", []) or []:
            s = summary_wrapper.get(summary_key)
            if not s:
                continue
            org = s.get("organization") or {}
            address = org.get("address") or {}
            start_y, start_m = _date_parts(s, "start-date")
            end_y, end_m = _date_parts(s, "end-date")
            out.append(
                {
                    "org_name": org.get("name"),
                    "org_country": (address.get("country") or "").upper() or None,
                    "org_city": address.get("city"),
                    "role_title": s.get("role-title"),
                    "department": s.get("department-name"),
                    "start_year": start_y,
                    "start_month": start_m,
                    "end_year": end_y,
                    "end_month": end_m,
                    "source": "orcid",
                }
            )
            break  # one summary per affiliation-group is enough
    return out


def dedupe_affiliations(rows: list) -> list:
    """Collapse duplicate assertions of the same post."""
    seen, out = set(), []
    for r in rows:
        key = (
            (r.get("org_name") or "").strip().lower(),
            (r.get("role_title") or "").strip().lower(),
            r.get("start_year"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def fetch_person(session: CachedSession, cfg: dict, orcid_id: str) -> dict:
    """Fetch employments and educations for one ORCID iD."""
    base = cfg["orcid"]["base_url"]
    employments = parse_affiliations(
        session.get_json(f"{base}/{orcid_id}/employments"), "employment"
    )
    educations = parse_affiliations(
        session.get_json(f"{base}/{orcid_id}/educations"), "education"
    )
    # Qualifications sometimes carry the doctorate when educations does not.
    qualifications = parse_affiliations(
        session.get_json(f"{base}/{orcid_id}/qualifications"), "qualification"
    )
    return {
        "employments": dedupe_affiliations(employments),
        "educations": dedupe_affiliations(educations + qualifications),
    }


def pubs_before(session: CachedSession, cfg: dict, openalex_id: str, year: int):
    """Count OpenAlex works published strictly before `year`."""
    if not year:
        return None
    base = cfg["openalex"]["base_url"]
    author_id = openalex_id.rstrip("/").split("/")[-1]
    url = (
        f"{base}/works?filter=author.id:{author_id},publication_year:<{year}"
        f"&per-page=1&mailto={cfg['openalex']['mailto']}"
    )
    doc = session.get_json(url)
    if not doc:
        return None
    return (doc.get("meta") or {}).get("count")


def export_manual_queue(rows: list, path: Path):
    """
    Write frame members with no usable ORCID employment history.

    These need departmental page / CV lookup. The brief plans for this: ORCID
    alone will not cover the sample.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["openalex_id", "orcid", "display_name", "field",
             "current_institution", "appointment_year_proxy", "reason"]
        )
        for r in rows:
            w.writerow(
                [
                    r["openalex_id"],
                    r.get("orcid") or "",
                    r.get("display_name") or "",
                    r["field"],
                    (r.get("current_institution") or {}).get("name") or "",
                    r.get("appointment_year_proxy") or "",
                    r.get("_manual_reason", ""),
                ]
            )
    print(f"Manual lookup queue: {len(rows)} people -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", default=str(FRAME_PATH))
    ap.add_argument("--output", default=str(ENRICHED_PATH))
    ap.add_argument("--limit", type=int, help="process only the first N frame rows")
    ap.add_argument("--export-manual-queue", action="store_true", default=True)
    ap.add_argument("--skip-pub-counts", action="store_true")
    args = ap.parse_args()

    cfg = load_config("fields.yaml")
    frame = read_jsonl(Path(args.frame))
    if not frame:
        raise SystemExit(f"No frame at {args.frame}. Run stage 1 first.")
    if args.limit:
        frame = frame[: args.limit]

    orcid_session, oa_session = make_sessions(cfg)
    enriched, manual = [], []

    try:
        for i, row in enumerate(frame, 1):
            if i % 25 == 0:
                print(f"  {i}/{len(frame)}")

            orcid_id = normalize_orcid(row.get("orcid"))
            record = {"employments": [], "educations": []}
            if orcid_id:
                record = fetch_person(orcid_session, cfg, orcid_id)

            row["orcid_employments"] = record["employments"]
            row["orcid_educations"] = record["educations"]

            if not args.skip_pub_counts:
                row["pubs_pre_move"] = pubs_before(
                    oa_session, cfg, row["openalex_id"], row.get("appointment_year_proxy")
                )

            titled = [e for e in record["employments"] if e.get("role_title")]
            if len(titled) < 2:
                row["_manual_reason"] = (
                    "no orcid id" if not orcid_id
                    else "no employment records" if not record["employments"]
                    else "fewer than two titled employments"
                )
                manual.append(row)
            enriched.append(row)
    except EgressBlocked as exc:
        raise SystemExit(
            f"\nNetwork blocked: {exc}\n"
            "ORCID/OpenAlex are unreachable from this environment."
        )

    write_jsonl(Path(args.output), enriched)
    covered = len(enriched) - len(manual)
    pct = 100 * covered / len(enriched) if enriched else 0
    print(f"\nEnriched {len(enriched)} people -> {args.output}")
    print(f"ORCID employment coverage: {covered}/{len(enriched)} ({pct:.1f}%)")
    print(f"ORCID cache: {orcid_session.stats}")

    if args.export_manual_queue and manual:
        export_manual_queue(manual, MANUAL_QUEUE_PATH)


if __name__ == "__main__":
    main()
