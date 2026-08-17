# Prior-Position Study

For scholars now holding ladder-rank positions at US universities in
computational social science and adjacent fields, what position did they hold
immediately before?

- **Path A** — ladder-rank post at a non-US institution (Hong Kong, Singapore,
  continental Europe, UK, Australia) → US ladder-rank
- **Path B** — non-ladder US post (adjunct, lecturer, research scientist,
  project scientist, academic coordinator, extended postdoc) → US ladder-rank

The deliverable is **base rates**, not a causal estimate.

---

## Status: built and tested offline; not yet run against live data

The pipeline is complete and every offline stage is verified against fixtures
(`python run_pipeline.py --selftest`, 93 assertions). **No real data has been
collected**, because this environment's egress policy denies the entire
scholarly-API class at the gateway:

```
api.openalex.org        403 (CONNECT denied by policy)
pub.orcid.org           403
api.crossref.org        403
api.ror.org             403
api.semanticscholar.org 403
```

Retrying or routing around an organization policy denial is not appropriate, so
stages 1–2 have never executed. Run them from an environment where those hosts
are permitted, or ask an admin to allow them for this one.

What that means for confidence in the code:

| Layer | Status |
|---|---|
| Title mapping, trajectory logic, path classification | **Verified** — 93 assertions over 24 handcrafted coding hazards |
| Validation harness (kappa, agreement, disagreement report) | **Verified** end to end |
| Analysis tables + notebook | **Verified** — executes cleanly, renders every table |
| OpenAlex/ORCID request and parsing layer | **Unverified** — never run against a live endpoint |

The parsers were written against the documented response shapes for OpenAlex
`/authors` and ORCID v3.0 `/employments`, but shapes drift. Expect to spend the
first live run fixing field paths in `stage1_frame/build_frame.py` and
`parse_affiliations()` in `stage2_orcid/fetch_employments.py`. Everything
downstream of `enriched.jsonl` should work unchanged.

---

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py --selftest          # offline, no network
python run_pipeline.py --stage all         # needs OpenAlex + ORCID

# or stage by stage
python run_pipeline.py --stage 1 --field communication --limit 200
python run_pipeline.py --stage 2
python run_pipeline.py --stage 3
python run_pipeline.py --stage 4 --n 50    # export validation sample
python run_pipeline.py --stage 5
```

Start with `--stage 1 --field communication --limit 200` to confirm the
OpenAlex filters return what you expect before harvesting all six fields.

Every API response is cached under `data/cache/`, so reruns are free and the
harvest stays auditable — the cache is the raw evidence behind the coded CSV.

---

## Pipeline

| Stage | Module | Does | Network |
|---|---|---|---|
| 1 | `stage1_frame/build_frame.py` | Sampling frame + affiliation timeline from OpenAlex | yes |
| 2 | `stage2_orcid/fetch_employments.py` | Role titles and dates from ORCID; pre-move publication counts | yes |
| 3 | `stage3_coding/code_trajectories.py` | Resolve titles → `coded_trajectories.csv` | no |
| 4 | `stage4_validation/validate_coding.py` | Export random 50; compute agreement + kappa | no |
| 5 | `stage5_analysis/base_rates.py` + notebook | Base-rate tables | no |

**Why two sources.** OpenAlex records the *institution* an author published
from in a given year, never the *title*, so it cannot distinguish ladder from
non-ladder on its own. ORCID carries structured role titles with dates but is
self-reported and patchy. OpenAlex builds the frame; ORCID supplies titles;
whatever ORCID misses lands in `data/manual_lookup_queue.csv` for CV lookup
rather than being dropped silently.

---

## The title mapping

`config/title_mapping.yaml` is the intellectual core. Titles are **never**
pattern-matched on the bare English string — they are resolved against the
academic system of the employing country, because the same words mean opposite
things across systems:

| Title | Country | Resolves to | Why |
|---|---|---|---|
| Lecturer | GB | **ladder** | UK ladder entry grade |
| Lecturer | US | non-ladder | teaching track |
| Senior Lecturer | GB | **ladder** | ≈ US Associate |
| Senior Lecturer | SG | non-ladder | NUS/NTU educator track |
| Research Assistant Professor | HK | non-ladder | HK's fixed-term RAP grade |
| Assistant Professor | HK | **ladder** | HK adopted the US ladder |
| Research Fellow | SG | postdoc | standard NUS/NTU postdoc title |
| Adjunkt | DK | **ladder** | Danish assistant professor |
| Adjunkt | SE | non-ladder | Swedish teaching post |
| Profesor Asociado | ES | non-ladder | an adjunct, not a US associate |
| Chargé de Recherche | FR | **ladder** | permanent CNRS research post |

The Hong Kong Research Assistant Professor case matters directly for the
brief's Hong Kong question: naive matching on "assistant professor" would
count every HK RAP as a Path A move.

Resolution order: non-academic employer override → universal statuses
(visiting, emeritus, student) → country system rules, in order → generic
fallback. Every rule carries a `confidence`, and ambiguous research titles are
split by duration (a 2-year "Research Fellow" is a postdoc; a 7-year one is a
staff scientist).

Adding a country: add its ISO code to `country_systems`, add a `systems` block,
add a case to `tests/test_coding.py`, re-run the selftest.

---

## Output schema

`data/coded_trajectories.csv`, one row per person. Beyond the brief's variables
(`prior_rank_class`, `prior_country`, `prior_institution`, `gap_years`,
`phd_country`, `phd_year`, `pubs_pre_move`, `field`):

| Column | Why it exists |
|---|---|
| `appointment_year_source` | `orcid` (a real start date) vs `openalex_affiliation_proxy` (inferred, biased late) |
| `path_a`, `path_b_strict`, `path_b_broad` | precomputed path flags |
| `coding_confidence`, `coding_system`, `coding_rule` | which rule fired, so any row can be audited |
| `needs_manual_review` | flagged: low confidence, missing country, transitional post, or out-of-window |
| `coding_source` | `automated` or `manual`, so hand-codings are excluded from the agreement calculation |

Two definitional choices worth knowing:

- **`gap_years` follows the brief's definition** — appointment year minus prior
  position *start* year. It measures time served in the prior post, not an
  employment gap. The name is the brief's; the semantics are documented here to
  avoid a misread.
- **Path B is reported strict and broad.** Strict is US non-ladder academic.
  Broad adds postdocs held ≥ 4 years (the brief's "extended postdoc"). Standard
  US postdocs are reported separately as a reference category — pooling the
  modal route into Path B would swamp the comparison.

---

## Validation

```bash
python -m stage4_validation.validate_coding export-sample --n 50
# hand-code the blank human_* columns from CVs and faculty pages
python -m stage4_validation.validate_coding compute-agreement \
    --human-file data/validation_sample_coded.csv
```

The sample is a **simple random** 50, as the brief specifies — that is what
gives an unbiased agreement estimate. Machine codings are withheld from the
export so the coder is not anchored. `--stratify confidence` oversamples
uncertain codings instead; it is a diagnostic for locating which national
systems misfire, and its agreement rate is biased downward by construction, so
it is not the headline number.

Below ~85% agreement on `prior_rank_class`, the tool says so and names the
national systems the disagreements came from.

---

## Known limitations

**Selection.** People who take non-US posts and return differ systematically
from those who do not. Base rates describe the observed path; they do not
identify an effect.

**Survivorship.** The frame contains only people who landed a US ladder-rank
job. Everyone who tried either path and failed is invisible. This inflates the
apparent success of both paths, so the *comparison* survives but no absolute
rate is a probability of success.

**Coverage.** ORCID skews younger and more quantitative — a real bias in a
sample spanning STS and digital humanities, and one that will not be uniform
across the six fields. Check the `appointment_year_source` breakdown before
comparing fields.

**Appointment year.** Where ORCID has no start date, the year is inferred from
the first publication carrying the new affiliation, which lags the actual hire.

**Transitional posts.** A one-year visiting position immediately before a hire
is coded as the prior position even when a longer substantive post preceded it.
Those rows are flagged with the masked position named in `coding_note`; re-run
stage 5 with `--exclude-flagged` to see how much they move the rates.

**PhD-granting proxy.** OpenAlex carries no Carnegie classification, so
doctoral status is approximated by institutional research output. Drop a
curated list at `config/phd_granting_institutions.csv` (column: `ror`) to
replace the proxy.

**Descriptive only.** No causal claim is made or supported anywhere.
