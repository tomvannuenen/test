# AI in Domestic and Home Life — 10 Paper Ideas for Tom van Nuenen

## Gaps identified (from 2025–2026 searches)

- **Smart-home LLM work is engineering-only.** Benchmarks like SmartBench (arXiv 2603.06636), HomeBench, and voice-query-rejection work evaluate command execution, not the *normative* judgments LLM assistants increasingly make inside families (who's right in a dispute, whose preferences win). No FAccT/CSCW-style moral audit of "home hub" LLMs exists, even as Alexa+ (Claude-powered) and Samsung's multi-LLM Home Hub ship.
- **Social sycophancy is documented, but not in domestic stakes.** ELEPHANT (arXiv 2505.13995) shows LLMs preserve users' face 45+ pp more than humans on AITA-type queries, but nobody has tested what this does in *asymmetric household relationships* (parent–child, spousal, elder care), where sycophancy toward one party is bias against a co-resident.
- **Parenting-advice audits are thin and non-naturalistic.** One PLOS One (Nov 2025) study found LLMs assign caregiving to mothers; a CHI 2026 paper probes parents' moderation desires with synthetic scenarios. No large-scale audit over naturalistic parenting corpora (r/Parenting, r/AmIOverreacting) with theory-derived coding frames.
- **Gender/labor bias audits stop at occupations.** Extensive occupational-persona audits (arXiv 2510.21011, cross-lingual audits) exist, but *unpaid domestic labor allocation* — chore division, mental load, kin-keeping — is essentially unaudited despite being the highest-volume household use case.
- **No deliberation designs in domestic AI.** Multi-LLM deliberation (his NeurIPS 2025 setup) has never been applied to family-conflict adjudication, and no one has compared LLM verdicts to *within-household* human disagreement rather than crowd consensus.

---

### 1. Alexa in the Middle: Auditing LLM Home Assistants as Household Arbiters
- **Pitch**: The first normative (not functional) audit of LLMs acting as in-home referees, testing whose side the assistant takes when household members conflict.
- **RQ**: When two household members make incompatible requests or present a dispute, which party do LLM assistants favor, and do verdicts shift with cues of gender, age, and household role?
- **Data/corpus**: 5,000+ disputes scraped from r/AmITheAsshole, r/relationship_advice, and r/JUSTNOMIL filtered to co-residing parties; rewritten into "smart speaker overheard both sides" prompts; systematic persona-swap variants.
- **Eval design**: GPT-5, Claude, Gemini, Llama judge each dispute; measure verdict direction, flip rates under role/gender swaps (mirroring the 2603.05651 perturbation logic), and agreement with Reddit verdicts; human validation on 500 items.
- **Why him**: Direct extension of his FAccT AITA pipeline into the deployment context (Alexa+ runs on Claude); reuses his multi-model comparison machinery.
- **Scores**: Impact 5/5, Virality 5/5, Importance 5/5

### 2. Who Loads the Dishwasher? A Large-Scale Audit of LLM Chore Allocation
- **Pitch**: Audit whether LLMs, asked to divide household labor fairly, reproduce gendered allocations of chores and "mental load."
- **RQ**: Do LLMs assign visible chores, invisible labor (scheduling, kin-keeping), and childcare differently by partner gender, and does "fair division" advice track equity theory or stereotype?
- **Data/corpus**: Chore-dispute posts from r/marriage, r/AskWomenOver30, r/workingmoms; plus a synthetic factorial grid (couple gender composition × income × hours worked) built from real post templates.
- **Eval design**: 6 models produce chore-division plans; LLM-as-coder labels each task on a sociology-derived frame (routine/intermittent, visible/invisible — Daminger's cognitive-labor taxonomy); measure allocation share by gender with name/pronoun swaps; krippendorff-validated human coding of 400 plans.
- **Why him**: Marries his ICWSM embedding-bias work (gendered language in communities) with his LLM-as-qualitative-coder pipeline and theory-derived frames.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 3. Sycophancy at the Dinner Table: Face-Saving Bias in Family-Conflict Advice
- **Pitch**: Test whether LLM sycophancy systematically advantages the *asking* family member, turning "supportive AI" into a conflict-escalation engine.
- **RQ**: When the same domestic conflict is narrated by each party (parent vs. teen, each spouse), how often do models validate both mutually incompatible positions, and does dual-validation exceed human advisors' rates?
- **Data/corpus**: Paired-perspective conflicts: r/AmITheAsshole posts plus their top dissenting comments; r/raisedbynarcissists vs. r/ParentingThruTrauma; 2,000 conflicts rewritten from both POVs.
- **Eval design**: Feed each POV separately to GPT, Claude, Gemini; measure "dual endorsement rate" (both sides told they're right), extending ELEPHANT metrics; compare against Reddit human advice; validate POV rewrites with human raters.
- **Why him**: His NeurIPS deliberation work already handles perspective-conditioned moral judgment; this weaponizes his POV-flip findings for a home context.
- **Scores**: Impact 5/5, Virality 5/5, Importance 5/5

### 4. Deliberating the Family: Multi-LLM Juries on Domestic Dilemmas
- **Pitch**: Apply his multi-LLM deliberation design to family conflicts, testing whether model "juries" converge toward Reddit consensus or toward systematic pro-parent/pro-partner priors.
- **RQ**: Does deliberation among heterogeneous LLMs reduce verdict instability and sycophancy on domestic dilemmas relative to solo judgment, and whose perspective gains ground during deliberation?
- **Data/corpus**: 3,000 family-tagged AITA dilemmas (he has the pipeline) stratified by relationship type (parent–child, spousal, in-law, sibling).
- **Eval design**: GPT+Gemini+Claude deliberation vs. solo baselines; measure verdict shift direction by relationship role, opinion-change asymmetry across deliberation turns, alignment with human vote distributions (pluralism, per arXiv 2507.17216).
- **Why him**: This is literally his NeurIPS 2025 architecture with a domestic stratification — lowest-cost, fastest-to-publish idea on this list.
- **Scores**: Impact 4/5, Virality 3/5, Importance 4/5

### 5. The Algorithmic Grandmother: Auditing LLM Parenting Advice Across Cultures and Classes
- **Pitch**: Large-scale audit of whether LLM parenting advice encodes a white, middle-class, intensive-parenting ideology.
- **RQ**: How does LLM advice vary when parent personas differ by class, culture, and family structure, and which parenting ideology (Lareau's concerted cultivation vs. natural growth) do models default to?
- **Data/corpus**: 8,000 questions from r/Parenting, r/daddit, r/breakingmom, r/AttachmentParenting; persona-augmented variants (single parent, low-income, immigrant household).
- **Eval design**: 6 models answer; LLM-as-coder applies a Lareau-derived coding frame (structure vs. autonomy, expert-deference, resource assumptions like "hire a sitter"); measure ideology scores by persona; human validation subset; compare against upvoted Reddit answers.
- **Why him**: Theory-derived coding frame + validation + aggregation is exactly his podcast-authority pipeline, transplanted; extends the PLOS One mother-bias finding far beyond gender.
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 6. Scripted Homes: How LLM Assistants Narrate the "Good Household"
- **Pitch**: A digital-hermeneutics study of the normative domestic imaginary embedded in LLM home-management outputs — the sequel to *Scripted Journeys*, indoors.
- **RQ**: What implicit scripts of domesticity (nuclear family, ownership, heteronormativity, productivity) recur when LLMs generate routines, meal plans, and household schedules?
- **Data/corpus**: 10,000 generated "daily household routines" across model × household-configuration grid; plus real routines from r/HomeAutomation and Home Assistant community forums as human baseline.
- **Eval design**: Word-embedding and topic drift analysis (his ICWSM method) plus LLM-coder tagging of script elements; multi-model comparison of script homogeneity; human interpretive close reading of outliers.
- **Why him**: Unique fusion of his book-length work on algorithmic scripting of experience with his computational bias-discovery methods; nobody else can write this paper.
- **Scores**: Impact 3/5, Virality 3/5, Importance 4/5

### 7. When the Smart Home Testifies: LLM Judgments Under Surveillance-Derived Evidence
- **Pitch**: Test whether LLMs treat smart-home data (camera logs, location, purchase history) as legitimate evidence when one family member surveils another.
- **RQ**: Do LLMs sanction intra-household surveillance asymmetrically (parents over teens, husbands over wives, adult children over elders), and does surveillance-derived "evidence" shift moral verdicts?
- **Data/corpus**: r/relationship_advice and r/legaladvice posts involving tracking apps, Ring cameras, AirTags (high-volume topics); controlled variants manipulating who surveils whom.
- **Eval design**: Verdict + justification generation across 5 models; measure sanction rate by surveiller/surveilled role; LLM-coder classifies justification types (safety, property, autonomy); human validation; flip-rate analysis when evidence provenance is revealed.
- **Why him**: Combines his algorithmic-transparency/fairness agenda (IEEE, CSCW, KCL collaboration) with his AITA eval engine; strong policy hook (domestic-abuse tech).
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 8. Manfluencer Advice at Home: Do LLMs Reproduce Red Pill Domestic Ideology?
- **Pitch**: Measure whether LLM relationship/household advice overlaps with the gendered domestic prescriptions of the manosphere he mapped a decade ago.
- **RQ**: How much do LLM answers to relationship-dynamics questions (headship, breadwinning, "gatekeeping") resemble Red Pill discourse vs. mainstream counseling discourse, and do jailbroken/persona-framed prompts close that gap?
- **Data/corpus**: His existing The Red Pill corpus + r/marriedredpill + AskTRP question sets; counter-corpus from r/Marriage and licensed-therapist content (e.g., Gottman blog).
- **Eval design**: Embedding-similarity of model answers to the two discourse poles (his ICWSM method); LLM-coder applies his enemy-construction/authority frame from the podcast study; multi-model + persona-prompt comparison; human coding validation.
- **Why him**: Reunites his two flagship strands — Red Pill hermeneutics and LLM evals — in one paper; the podcast coding frame transfers almost verbatim.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 9. The Child in the Loop: LLMs Judging Parent–Child Conflicts vs. Parents and Teens
- **Pitch**: Compare LLM verdicts on parent–teen conflicts against verdicts from actual parents and teens, testing for adultist bias in models marketed as family mediators.
- **RQ**: Do LLMs systematically side with parental authority in parent–teen disputes relative to human judgment distributions, and does telling the model the asker's age change verdicts?
- **Data/corpus**: Parent–teen conflicts from r/AmItheAsshole (age-tagged), r/insaneparents, r/Parenting; existing comment-vote distributions as human baseline split by commenter-disclosed parental status.
- **Eval design**: 6 models judge; verdict alignment vs. parent-flavored and teen-flavored human subsamples; asker-age perturbation flips; LLM-coder tags authority-justifying rhetoric using his authorization-moves frame; validation on 300 items.
- **Why him**: Extends his pluralistic-judgment interest (whose consensus is the ground truth?) and imports the epistemic-authority frame from the podcast study.
- **Scores**: Impact 4/5, Virality 4/5, Importance 4/5

### 10. Companion or Co-Resident? Auditing AI-Companion Advice That Reshapes Household Bonds
- **Pitch**: Audit how companion chatbots respond when users describe household conflict, testing whether they encourage repair with cohabitants or displacement toward the bot.
- **RQ**: Given naturalistic disclosures of loneliness or family conflict, how often do companion-tuned vs. general LLMs recommend reconnecting with household members versus deepening the human–AI bond, and how does this vary by user age framing?
- **Data/corpus**: Public r/replika and r/CharacterAI posts describing household situations; standardized disclosure vignettes derived from them (elderly widow, isolated teen, new parent).
- **Eval design**: Compare Replika-style persona prompts on open models vs. GPT/Claude/Gemini defaults; LLM-as-coder labels responses (repair-directed, bot-directed, neutral) with a parasocial-attachment frame; escalation analysis over multi-turn conversations; human validation; measure age-framing effects.
- **Why him**: His parasocial-authority and self-positioning coding frame (podcast study) maps directly onto companion bots; timely post-Character.AI litigation, strong FAccT fit.
- **Scores**: Impact 5/5, Virality 5/5, Importance 5/5

---

**Sources:** [SmartBench](https://arxiv.org/pdf/2603.06636) · [Voice-assistant query rejection benchmark](https://arxiv.org/pdf/2512.10257) · [Fragility of Moral Judgment in LLMs](https://arxiv.org/abs/2603.05651) · [His FAccT 2025 paper](https://dl.acm.org/doi/10.1145/3715275.3732044) · [ELEPHANT: social sycophancy](https://arxiv.org/html/2505.13995) · [Pluralistic Moral Gap](https://arxiv.org/html/2507.17216) · [PLOS One gender-roles-in-parenting audit](https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0335706) · [CHI 2026 parents & GenAI moderation](https://dl.acm.org/doi/10.1145/3772318.3791622) · [Occupational persona audit](https://arxiv.org/html/2510.21011)