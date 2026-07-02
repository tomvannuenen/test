## Gaps identified

- **Hiring-bias audits are saturated with synthetic resumes; naturalistic advice corpora are untouched.** 2025-26 work covers name-swapped resume screening ([Brookings](https://www.brookings.edu/articles/gender-race-and-intersectional-bias-in-ai-resume-screening-via-language-model-retrieval/)), LLM self-bias toward AI-written resumes ([WebProNews](https://www.webpronews.com/llms-show-strong-self-bias-in-resume-screening-giving-ai-written-applications-a-big-edge-over-human-ones/)), and human mirroring of AI hiring bias ([phys.org](https://phys.org/news/2025-11-people-mirror-ai-hiring-biases.html)) — but no one evaluates LLMs on the messy, real workplace disputes people actually bring to them.
- **AI-mediated communication research is survey/experiment-based, not corpus-based.** Cardon & Coman's professionalism-trust paradox ([USC Marshall](https://www.marshall.usc.edu/news/ai-assisted-emails-may-put-trustworthiness-at-risk-in-workplace-communications), [ScienceDaily](https://www.sciencedaily.com/releases/2025/08/250811104226.htm)) uses stimulus emails; nobody has audited what LLMs actually write at scale (rejections, PIPs, apologies) or how workers discuss receiving them.
- **Advice-audit designs exist but are narrow.** The salary-negotiation perturbation audit ([PLOS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318500), 98,800 prompts) varies demographics on one task; no study varies *which side of a labor conflict* the model is asked to advise, or whose interests its "neutral" advice serves.
- **Employment lawyers report workers losing winnable cases via ChatGPT-drafted documents** ([Tamir Litigation](https://tamirlitigation.com/2025/05/why-employees-should-think-twice-before-using-chatgpt-in-workplace-disputes/)) — but no systematic eval of LLM quality/harm on real employment-dispute questions exists.
- **The discourse of "AI at work" itself (r/antiwork etc.) has no computational-hermeneutic treatment** in the style of his Red Pill / AITA work.

---

### 1. Am I the Asshole at Work? Multi-LLM Judgment of Workplace Conflicts
- **Pitch:** Direct AITA-pipeline extension: do LLMs side with workers or managers when judging real workplace conflicts?
- **RQ:** Do LLMs exhibit systematic pro-employer or pro-employee alignment when adjudicating naturalistic workplace disputes, and does it vary by narrator role, industry, and model?
- **Data:** 15-20k posts with verdict-bearing comments from r/antiwork, r/AskManagers, r/work, r/jobs; role/industry metadata via LLM extraction, human-validated.
- **Eval design:** GPT-5, Claude, Gemini, Llama render verdicts + justifications; compare to community consensus; perspective-flip each dispute (retell from manager's POV) to measure narrator-sympathy bias; taxonomize justification frames (proceduralism, loyalty, legality, dignity) with validated LLM coder (κ vs. two human coders).
- **Why him:** Literally his AITA methodology transplanted to a higher-stakes domain; instant reviewer legibility.
- **Scores:** Impact 5/5, Virality 5/5, Importance 4/5

### 2. Whose Side Is the Bot On? Dual-Advocacy Audits of LLM Workplace Advice
- **Pitch:** Ask each model to advise *both* the employee and the manager on the same real dispute, and measure whether its "neutral" counsel is symmetric.
- **RQ:** When LLMs advise opposing parties in identical conflicts, do they systematically transfer more strategic resources (legal framings, escalation options, documentation tactics) to one side?
- **Data:** 3,000 disputes from r/antiwork + r/AskManagers where both perspectives are reconstructable; matched prompt pairs.
- **Eval design:** 5+ models × employee/manager framing; LLM-coder scores advice on assertiveness, legal-rights mention, de-escalation pressure, warning-to-comply; paired asymmetry metrics; validate coder against annotated subsample; employment-law expert rates accuracy on a stratified sample.
- **Why him:** Extends the PLOS One salary-audit paradigm from demographics to structural power — his fairness + Such collaboration wheelhouse.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 3. The Rejection Machine: Auditing LLM-Written Bad News at Scale
- **Pitch:** Large-scale audit of the rejection letters, PIP notices, and layoff emails LLMs generate when given real cases.
- **RQ:** How do LLM-drafted bad-news communications vary in warmth, blame attribution, legalistic hedging, and false hope across models, recipient demographics, and seniority?
- **Data:** Scenario bank derived from 2,000 real r/recruitinghell and r/jobs rejection/firing narratives; demographic and seniority perturbations.
- **Eval design:** 6 models draft messages; theory-derived coding frame from bad-news-delivery literature (buffering, justification, face-work) applied by validated LLM coder; perturbation analysis for demographic disparities; human panel rates a subsample for perceived sincerity (linking to Cardon & Coman's trust paradox).
- **Why him:** Theory-derived coding frame + multi-model audit is exactly his podcast-study machinery.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 4. Deliberating the Dismissal: Multi-LLM Juries on Contested Firings
- **Pitch:** Put GPT, Claude, and Gemini in structured deliberation over real "was this firing fair?" cases and study opinion dynamics.
- **RQ:** Does multi-LLM deliberation on employment disputes converge toward employer-protective consensus, and which rhetorical moves drive verdict changes?
- **Data:** 500 richly detailed contested-termination narratives (r/antiwork, r/legaladvice, r/AskHR) with community verdicts.
- **Eval design:** His NeurIPS deliberation protocol: independent verdicts → 3 rounds of exchange → final verdict; measure verdict drift, conformity, position-by-model-identity effects; discourse-code deliberation turns (concession, authority citation, reframing); ablate anonymized vs. named models.
- **Why him:** Direct reuse of his multi-LLM deliberation infrastructure on a new domain.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 5. "HR Speak": Do LLMs Launder Managerial Power into Neutral Language?
- **Pitch:** A discourse-analytic eval of how LLMs rewrite blunt managerial messages into "professional" text — and what gets erased.
- **RQ:** When LLMs "professionalize" workplace communication, do they systematically remove accountability, agency, and worker-protective information (passive voice, nominalization, deleted actors)?
- **Data:** 2,000 raw manager drafts/quotes from r/AskManagers and r/managers; each rewritten by 5 models under "make this professional" prompts.
- **Eval design:** Before/after analysis with a critical-discourse-analysis coding frame (agent deletion, mitigation, euphemism) operationalized as an LLM coder validated against CDA-trained annotators; syntactic measures (passivization rates) as convergent evidence.
- **Why him:** Marries his hermeneutic/CDA training with LLM-coder validation — almost no one else can referee both halves.
- **Scores:** Impact 4/5, Virality 4/5, Importance 5/5

### 6. Reference-Check Roulette: Auditing LLM-Drafted References and Their Silences
- **Pitch:** Audit what LLMs write — and pointedly don't write — when drafting employment references for described workers.
- **RQ:** Do LLM-generated reference letters encode demographic and role-based disparities in the doubt-raisers, agentic/communal language, and strategic omissions known from human letter research?
- **Data:** 1,500 worker vignettes distilled from real r/jobs and r/AskManagers reference-request threads; controlled demographic/occupation perturbations.
- **Eval design:** 5 models × perturbations; apply the established letter-of-recommendation bias lexicon plus LLM-coded "hedging/omission" measures; downstream test: feed letters back to LLM screeners (connecting to the self-bias literature) to quantify callback impact.
- **Why him:** Word-level bias discovery (ICWSM 2020) upgraded to generative audit with a closed feedback loop.
- **Scores:** Impact 4/5, Virality 3/5, Importance 4/5

### 7. The Ghost in the Inbox: Detecting and Theorizing "AI Voice" Accusations at Work
- **Pitch:** Study the new social practice of accusing colleagues and employers of sending AI-written messages.
- **RQ:** What triggers "this was written by AI" accusations in workplace contexts, how accurate are the folk heuristics, and what moral work do accusations perform?
- **Data:** 10k+ Reddit posts/comments (2023-2026) containing AI-authorship accusations about workplace messages (rejections, condolences, reviews), via keyword + LLM-filtered retrieval.
- **Eval design:** LLM coder applies a grounded-then-theorized frame (cue cited, moral charge, relational damage); benchmark folk cues against detector performance on a constructed human/AI email set; temporal trend analysis.
- **Why him:** Digital hermeneutics of an emergent vernacular literacy — his Red Pill scaled-reading approach meets his eval engineering.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 8. Chatbot as Shop Steward: Evaluating LLM Advice on Organizing and Collective Action
- **Pitch:** Test whether LLMs chill or support legally protected collective activity when workers ask about unionizing, strikes, and pay transparency.
- **RQ:** Do LLMs accurately convey protected-activity rights (NLRA §7), and do safety behaviors (hedging, both-sidesing, discouragement) suppress lawful organizing advice relative to individual-exit advice?
- **Data:** 1,200 real organizing questions from r/union, r/antiwork, r/WorkersRights; labor-law expert-annotated gold answers for a 200-item core set.
- **Eval design:** 6 models incl. open-weights; measure legal accuracy, hedging density, discouragement framing (LLM coder + expert validation); contrast matched "should I quit?" vs. "should we organize?" prompts to isolate collective-action penalty.
- **Why him:** Fairness-as-power analysis with an eval harness; policy-legible and novel.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 9. Performance Review, Reviewed: LLMs as Coders of Evaluative Authority
- **Pitch:** Apply his podcast-study authority coding frame to performance-review language — who gets described as having potential, and in whose voice?
- **RQ:** How do epistemic authority and self-positioning moves distribute across human- vs. LLM-drafted performance reviews, and across ratee demographics?
- **Data:** Public review-snippet corpora (shared reviews on r/jobs/r/cscareerquestions, published datasets) plus LLM-generated reviews from matched employee profiles.
- **Eval design:** Port his validated coding frame (authority claims, authorization moves, positioning) to reviews; multi-model generation × perturbation; human-LLM coder agreement study as a standalone methods contribution.
- **Why him:** Direct reuse of his unpublished pipeline's frame — fastest path to a second paper from that infrastructure.
- **Scores:** Impact 3/5, Virality 3/5, Importance 4/5

### 10. "AI Took the Humanity Out of My Firing": The Vernacular Discourse of AI at Work
- **Pitch:** A computational-hermeneutic map of how workers narrate AI's arrival in their workplaces across five years of Reddit.
- **RQ:** What recurring narrative frames (deskilling, surveillance, absurdity, complicity, relief) structure worker talk about AI, and how have they shifted 2022-2026?
- **Data:** ~500k posts/comments mentioning workplace AI from r/antiwork, r/jobs, r/WorkReform, r/cscareerquestions.
- **Eval design:** Theory-derived frame taxonomy (labor-process theory + his authority categories) applied by an LLM coder validated on 1,000 human-coded items; embedding-based frame drift over time (ICWSM 2020 method, updated); multi-model coder-agreement comparison as robustness.
- **Why him:** The Scripted Journeys move — algorithms as cultural narrative — executed with modern eval rigor.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

---

**Sources:** [Brookings resume-screening bias](https://www.brookings.edu/articles/gender-race-and-intersectional-bias-in-ai-resume-screening-via-language-model-retrieval/) · [LLM self-bias in screening](https://www.webpronews.com/llms-show-strong-self-bias-in-resume-screening-giving-ai-written-applications-a-big-edge-over-human-ones/) · [Humans mirror AI hiring bias](https://phys.org/news/2025-11-people-mirror-ai-hiring-biases.html) · [USC Marshall AI-email trust study](https://www.marshall.usc.edu/news/ai-assisted-emails-may-put-trustworthiness-at-risk-in-workplace-communications) · [ScienceDaily on AI emails and trust](https://www.sciencedaily.com/releases/2025/08/250811104226.htm) · [PLOS One salary-negotiation audit](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318500) · [Tamir Litigation on ChatGPT in disputes](https://tamirlitigation.com/2025/05/why-employees-should-think-twice-before-using-chatgpt-in-workplace-disputes/)