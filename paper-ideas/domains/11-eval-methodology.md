# Research Paper Ideas: The Methodology of LLM Evals for Everyday Life

## Gaps identified (2025–2026 state of the art)

- **Interpretive-construct gap in LLM annotation**: Comparative studies ([SAGE 2026](https://journals.sagepub.com/doi/10.1177/16094069261426100), [ACL Findings 2025](https://aclanthology.org/2025.findings-naacl.361/), [Than et al., *Sociological Methods & Research* 2025](https://journals.sagepub.com/doi/10.1177/00491241251339188)) converge on one finding: LLMs match humans on concrete codes but diverge on constructs requiring interpretive work — yet nobody has systematically *mapped* which properties of a construct predict divergence, or built decision procedures around it (cf. [Carlson, *SMJ* 2026 guidelines](https://sms.onlinelibrary.wiley.com/doi/10.1002/smj.70023), [SILICON reproducibility protocol](https://arxiv.org/pdf/2412.14461)).
- **"Reliability without validity"**: LLM-as-judge critiques ([arXiv 2508.18076](https://arxiv.org/html/2508.18076v1), [arXiv 2606.19544](https://arxiv.org/pdf/2606.19544), [NeurIPS 2025 D&B construct-validity review](https://openreview.net/forum?id=mdA5lVvNcU)) show judges are consistent but of unexamined construct validity; almost no work imports psychometric validation (convergent/discriminant/known-groups) into social-science LLM coding pipelines.
- **Deliberation changes the measurement, not just the accuracy**: Multi-agent work shows deliberating LLMs exhibit a utilitarian boost and conformity effects ([Many LLMs Are More Utilitarian Than One](https://arxiv.org/html/2507.00814v1), [multi-agent social science paradigm](https://arxiv.org/html/2506.01839v1)) — deliberation as a *validation instrument* for coding (not for accuracy gains) is unexplored.
- **Silicon-sampling debate lacks naturalistic ground truth**: Synthetic-respondent critiques ([analytic flexibility threat](https://arxiv.org/pdf/2509.13397), [synthetic social agents eval](https://arxiv.org/pdf/2509.26080)) test against surveys (ANES), not against everyday-life data where real verdicts exist (e.g., AITA), leaving ecological validity untested.
- **No positionality/hermeneutics of the machine coder**: Human coders disclose standpoint; nothing operationalizes "whose reading" an LLM produces, and no benchmark exists for prompt sensitivity as construct-validity failure.

---

### 1. The Interpretive Gradient: Predicting When LLM Coders Fail Human Agreement
- **Pitch**: Turn the scattered "LLMs fail on interpretive codes" finding into a predictive theory by coding the *codes themselves* and modeling agreement.
- **RQ**: Which measurable properties of a qualitative construct (inference depth, context window needed, latent vs. manifest content, normative loading, definitional consensus among humans) predict LLM–human agreement?
- **Data**: 40–60 constructs pooled across his existing pipelines — AITA moral categories, Red Pill discourse codes, podcast authority frame — plus 3–4 public codebooks (e.g., GoEmotions, hate-speech taxonomies); ~200 doubly human-coded items per construct.
- **Eval design**: GPT-5-class, Claude, Gemini, one open model code every item; meta-regression of κ(LLM, human-consensus) on construct features (rated by independent experts); pre-registered; cross-validated to predict agreement on held-out constructs before any human coding.
- **Why him**: He owns exactly the multi-construct, multi-domain corpora needed; humanities training lets him theorize "interpretive depth" rigorously.
- **Scores**: Impact 5/5, Virality 3/5, Importance 5/5

### 2. AITA in the Wild: Ecological Validity of Moral-Judgment Benchmarks
- **Pitch**: Test whether performance on curated moral benchmarks (ETHICS, Scruples, MoralChoice) transfers to naturalistic everyday dilemmas — his 10K AITA corpus as the "field site."
- **RQ**: Do model rankings and error patterns on lab-style moral benchmarks predict alignment with crowd verdicts on real dilemmas, and what features (narrative messiness, missing information, stakes ambiguity) break transfer?
- **Data**: His AITA corpus (with crowd verdicts) + 3 standard moral benchmarks; a "messiness" feature layer coded per item.
- **Eval design**: 7+ models scored on both; rank-correlation of leaderboards across settings; item-response-theory analysis of which item features drive divergence; ablation converting AITA posts into "benchmark-style" cleaned vignettes to isolate the ecological gap.
- **Why him**: Direct sequel to FAccT/NeurIPS papers; the cleaned-vignette ablation is a clean causal story reviewers love.
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 3. Deliberation as Audit: Multi-LLM Panels as a Validity Instrument for Machine Coding
- **Pitch**: Repurpose his multi-LLM deliberation setup from moral judgment to qualitative coding, testing whether inter-model deliberation flags exactly the items human coders would dispute.
- **RQ**: Does disagreement/convergence dynamics in a GPT–Claude–Gemini coding panel predict human inter-coder disagreement better than single-model confidence or self-consistency sampling?
- **Data**: Podcast authority corpus (has human double-coding) + AITA subset; ~2,000 items.
- **Eval design**: Three conditions — solo coder, self-consistency (k samples), 3-model deliberation with argument exchange; outcome: AUC predicting human-disputed items; also test for conformity artifacts (does deliberation *erase* legitimate ambiguity, echoing the utilitarian-boost finding?).
- **Why him**: Literally extends his NeurIPS deliberation work into his unpublished pipeline; nobody else has both pieces.
- **Scores**: Impact 4/5, Virality 4/5, Importance 4/5

### 4. The Machine's Standpoint: Operationalizing Positionality for LLM Coders
- **Pitch**: Give LLM coders the reflexivity statement human qualitative researchers are required to produce — empirically, not rhetorically.
- **RQ**: Can we measure an LLM's "positionality" (systematic interpretive tilt) by comparing its codings against demographically and epistemically diverse human coder panels, and does persona steering shift it predictably?
- **Data**: 500 Red Pill + 500 right-wing podcast segments coded by a stratified human panel (gender, politics, religiosity; via Prolific) on contested constructs (misogyny, irony, victimhood claims).
- **Eval design**: Estimate each coder's (human and LLM) position in disagreement space (cultural-consensus / IRT models); locate default vs. persona-prompted LLMs relative to human clusters; test whether "neutral" prompting is actually a specific standpoint.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5
- **Why him**: The hermeneutics framing is his signature; he can speak to both *Qualitative Inquiry* and FAccT audiences.

### 5. Prompt Sensitivity Is a Construct-Validity Problem: A Psychometric Audit Protocol
- **Pitch**: Reframe prompt brittleness — usually treated as an engineering nuisance — as measurement error, and build the test–retest/parallel-forms toolkit for LLM annotation.
- **RQ**: How much of LLM coding variance is attributable to construct-irrelevant prompt features (ordering, synonyms, formatting, persona), and can generalizability theory (G-theory) decompose it?
- **Data**: AITA + podcast corpora; 30+ semantically equivalent prompt variants per construct, generated systematically.
- **Eval design**: Fully crossed design (items × prompts × models × temperature); G-study variance decomposition; deliver a "dependability coefficient" researchers can report, plus a minimal-variant protocol (how many paraphrases suffice); release as open toolkit.
- **Why him**: Bridges psychometrics and his eval engineering; instantly citable methods contribution; D-Lab can teach the toolkit.
- **Scores**: Impact 5/5, Virality 3/5, Importance 5/5

### 6. EverydayBench: Building a Living Benchmark from Naturalistic Dilemmas with Documented Construct Validity
- **Pitch**: Answer the NeurIPS construct-validity critique by constructing a benchmark *the right way* — from naturalistic data, with a full validity dossier — as a template others copy.
- **RQ**: What does a benchmark construction pipeline look like when convergent, discriminant, and ecological validity are documented at every stage?
- **Data**: Fresh advice/judgment communities beyond AITA (r/relationships, r/legaladvice, r/work) with community verdicts; temporal splits to resist contamination.
- **Eval design**: Multi-stage: construct definition → sampling audit → human verdict reliability → contamination checks (n-gram + membership inference) → known-groups validation; evaluate 8 models; publish validity dossier alongside leaderboard.
- **Why him**: Combines his Reddit-mining pedigree (ICWSM 2020) with eval engineering; benchmark papers accrue citations fast.
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 7. Silicon Coders vs. Silicon Subjects: Do LLMs Judge Dilemmas the Way They Simulate Judges?
- **Pitch**: Collide the silicon-sampling debate with his AITA data: compare LLMs *as* moral judges to LLMs *simulating* demographic panels of judges, against real crowd verdicts.
- **RQ**: Does demographic conditioning (silicon sampling) improve or distort alignment with actual verdict distributions on everyday dilemmas, relative to unconditioned judgment?
- **Data**: AITA items where commenter-level verdict distributions are recoverable; ANES-style demographic profiles for conditioning.
- **Eval design**: Conditions: bare model, persona-conditioned panels, deliberating panels; measure distributional alignment (Wasserstein distance to real verdict distributions), not just majority accuracy; test the socially-sensitive-item failure documented in silicon-sampling critiques.
- **Why him**: Uniquely positioned — he has real verdict distributions, which survey-based silicon-sampling work lacks.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 8. Codebook Drift: Version Instability of LLM Coders Across Model Updates
- **Pitch**: Longitudinal measurement invariance study — does the "same" coding instrument still measure the same thing after a model update?
- **RQ**: How much do codings of identical items drift across model versions/snapshots, and does drift concentrate in interpretive constructs; can anchor-item calibration correct it?
- **Data**: His frozen podcast/AITA coded sets recoded at each major model release over 12–18 months (plus retrospective via pinned API snapshots and open-weights checkpoints).
- **Eval design**: Differential item functioning analysis across versions; identify anchor items; propose and test a recalibration procedure that restores comparability of longitudinal research pipelines.
- **Why him**: He runs persistent pipelines that this problem actively threatens; strong reproducibility-crisis hook.
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 9. Can LLMs Do Hermeneutics? Irony, Dogwhistles, and Layered Meaning as the Hard Frontier
- **Pitch**: A targeted adversarial eval of the one thing his interpretive corpora are full of and benchmarks ignore: non-literal, audience-dependent meaning.
- **RQ**: Can LLMs distinguish surface content from pragmatic function (irony, plausible deniability, coded appeals) in extremist-adjacent discourse, and do failures inflate reported LLM-coder agreement on toxicity-style constructs?
- **Data**: Red Pill + podcast corpora; expert-annotated set of ~1,500 segments with layered labels (literal content / pragmatic intent / audience uptake, the latter validated via reply threads).
- **Eval design**: Models code at each layer; confusion analysis between layers; key result: show standard single-layer agreement metrics overstate validity when meaning layers dissociate.
- **Why him**: This is digital hermeneutics made operational — his career thesis as an empirical paper.
- **Scores**: Impact 4/5, Virality 4/5, Importance 4/5

### 10. Teaching the Machine Coder: A Pedagogical RCT on Validity Practices for Social Scientists
- **Pitch**: Use D-Lab workshops as a natural lab: does teaching a validation-first protocol change whether social scientists produce valid LLM-annotation studies?
- **RQ**: Do researchers trained with a structured validity protocol (agreement floors, prompt-variant checks, human audit sampling) produce measurably more reliable/valid LLM coding pipelines than those given standard tool training?
- **Data**: 4–6 D-Lab workshop cohorts (~120 researchers), randomized to curriculum arms; their take-home coding projects on shared corpora as outcomes.
- **Eval design**: Blinded scoring of projects (validation steps performed, agreement achieved, overclaiming in write-ups); pre/post knowledge measures; release the curriculum and protocol as the SILICON-style checklist for social science.
- **Why him**: Only he has the classroom; converts curriculum leadership into a publishable methods+education contribution (e.g., *SMR*, CSCW, or *PNAS Nexus*).
- **Scores**: Impact 4/5, Virality 3/5, Importance 4/5

---

**Sources:** [SAGE 2026 LLM qualitative analysis](https://journals.sagepub.com/doi/10.1177/16094069261426100) · [Carlson SMJ 2026](https://sms.onlinelibrary.wiley.com/doi/10.1002/smj.70023) · [CHI 2026 open-source LLM coding](https://dl.acm.org/doi/10.1145/3772363.3798320) · [NAACL 2025 inductive coding](https://aclanthology.org/2025.findings-naacl.361/) · [SILICON](https://arxiv.org/pdf/2412.14461) · [Than et al. 2025](https://journals.sagepub.com/doi/10.1177/00491241251339188) · [Neither Valid nor Reliable](https://arxiv.org/html/2508.18076v1) · [NeurIPS 2025 construct validity](https://openreview.net/forum?id=mdA5lVvNcU) · [Reliability without Validity](https://arxiv.org/pdf/2606.19544) · [Many LLMs Are More Utilitarian Than One](https://arxiv.org/html/2507.00814v1) · [Analytic flexibility threat](https://arxiv.org/pdf/2509.13397) · [Synthetic social agents](https://arxiv.org/pdf/2509.26080) · [Multi-agent social science](https://arxiv.org/html/2506.01839v1)