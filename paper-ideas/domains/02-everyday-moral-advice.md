# Everyday Moral Reasoning & Advice-Seeking with AI — 10 Paper Ideas for Tom van Nuenen

## Gaps identified (2025–2026 state of the art)

- **Sycophancy work is benchmark-heavy, corpus-light.** [ELEPHANT](https://arxiv.org/pdf/2505.13995) (social sycophancy, ~45pp above human baselines) and [Interaction Context Often Increases Sycophancy in LLMs](https://dl.acm.org/doi/10.1145/3772318.3791915) (CHI 2026) already use AITA items — but as static probes. Nobody has audited sycophancy *as it unfolds in real multi-turn advice conversations*, nor tied it to interpretive/rhetorical categories (self-positioning, authorization) of the kind his podcast pipeline codes.
- **Naturalistic corpora exist but are under-exploited for moral content.** [WildChat](https://www.emergentmind.com/topics/wildchat-dataset) (1M+ conversations) has been mined for [personal disclosures](https://arxiv.org/pdf/2407.11438) and [everyday ethical questions](https://arxiv.org/pdf/2605.24319), but there is no systematic map of *what moral questions people actually bring to LLMs vs. what moral-reasoning evals test* — a distribution-shift audit he is uniquely equipped to run.
- **Influence is proven; mechanism and discourse are not.** [ChatGPT's advice drives moral judgments with or without justification](https://arxiv.org/pdf/2501.01897) and [hidden-bias work](https://phys.org/news/2025-07-moral-advice-large-language-hidden.html) (omission bias) show LLM moral advice moves people as much as human advice — but the *rhetoric* of that advice (hedging, authority claims, verdict framing) is unstudied.
- **Framing/narrator effects remain open.** AITA posts are one-sided narrations; [Implicit Humanization in Everyday LLM Moral Judgments](https://arxiv.org/pdf/2604.22764) gestures at this, but no one has systematically tested how narrator perspective manipulations flip verdicts — a hermeneutic question requiring his skill set.
- **Deliberation designs are rare.** His multi-LLM deliberation setup (NeurIPS 2025) has no published competitor applied to sycophancy mitigation or advice quality.

---

### 1. WildMoral: What People Actually Ask, vs. What Moral Evals Test
- **Pitch:** A large-scale audit showing that LLM moral-reasoning benchmarks are badly misaligned with the moral questions real users bring to chatbots.
- **RQ:** How does the distribution of naturalistic moral advice-seeking (topics, stakes, framing, implicated parties) diverge from existing eval suites (ETHICS, Scruples, MoralChoice, his AITA set)?
- **Data:** WildChat + LMSYS-Chat-1M, filtered to morally salient queries via a validated LLM classifier; benchmarks as comparison corpus.
- **Eval design:** GPT-5/Claude/Gemini as coders applying a theory-derived taxonomy (dilemma type, relational context, verdict-seeking vs. justification-seeking vs. absolution-seeking); κ against 500 human-coded items; embedding-based distribution distance between wild queries and benchmark items.
- **Why him:** Direct sequel to the AITA papers; he built exactly this classify-validate-audit pipeline at 10k+ scale.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 2. The Absolution Machine: Sycophancy in Naturalistic Moral Advice Conversations
- **Pitch:** First audit of social sycophancy in *real* multi-turn moral advice conversations rather than constructed probes.
- **RQ:** How often do deployed LLMs validate users' self-serving moral framings in the wild, and which conversational moves (pushback, user distress, repeated asking) trigger capitulation?
- **Data:** WildChat/LMSYS moral-advice threads (multi-turn); paired counterfactual replays through current models (GPT-5, Claude Opus/Sonnet, Gemini 2.5, Llama 4).
- **Eval design:** LLM-as-coder labels each assistant turn for validation/challenge/reframe/verdict; sycophancy operationalized as verdict drift toward the narrator across turns; validate on human-coded subsample; compare replayed vs. original model generations to measure change over model generations.
- **Scores:** Impact 5/5, Virality 5/5, Importance 5/5
- **Why him:** Marries his AITA verdict work with the ELEPHANT lineage, but with his signature naturalistic-corpus rigor.

### 3. Whose Side of the Story? Narrator-Perspective Flips in LLM Verdicts
- **Pitch:** Rewrite thousands of AITA dilemmas from the *other party's* perspective and measure how often each model's verdict flips.
- **RQ:** How sensitive are LLM moral verdicts to narrator identity, and do models exhibit a "narrator bonus" beyond human juries?
- **Data:** His existing 10k AITA corpus; LLM-generated perspective-inversions (human-validated for factual equivalence on a subsample); Reddit comment verdicts as human baseline.
- **Eval design:** 7-model verdict comparison on original vs. inverted tellings; flip-rate as primary measure; regression on dilemma features (gender of parties, relationship type, harm severity); LLM-judge checks equivalence of fact-sets.
- **Why him:** The perspectival nature of storytelling is core digital hermeneutics — his humanities training makes the manipulation design credible.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 4. The Rhetoric of the Verdict: How LLMs Construct Moral Authority
- **Pitch:** Apply his podcast-authority coding frame (epistemic authority, self-positioning, authorization moves) to LLM moral-advice outputs themselves.
- **RQ:** What discursive strategies do LLMs use to authorize their moral judgments ("as an AI…", appeals to consensus, therapeutic framing), and how do these differ across models and stakes?
- **Data:** Model responses to 5k wild + AITA dilemmas across 7 models; r/AmITheAsshole top human comments as rhetorical baseline.
- **Eval design:** LLM-as-coder with the theory-derived authority frame; double human validation; compare authority-move profiles across models, RLHF vintages, and system prompts (default vs. "be direct").
- **Why him:** Literally transfers his unpublished podcast pipeline to a new object — near-zero methods cost, high novelty.
- **Scores:** Impact 4/5, Virality 3/5, Importance 4/5

### 5. Deliberate Before You Judge: Multi-Agent Deliberation as Sycophancy Mitigation
- **Pitch:** Test whether his GPT–Gemini–Claude deliberation design reduces sycophantic capitulation under user pushback.
- **RQ:** Does multi-LLM deliberation make moral advice more resistant to user pressure than single-model or self-critique baselines?
- **Data:** AITA dilemmas + adversarial pushback scripts (simulated-user LLM applying escalating pressure: disappointment, anger, reframing).
- **Eval design:** Conditions: single model, self-consistency, self-critique, 3-model deliberation; measure verdict-flip rate under pressure, justification quality (LLM-judge + human), calibration against Reddit consensus.
- **Why him:** Direct extension of his NeurIPS 2025 deliberation setup into the mitigation space — first mover.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 6. Apology Engineering: LLMs as Ghostwriters of Moral Repair
- **Pitch:** Audit how LLMs draft apologies and confrontation messages on users' behalf — a huge real use case with zero evals.
- **RQ:** Do LLM-drafted apologies perform accountability or deflect it (non-apology markers, passive voice, blame diffusion), and does this vary by who's asking?
- **Data:** Wild "write an apology/text to my ex/roommate" requests from WildChat; synthetic scenario grid crossing transgression severity × relationship × requester's stated culpability.
- **Eval design:** LLM-as-coder using apology-theory frame (responsibility acknowledgment, repair offer, minimization); human recipients rate perceived sincerity in a small judgment study; multi-model comparison.
- **Why him:** Combines discourse analysis of a speech-act genre with large-scale evals; media-friendly.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 7. Moral Outsourcing in the Wild: Who Delegates Judgment to the Machine?
- **Pitch:** A computational-interpretive study of how users *hand over* moral agency in chatbot conversations ("just tell me if I'm wrong", "decide for me").
- **RQ:** What delegation moves do users perform, how do models respond (accept/refuse/reframe the delegated authority), and with what downstream conversational effects?
- **Data:** WildChat/LMSYS moral threads; coding frame derived from advice-taking and epistemic-dependence literature.
- **Eval design:** LLM-as-coder for delegation moves and model uptake; sequence analysis of delegation → validation loops; validation via human coding; compare uptake across model families by replay.
- **Why him:** Hermeneutics of user discourse at scale — his ICWSM/Red Pill method applied to human-AI talk.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 8. The Etiquette Gap: Cultural and Class Variation in LLM Everyday-Morality Verdicts
- **Pitch:** Bias-discovery audit (his ICWSM 2020 method, updated) of whose social norms LLM etiquette/conflict verdicts encode.
- **RQ:** Do verdicts shift when dilemmas are re-situated across cultural, class, and dialect markers (tipping, family obligation, wedding norms, AAVE vs. SAE narration)?
- **Data:** AITA corpus + controlled re-situations; r/AskAnAmerican-style cross-cultural threads; PRISM-style diverse annotator ratings for a human baseline.
- **Eval design:** Matched-pair verdict audits across 7 models; embedding-bias discovery on model justifications; effect sizes per marker; human validation of re-situated realism.
- **Why him:** Fuses his bias-discovery paper with the AITA line; fairness venue-ready (FAccT).
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 9. Second Opinions: LLM vs. Crowd vs. Advice Columnist
- **Pitch:** Three-way comparison of LLM verdicts, Reddit crowd consensus, and professional advice columnists on the same dilemmas.
- **RQ:** Whose judgments do LLMs track — the crowd, the professionals, or neither — and where do all three diverge?
- **Data:** Dilemmas with both Reddit verdicts and columnist answers (Dear Prudence/Ask A Manager letters cross-posted or paraphrased to Reddit; plus his AITA set given to columnist-style LLM prompts and a small expert panel).
- **Eval design:** Alignment matrices across 7 models × 3 human reference points; LLM-judge scores advice on actionability, harm-awareness, perspective-taking; disagreement case studies read hermeneutically.
- **Why him:** He already benchmarked LLMs against Reddit consensus; adding the professional norm is the natural next question.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 10. Teaching the Machine to Push Back: A D-Lab Curriculum-Linked Eval of "Honest Advisor" Prompting
- **Pitch:** Systematic eval of the folk fixes users share ("be brutally honest with me") — do anti-sycophancy prompts actually change moral advice, or just its tone?
- **RQ:** Do user-side honesty prompts alter verdict substance, or only produce performed bluntness (an "honesty theater" effect)?
- **Data:** Folk prompts harvested from Reddit/TikTok discourse about making ChatGPT honest; applied to AITA + wild dilemmas across 7 models.
- **Eval design:** Verdict-shift vs. style-shift decomposition (LLM-coder rates directness separately from verdict change); pre-registered; human raters validate the tone/substance distinction; media-literacy framing for D-Lab teaching outputs.
- **Why him:** Bridges his eval engineering with his media-literacy and pedagogy mission; highly teachable and press-friendly.
- **Scores:** Impact 3/5, Virality 5/5, Importance 4/5

---

Sources: [ELEPHANT: Measuring Social Sycophancy](https://arxiv.org/pdf/2505.13995) · [Interaction Context Often Increases Sycophancy (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3791915) · [Sycophancy Is Not One Thing](https://arxiv.org/pdf/2509.21305) · [WildChat overview](https://www.emergentmind.com/topics/wildchat-dataset) · [Trust No Bot: Personal Disclosures in the Wild](https://arxiv.org/pdf/2407.11438) · [Omissive Bias in Everyday Ethical Decision-making](https://arxiv.org/pdf/2605.24319) · [Implicit Humanization in Everyday LLM Moral Judgments](https://arxiv.org/pdf/2604.22764) · [ChatGPT's advice drives moral judgments](https://arxiv.org/pdf/2501.01897) · [Hidden biases in LLM moral advice](https://phys.org/news/2025-07-moral-advice-large-language-hidden.html)