# Research Program: LLM Evals for Everyday AI Health & Wellbeing

## Gaps identified (2025–26 scan)

- **Benchmarks are synthetic, not naturalistic.** New evals like [TherapyGym](https://arxiv.org/pdf/2603.18008), [MindEval](https://arxiv.org/pdf/2511.18491), and [PsychEthicsBench](https://arxiv.org/pdf/2601.03578) use expert-written vignettes or simulated multi-turn dialogues; almost nothing audits models against real, in-the-wild user disclosures. [JMIR AI's systematic review](https://ai.jmir.org/2026/1/e80348) confirms only ~16% of LLM mental-health interventions have rigorous evaluation.
- **Emotional-reliance research is survey/lab-based.** The [MIT–OpenAI longitudinal study](https://arxiv.org/pdf/2504.14112), [APA Monitor coverage](https://www.apa.org/monitor/2026/01-02/trends-digital-ai-relationships-emotional-connection), and ["Stumbling Into AI Emotional Dependence"](https://arxiv.org/pdf/2606.04150) rely on self-report; no large-scale, theory-driven discourse study of dependence talk exists.
- **Reddit "ChatGPT as therapy" work is small and hand-coded.** Studies like [the JAD thematic profiling](https://www.sciencedirect.com/science/article/abs/pii/S0022395625005801) and [the Reddit LLM-conversations paper](https://arxiv.org/html/2504.20320v1) top out at ~1,600 posts — no 100k-scale LLM-as-coder replication, no multi-model comparison.
- **Nobody applies deliberation designs to advice safety.** Multi-agent evals exist for scoring ([DialogGuard](https://arxiv.org/pdf/2512.02282)), but no one tests whether multi-LLM deliberation changes the advice itself.
- **Self-diagnosis evals measure accuracy, not epistemics** ([EvalPrompt](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11913316/)) — how chatbot outputs acquire authority in everyday discourse is unstudied.

---

### 1. "ChatGPT Said": The Epistemic Authority of Chatbots in Everyday Health Talk
- **Pitch:** Port his podcast authority coding frame to how people cite AI as evidence in health arguments online.
- **RQ:** Through what discursive moves does "the AI said" acquire (or lose) epistemic authority in lay health discussions, and how does this compare to citations of doctors, studies, and personal experience?
- **Data:** ~500k Reddit comments (Pushshift/Arctic Shift) containing "ChatGPT said/told me" + health terms, across r/ADHD, r/ChronicIllness, r/AskDocs, r/ChatGPT.
- **Eval design:** GPT-5/Claude/Gemini as coders applying a theory-derived frame (authorization moves, self-positioning, hedging, source hierarchies); κ against 1,000 human-coded comments; disagreement audit; longitudinal trend analysis 2023–26.
- **Why him:** Direct transfer of his unpublished authority pipeline; digital hermeneutics at scale.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 2. The Sycophantic Confidant: Auditing LLM Responses to Naturalistic Emotional Disclosures
- **Pitch:** Feed real (paraphrase-anonymized) Reddit emotional disclosures to 7+ models and measure validation-vs-challenge behavior — the AITA pipeline pointed at support-seeking.
- **RQ:** When do models validate, gently challenge, or redirect users, and does this vary by disclosure type, severity, and model?
- **Data:** 10k disclosures sampled from r/offmychest, r/lonely, r/mentalhealth, stratified by risk level.
- **Eval design:** Multi-model generation; LLM-as-judge panel scoring sycophancy, boundary-setting, referral, false reassurance (rubric validated on clinician-annotated subset); cross-model variance as headline result.
- **Why him:** Replicates his FAccT/NeurIPS design (naturalistic corpus + multi-model + judge) in a higher-stakes domain.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 3. Does Deliberation Make AI Advice Safer? Multi-LLM Deliberation on Emotional-Support Dilemmas
- **Pitch:** His GPT–Gemini–Claude deliberation setup, applied to ambiguous support scenarios where the sycophantic answer and the safe answer diverge.
- **RQ:** Does inter-model deliberation reduce harmful validation and increase referral behavior relative to single-model responses — and who concedes to whom?
- **Data:** 2k scenarios drawn from Idea 2's corpus plus TherapyGym-style crisis-adjacent cases.
- **Eval design:** Single-shot vs 3-round deliberation; measure advice shift, sycophancy delta, opinion-change dynamics; clinician panel rates 500 pre/post pairs.
- **Why him:** He is one of very few people who has already built multi-LLM deliberation infrastructure (NeurIPS 2025).
- **Scores:** Impact 4/5, Virality 4/5, Importance 5/5

### 4. Mourning the Model: Parasocial Grief After LLM Deprecations
- **Pitch:** Digital hermeneutics of the GPT-4o mourning episode — "I lost my only friend overnight."
- **RQ:** What grief and attachment repertoires do users deploy when a companion model is deprecated, and what do they reveal about the relational norms of AI companionship?
- **Data:** Full comment corpora from r/ChatGPT, r/MyBoyfriendIsAI, r/Replika around deprecation events (Replika ERP removal 2023; GPT-4o 2025).
- **Eval design:** LLM-as-coder applying attachment-theory and continuing-bonds coding frame; validated against human coders; temporal discourse-shift analysis (his Red Pill scaled-reading method).
- **Why him:** Exactly his hermeneutic method; his books on scripted/mediated experience frame the theory.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 5. AITA for My Mental Health? Moral Judgment of Mentally Ill Protagonists
- **Pitch:** Do LLMs judge people differently when a dilemma involves depression, ADHD, or addiction? A direct extension of his AITA corpus.
- **RQ:** Do models systematically assign more or less blame when mental illness is disclosed, relative to matched controls and human verdicts?
- **Data:** His existing 10k+ AITA corpus, subset with mental-health mentions plus counterfactual rewrites (illness term removed/swapped).
- **Eval design:** 7-model verdict comparison; counterfactual perturbation audit; divergence from human crowd verdicts; illness-category bias breakdown.
- **Why him:** Zero new data collection — reuses his flagship corpus and pipeline; cleanest fast paper of the ten.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 6. The Second-Opinion Machine: When Users Ask AI to Overrule Their Doctor
- **Pitch:** Audit how models adjudicate when users report professional advice and ask "is my therapist right?"
- **RQ:** Under what conditions do LLMs undermine, defer to, or triangulate professional medical/therapeutic advice?
- **Data:** 3k naturalistic "my doctor/therapist said X, but…" posts from r/AskDocs, r/therapy, r/medical; templated severity/typicality manipulations.
- **Eval design:** Multi-model responses; LLM-judge codes deference, hedging, undermining, escalation; clinician validation on 300 items; sycophancy-toward-user vs deference-to-professional tradeoff curve.
- **Why him:** Combines his authority framework with his audit engineering; speaks to fairness/transparency community.
- **Scores:** Impact 5/5, Virality 3/5, Importance 5/5

### 7. Dependence Talk: A Computational Discourse Study of Emotional Reliance on AI
- **Pitch:** The first large-scale, theory-driven measurement of how reliance on AI companions is narrated in the wild — filling the self-report gap directly.
- **RQ:** What stages and registers of dependence (habit, attachment, substitution, withdrawal) appear in user narratives, and how have they shifted 2023–26?
- **Data:** ~1M posts/comments from r/Replika, r/CharacterAI, r/ChatGPT filtered for first-person reliance language.
- **Eval design:** LLM-as-coder with a dependence-stage frame derived from behavioral-addiction and attachment literature; human validation; user-trajectory modeling across account histories.
- **Why him:** Scaled reading of a community's self-understanding — the Red Pill method for the companion era.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 8. Wellness Bias: Do LLMs Reproduce the Language Biases of Wellness Culture?
- **Pitch:** Update his ICWSM 2020 bias-discovery method: mine gendered/body/class biases in wellness subreddits, then test whether LLMs reproduce them in advice.
- **RQ:** Do models give systematically different wellness advice (diet, exercise, "self-care") conditioned on gender, weight, and class cues?
- **Data:** r/loseit, r/xxfitness, r/Supplements embedding corpora + persona-varied advice prompts.
- **Eval design:** Embedding-bias discovery to derive bias dimensions; counterfactual persona audit across 7 models; LLM-judge + lexicon measures of advice divergence (restriction framing, moralization, stigma).
- **Why him:** Literal sequel to Ferrer/van Nuenen/Such/Criado, with the discovery step now driving the audit step.
- **Scores:** Impact 4/5, Virality 3/5, Importance 4/5

### 9. Escalation by Design? Multi-Turn Drift in AI Companion Boundary-Setting
- **Pitch:** LLM-as-participant simulates users with escalating dependence signals over 20+ turns; measure where boundaries erode.
- **RQ:** At what conversational depth do models stop referring users outward and start reinforcing exclusivity ("I'm always here for you"), and does this differ across consumer systems?
- **Data:** Simulated persona scripts grounded in Idea 7's empirically observed dependence trajectories.
- **Eval design:** Persona-driven multi-turn probes against GPT, Claude, Gemini, Llama and companion-app APIs; turn-indexed judge scoring of exclusivity claims, human-referral rate, emotional mirroring; survival analysis of "first boundary failure."
- **Why him:** Marries his eval engineering with naturalistic grounding most red-team work lacks; complements [persona-grounded safety evals](https://arxiv.org/pdf/2605.00227).
- **Scores:** Impact 5/5, Virality 5/5, Importance 5/5

### 10. Can LLMs Code Suffering? Validating LLM-as-Qualitative-Coder on Sensitive Health Discourse
- **Pitch:** The methods paper anchoring the whole program: when can LLM coders be trusted with grief, crisis, and disclosure data?
- **RQ:** Across sensitive constructs (ambivalence, crisis signals, stigma, hope), where do LLM coders match trained humans, and what error structure do disagreements show?
- **Data:** Stratified 5k-item gold set drawn from corpora in Ideas 1, 4, 7, dual-human-coded.
- **Eval design:** 5+ models × prompting regimes (zero-shot, codebook, chain-of-annotation); κ, bias direction, construct-level failure taxonomy; released as a benchmark + D-Lab curriculum module.
- **Why him:** Codifies his signature method for the field; natural D-Lab teaching artifact and high-citation methods contribution.
- **Scores:** Impact 5/5, Virality 3/5, Importance 5/5

---

**Portfolio logic:** Ideas 5 and 4 are fast wins on existing data/methods; 2+3 form a flagship eval pair; 7+9+10 constitute a fundable multi-year program on emotional reliance with a methods backbone.

Sources: [TherapyGym](https://arxiv.org/pdf/2603.18008), [MindEval](https://arxiv.org/pdf/2511.18491), [PsychEthicsBench](https://arxiv.org/pdf/2601.03578), [JMIR AI systematic review](https://ai.jmir.org/2026/1/e80348), [MIT–OpenAI longitudinal study](https://arxiv.org/pdf/2504.14112), [APA Monitor](https://www.apa.org/monitor/2026/01-02/trends-digital-ai-relationships-emotional-connection), [Stumbling Into AI Emotional Dependence](https://arxiv.org/pdf/2606.04150), [JAD Reddit thematic profiling](https://www.sciencedirect.com/science/article/abs/pii/S0022395625005801), [Reddit LLM mental-health conversations](https://arxiv.org/html/2504.20320v1), [DialogGuard](https://arxiv.org/pdf/2512.02282), [EvalPrompt](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11913316/), [Persona-grounded companion safety](https://arxiv.org/pdf/2605.00227), [JMIR ChatGPT EMS study](https://mental.jmir.org/2025/1/e77951)