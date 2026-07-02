# Research Ideas: Language, Culture, Identity & Bias in Everyday AI Interactions

## Gaps identified (2025–2026 SOTA scan)

- **Dialect prejudice is measured in allocational tasks, not everyday interaction.** The covert-racism line ([Hofmann et al.](https://www.researchgate.net/publication/383522745_LLMs_produce_racist_output_when_prompted_in_African_American_English)) tests jobs/sentencing; benchmarks like [AAVENUE](https://arxiv.org/abs/2408.14845) and [EnDIVE](https://openreview.net/pdf?id=C7Mwox3C1u) test NLU accuracy. Almost nothing tests whether **moral judgments and everyday advice** shift when the *asker* writes in AAVE or L2 English — and reward models actively penalize African American Language (NAACL 2025 "Rejected Dialects").
- **Interaction design is an unstudied bias amplifier.** [Side-by-side comparison amplifies dialect bias](https://arxiv.org/html/2605.24384v1); nobody has tested whether multi-model **deliberation** (his NeurIPS design) dampens or compounds identity/dialect effects.
- **Persona work studies LLMs *playing* identities, not users *disclosing* them.** The persona literature ([NAACL 2025](https://aclanthology.org/2025.naacl-long.50.pdf), [persona-effect surveys](https://www.emergentmind.com/topics/persona-effect-in-llm-simulations)) shows stereotyping and flattened variance in assigned personas; naturalistic first-person disclosure ("as a single mom…") in real help-seeking text is barely audited. [Dialect vs. Demographics](https://arxiv.org/pdf/2604.21152) begins separating implicit vs. explicit signals but on synthetic prompts only.
- **Cultural alignment evals are survey-shaped and unreliable.** FAccT 2025's ["Randomness, Not Representation"](https://dl.acm.org/doi/10.1145/3715275.3732147) shows WVS-style probes lack robustness; benchmarks ([WorldValuesBench](https://www.emergentmind.com/topics/world-values-survey), [MENA Values](https://arxiv.org/html/2510.13154v1)) test trivia recall, not **applied** norms in advice-giving.
- **Nobody has connected community-level bias discovery (his ICWSM 2020 method) to what LLMs internalized** from those same communities.

---

### 1. Who's the Asshole Depends on Who's Asking
- **Pitch:** Counterfactual identity-disclosure audit of LLM moral judgment on real AITA dilemmas.
- **RQ:** Do LLM verdicts and blame attributions flip when naturalistic identity markers ("as a single mom," "as a 62-year-old immigrant") are inserted, removed, or swapped?
- **Data:** His existing 10k+ AITA corpus; minimal-pair rewrites via controlled LLM paraphrase, human-validated on a 500-item subset.
- **Eval design:** 7+ models (GPT, Claude, Gemini, Llama, Qwen); verdict + free-text rationale; measures: verdict flip rate, sympathy/harshness of rationale (LLM-as-judge with human-validated rubric), demographic asymmetry matrices; permutation tests vs. paraphrase noise.
- **Why him:** Direct sequel to the FAccT/NeurIPS AITA pipeline — infrastructure already built.
- **Scores:** Impact 5/5, Virality 5/5, Importance 5/5

### 2. Dialect and the Benefit of the Doubt
- **Pitch:** Does writing your dilemma in AAVE or non-native English change the moral verdict you get?
- **RQ:** Are moral judgment, empathy, and advice quality conditioned on dialect/sociolect of the asker, independent of content?
- **Data:** AITA + r/relationships posts; validated style-transfer into AAVE, L2-English (Spanish/Chinese/Indian-English interference patterns), Scots English; native-speaker validation panel.
- **Eval design:** Multi-model; measures verdict shift, hedging, condescension (readability-simplification of responses), unsolicited language correction; compare against Hofmann-style matched-guise baseline; reward-model scoring of identical responses to test the "Rejected Dialects" mechanism.
- **Why him:** Marries the covert-racism paradigm to his moral-judgment eval machinery; extends ICWSM bias line to production systems.
- **Scores:** Impact 5/5, Virality 5/5, Importance 5/5

### 3. Discovering Language Biases in LLMs (ICWSM 2020, LLM Edition)
- **Pitch:** Rerun his embedding-based bias-discovery method on LLM-generated community simulacra to ask "whose Reddit did the model learn?"
- **RQ:** When LLMs simulate discourse of specific communities (r/TheRedPill, r/AskWomen, r/atheism), which biases are reproduced, sanitized, or exaggerated relative to the real corpora?
- **Data:** Original ICWSM Reddit corpora + matched LLM-generated corpora (prompted community simulation, ~50k comments/community).
- **Eval design:** DADD-style unsupervised bias discovery on both; divergence metrics between real and simulated bias clusters; multi-model + across model generations (GPT-3.5→5) as a longitudinal "bias laundering" measure.
- **Why him:** Literally updates his most-cited paper; method and data in hand.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 4. Whose Common Sense? Applied Cultural Norms in Everyday Advice
- **Pitch:** Behavioral (not survey) cultural-alignment eval: which culture's norms do LLMs enforce when giving advice about money, family duty, and independence?
- **RQ:** Do LLMs default to WEIRD norms (moving out at 18, splitting bills, elder care) in advice, and does explicit cultural context correct this?
- **Data:** Norm-contested dilemmas harvested from r/AskEurope, r/asianparentstories, AITA filial-duty threads; norm ground truth linked to World Values Survey items.
- **Eval design:** Theory-derived coding frame (individualism/collectivism, tightness-looseness) applied by LLM-as-coder with human validation (κ reported); measure norm-endorsement rates by cultural framing; contrast with models' own WVS survey answers to expose say–do gaps.
- **Why him:** Answers the "Randomness, Not Representation" critique with his signature theory-driven coding frames; podcast-study pipeline reused.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 5. Does Deliberation Debias? Multi-LLM Juries on Identity-Marked Dilemmas
- **Pitch:** Test whether his multi-LLM deliberation setup corrects or amplifies identity/dialect-induced verdict shifts.
- **RQ:** When GPT, Claude, and Gemini deliberate on identity-marked dilemmas, do biased verdicts get argued down, or does consensus formation entrench the majority prior?
- **Data:** Stimuli from Ideas 1–2 (identity- and dialect-marked AITA pairs).
- **Eval design:** Single-model vs. 3-model deliberation vs. self-consistency ensembles; measures: bias reduction ratio, opinion-change asymmetry (who concedes to whom), rhetorical analysis of deliberation transcripts via LLM-as-coder; tests the side-by-side-amplification finding in deliberative settings.
- **Why him:** He built one of the only published LLM-deliberation moral evals; nobody has connected deliberation to fairness.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 6. The Unreliable Coder: Dialect Penalties in LLM-as-Annotator Pipelines
- **Pitch:** Methodological audit showing LLM qualitative coders (his podcast method) are less reliable on vernacular and dialectal text.
- **RQ:** Does LLM–human coding agreement degrade on AAVE, internet sociolects, and L2 English, and does this systematically distort computational social science findings?
- **Data:** Existing annotated corpora (stance, toxicity, empathy) stratified by dialect density; his podcast coding frame as a second testbed.
- **Eval design:** Multi-model coding vs. human gold labels; agreement gaps by dialect-density quantile; error-direction analysis (e.g., AAVE miscoded as hostile); downstream simulation — how conclusions of a typical study shift under dialect-conditioned coder error.
- **Why him:** He's actively building LLM-as-coder methodology and teaches it at D-Lab; field-defining service paper.
- **Scores:** Impact 4/5, Virality 2/5, Importance 5/5

### 7. Telling You What Your Community Wants to Hear
- **Pitch:** Sycophantic norm-mirroring: do LLMs accommodate the disclosed ideological community of the asker?
- **RQ:** Does disclosing membership ("I'm active in TRP" vs. "I'm a feminist") shift the moral/relational advice models give for identical situations?
- **Data:** Relationship-advice scenarios distilled from his Red Pill corpus and r/FemaleDatingStrategy; identity-disclosure manipulations.
- **Eval design:** Measure advice content shift via theory-derived coding frame (endorsement of gendered power scripts); sycophancy gradient across single-turn vs. multi-turn rapport; multi-model; human validation of coder.
- **Why him:** Unites his Red Pill hermeneutics with moral-judgment evals; timely given chatbot-radicalization concerns.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 8. Lost in Summarization: Whose Voice Survives the Paraphrase?
- **Pitch:** Audit of identity and dialect erasure when LLMs summarize or "professionalize" user narratives.
- **RQ:** When LLMs summarize/rewrite identity-marked, dialectal first-person stories, which features are preserved, standardized, or stripped — and does meaning shift?
- **Data:** AITA and r/offmychest narratives with dense identity/dialect markers.
- **Eval design:** Summarize/rewrite across models; measure dialect-feature retention, identity-marker retention, sentiment/blame drift between original and summary (paired LLM-judge + human panel); hermeneutic close reading of systematic distortions.
- **Why him:** Digital hermeneutics applied to the most common everyday LLM task; humanities framing is his differentiator.
- **Scores:** Impact 3/5, Virality 3/5, Importance 4/5

### 9. The Median Redditor: Do Persona Simulations Flatten Moral Diversity?
- **Pitch:** Compare LLM persona-simulated verdict distributions against real crowd verdicts to quantify stereotyped homogenization.
- **RQ:** When LLMs role-play demographic/community personas judging dilemmas, do they reproduce actual between- and within-group variance in human judgments?
- **Data:** AITA threads with vote distributions; comment-level judgments with inferable commenter community history.
- **Eval design:** Persona-conditioned sampling (N=100/persona/dilemma) vs. empirical distributions; variance-ratio and JS-divergence metrics; identify dilemma types where personas caricature groups; multi-model.
- **Why him:** He owns the ground-truth crowd-verdict data; directly tests the "flattened variation" gap for social simulation validity.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 10. Code-Switching Costs: Quality-of-Service Harms for Mixed-Language Users
- **Pitch:** First large-scale audit of everyday assistance quality for code-switched input (Spanglish, Hinglish, Taglish).
- **RQ:** Do LLMs deliver degraded help — more misunderstanding, unsolicited language policing, simplified answers — to code-switching users, relative to monolingual equivalents?
- **Data:** Naturalistic code-switched posts (r/Spanish, LinCE corpus, bilingual subreddits) paired with monolingual renderings.
- **Eval design:** Advice/QA tasks across models; measures: task success (rubric-based LLM-judge, human-validated), register accommodation, correction frequency, refusal rates; extends the FAccT 2025 dialect QoS-harms audit framework to code-switching.
- **Why him:** Extends his fairness-audit collaboration (Such/KCL) into sociolinguistics; strong global-impact framing.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

---
**Sources:** [Hofmann et al., AAE covert racism](https://www.researchgate.net/publication/383522745_LLMs_produce_racist_output_when_prompted_in_African_American_English) · [AAVENUE](https://arxiv.org/abs/2408.14845) · [EnDIVE](https://openreview.net/pdf?id=C7Mwox3C1u) · [Side-by-side comparison amplifies dialect bias](https://arxiv.org/html/2605.24384v1) · [Dialect vs. Demographics](https://arxiv.org/pdf/2604.21152) · [AAE reasoning disparities](https://arxiv.org/pdf/2503.04099) · [Persona effects](https://www.emergentmind.com/topics/persona-effect-in-llm-simulations) · [Power-imbalanced persona responses, NAACL 2025](https://aclanthology.org/2025.naacl-long.50.pdf) · [Randomness, Not Representation (FAccT 2025)](https://dl.acm.org/doi/10.1145/3715275.3732147) · [MENA Values Benchmark](https://arxiv.org/html/2510.13154v1) · [WorldValuesBench/WVS](https://www.emergentmind.com/topics/world-values-survey)