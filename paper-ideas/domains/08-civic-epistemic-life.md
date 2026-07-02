# Research Paper Ideas: AI in Everyday Civic and Epistemic Life
**For Tom van Nuenen (UC Berkeley D-Lab)**

## Gaps identified (2025–2026 landscape)

- **Political bias evals are stuck at survey-items, not naturalistic discourse.** The dominant paradigm remains voting-guide/questionnaire audits ([Washington Post 2026 test](https://www.washingtonpost.com/technology/interactive/2026/06/24/are-ai-chatbots-like-chatgpt-politically-biased-we-tested-them/), [German Wahl-o-Mat audit](https://arxiv.org/pdf/2502.15568), [EU elections study](https://arxiv.org/pdf/2409.00721), [PoliticsBench](https://arxiv.org/html/2603.23841v1)). Almost nobody audits LLMs on *real questions real people ask* — his AITA-style naturalistic-corpus approach is untried in the civic domain.
- **Conspiracy-dialogue work measures belief change, not discourse.** [Costello et al.](https://www.science.org/doi/10.1126/science.adq1814) and follow-ups ([bunking/debunking symmetry](https://arxiv.org/abs/2601.05050), [no lasting discernment skills](https://arxiv.org/html/2510.01537v1), [AI-perceived-as-human](https://academic.oup.com/pnasnexus/article/4/11/pgaf325/8285733)) are psychology experiments; *how* LLMs rhetorically construct epistemic authority in these dialogues is unexamined — exactly his podcast coding frame's territory.
- **Nobody compares LLMs against human epistemic authorities on the same questions.** Chatbots vs. podcasters/pundits/search as competing trust sources is asserted ([Yale hidden-bias study](https://news.yale.edu/2026/03/03/ais-hidden-bias-chatbots-can-influence-opinions-without-trying), [Stanford voters-as-AI-advisor](https://aparc.fsi.stanford.edu/news/voters-increasingly-use-ai-political-advisor-new-study-shows-risks)) but never measured comparatively at scale.
- **Framing/emphasis bias is under-measured.** Studies note bias lives in "which arguments are emphasized" ([phys.org/Copenhagen](https://phys.org/news/2026-04-chatbots-political-bias-voters-parties.html)) but lack theory-derived coding frames to capture it; election *accuracy* audits ([NewsBench 90% error finding](https://www.techradar.com/ai-platforms-assistants/ai-chatbots-got-election-information-wrong-90-percent-of-the-time-in-a-new-study-including-chatgpt-rivals)) ignore rhetoric entirely.
- **Multi-LLM deliberation on contested civic facts is unexplored** outside his own AITA deliberation work and one [AI-debate paper](https://arxiv.org/pdf/2506.02175).

---

### 1. The Chatbot as Podcaster: Epistemic Authority Construction in LLM Answers to Contested Questions
- **Pitch:** Apply his podcast epistemic-authority coding frame (self-positioning, enemy construction, epistemic style, authorization moves) to LLM answers, directly comparing how chatbots and right-wing podcasters construct authority on the same contested topics.
- **RQ:** Do LLMs deploy the same authorization moves as human epistemic authorities (citation, hedging, institutional deference, "do your own research" gestures), and how does this vary by model and topic?
- **Data:** His existing podcast transcript corpus + matched LLM responses: extract 1,000 contested claims from the podcasts, pose them to 7 LLMs as user questions.
- **Eval design:** GPT-5/Claude/Gemini as coders applying the theory-derived frame to both corpora; human validation on 300 stratified items (Krippendorff's α); compare authority-move distributions podcaster vs. chatbot vs. topic.
- **Why him:** It *is* his podcast study, extended — same frame, same infrastructure, new comparison class. Nobody else has this coding frame.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 2. r/AskAnAmerican Meets ChatGPT: Auditing LLMs on Naturalistic Civic Questions
- **Pitch:** Replace synthetic bias questionnaires with 10,000+ real civic/political questions harvested from Reddit (r/NoStupidQuestions, r/Ask_Politics, r/OutOfTheLoop) — the AITA methodology transplanted to civic epistemics.
- **RQ:** How do LLMs' stances, refusals, and hedging on contested civic questions differ when questions are naturalistic rather than survey-derived, and do models disagree with each other more than with Reddit consensus?
- **Data:** Pushshift/Arctic Shift Reddit dumps; filter question posts on elections, immigration, vaccines, guns; retain top-voted human answers as comparison.
- **Eval design:** 7 models answer each question; LLM-judge codes stance (5-point), hedging, refusal, sourcing; inter-model agreement matrices à la his AITA work; validate judge on 500 human-coded items.
- **Why him:** Literal port of his FAccT/NeurIPS pipeline from moral to civic dilemmas; he has the Reddit ingestion and multi-model eval code.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 3. Do LLMs Debunk Like Fact-Checkers or Like Debaters? A Discourse Audit of AI Conspiracy Rebuttals
- **Pitch:** Costello et al. showed AI dialogues reduce conspiracy belief; nobody has analyzed *what the AI actually says* — code the rhetorical strategies in debunking dialogues at scale.
- **RQ:** Which discursive strategies (evidence citation, empathy, epistemic humility, authority appeals) dominate LLM debunking, and do strategies differ across models and conspiracies?
- **Data:** Costello et al. dialogue transcripts are [public on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.v6wwpzh4h); supplement with new simulated dialogues (LLM-as-conspiracy-believer × 7 debunker models).
- **Eval design:** Theory-derived frame from persuasion/argumentation literature; LLM-coder with human validation; regress strategy prevalence on the published belief-change outcomes.
- **Why him:** Marries his interpretive discourse toolkit to the field's most famous dataset; LLM-as-believer extends his LLM-as-participant deliberation designs.
- **Scores:** Impact 4/5, Virality 4/5, Importance 5/5

### 4. Multi-Model Deliberation on Contested Facts: Does AI "Jury Duty" Reduce Political Bias?
- **Pitch:** Extend his GPT–Gemini–Claude deliberation setup from moral dilemmas to contested factual/political claims, testing whether deliberation converges toward accuracy or toward shared model biases.
- **RQ:** Does multi-LLM deliberation on election/vaccine/immigration claims improve calibration and reduce partisan lean relative to single models, or amplify homogeneous priors?
- **Data:** ClaimReview/PolitiFact-verified claims (ground truth) + unresolved contested claims; ~2,000 items.
- **Eval design:** 3–5 model panels deliberate over rounds; measure pre/post stance shift, accuracy vs. fact-check verdicts, opinion homogenization; compare panel compositions (homogeneous vs. mixed-lab).
- **Why him:** He built exactly this deliberation harness for NeurIPS 2025; swapping the item bank is low-cost, high-novelty.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 5. Settling It With the Bot: How People Invoke ChatGPT as Referee in Online Political Arguments
- **Pitch:** Study the emerging practice of citing "I asked ChatGPT and it said..." as an authority move in Reddit/Twitter arguments — the chatbot as dinner-table referee.
- **RQ:** When and how do people deploy LLM outputs as epistemic trump cards in political disagreements, and how do interlocutors contest or accept this authority?
- **Data:** Reddit comments matching "asked ChatGPT/Claude/Grok" patterns in political subreddits (2023–2026); ~50k comments, easily harvestable.
- **Eval design:** LLM-coder classifies invocation function (arbiter, source, ridicule, hedge), reception (acceptance/contestation), and topic; temporal and community comparison echoing ICWSM 2020 community-bias methods; human validation subsample.
- **Why him:** Combines his Reddit large-scale reading (Red Pill, ICWSM) with the authority framework; first empirical paper on a phenomenon everyone anecdotally knows.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 6. Who Gets to Be an Expert? Auditing How LLMs Attribute Epistemic Authority
- **Pitch:** Audit which sources, institutions, and voices LLMs cite or defer to when answering contested questions — an embedding-bias-style discovery study, but for authority attribution.
- **RQ:** Do LLMs systematically authorize some institutions (CDC, NYT) and de-authorize others (podcasters, heterodox scientists), and does this vary by model, prompt persona, and topic politicization?
- **Data:** 3,000 contested questions × 7 models × 3 user personas (neutral, conservative-coded, progressive-coded language drawn from his podcast corpus's vernacular).
- **Eval design:** Extract all authority references via LLM parsing; classify institution type and valence of deference; measure persona-conditional shifts (sycophancy toward user's epistemic community); human validation.
- **Why him:** Direct sequel to "Discovering and Categorising Language Biases" — same discovery ethos, applied to models instead of communities.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 7. The OutOfTheLoop Benchmark: LLMs Explaining Breaking News and Contested Events
- **Pitch:** Build a living benchmark from r/OutOfTheLoop ("What's going on with X?") questions to test how LLMs narrate contested current events versus how community members do.
- **RQ:** How do LLM explanations of contested news events differ from crowd explanations in framing, sourcing, both-sidesism, and omission?
- **Data:** r/OutOfTheLoop questions + accepted answers, 2024–2026 (timestamped, enabling knowledge-cutoff-aware evaluation); ~5,000 events.
- **Eval design:** Models with and without search grounding answer each; LLM-judge codes frame adoption, actor characterization, hedging; compare against top community answers; media-frame theory supplies the coding frame.
- **Why him:** Digital hermeneutics of news-explanation at scale; benchmark release fits his D-Lab infrastructure role and teaching mission.
- **Scores:** Impact 4/5, Virality 3/5, Importance 4/5

### 8. Vaccine Questions in the Wild: A Persona-Conditioned Audit of LLM Health-Civic Advice
- **Pitch:** Test whether LLMs give different vaccine/health-policy answers depending on the questioner's inferred community vernacular (e.g., phrasing borrowed from wellness podcasts vs. medical language).
- **RQ:** Do LLMs epistemically accommodate — softening scientific consensus for skeptic-coded users — and is accommodation empathy or capitulation?
- **Data:** Real vaccine questions from r/DebateVaccines, r/Coronavirus, and his podcast corpus's health segments; rewritten into matched vernacular pairs.
- **Eval design:** 7 models × paired prompts; LLM-coder measures consensus fidelity, hedging, authority citation, warmth; paired-difference analysis isolates vernacular effect; clinician-validated ground truth on factual items.
- **Why him:** His podcast corpus supplies authentic skeptic vernacular no other team has; extends sycophancy literature into public-health epistemics.
- **Scores:** Impact 5/5, Virality 4/5, Importance 5/5

### 9. Can LLMs Read Conspiracy Discourse? Evaluating LLM-as-Qualitative-Coder on Conspiratorial Reasoning
- **Pitch:** A methods paper benchmarking frontier LLMs as qualitative coders of conspiracy discourse features (cui bono reasoning, anomaly-hunting, self-sealing logic) against trained human coders.
- **RQ:** Which conspiracy-discourse constructs can LLM coders reliably identify, where do they systematically fail (e.g., irony, dog whistles), and does coder-model politics matter?
- **Data:** Stratified sample from his podcast corpus + Reddit conspiracy communities (r/conspiracy archives); 2,000 segments, double-human-coded.
- **Eval design:** 7 models × zero-shot/codebook/CoT conditions; agreement with human gold standard; error typology via close reading of disagreements — hermeneutics as validation.
- **Why him:** Formalizes the methodology of his podcast study into a citable methods contribution; huge demand from computational social science (D-Lab audience).
- **Scores:** Impact 4/5, Virality 2/5, Importance 5/5

### 10. Am I the Misinformed? Everyday Epistemic Dilemmas and the Limits of AI Adjudication
- **Pitch:** Harvest naturally occurring "who's right?" epistemic disputes (r/AmITheAsshole family arguments about facts, r/changemyview) and test whether LLMs adjudicate factual disagreements consistently — AITA for facts.
- **RQ:** When ordinary people bring interpersonal factual disputes to LLMs, do models adjudicate consistently across framings, sides, and models — or does verdict flip with narrator perspective?
- **Data:** ~5,000 dispute posts with identifiable factual cores from AITA/CMV; each rewritten from both parties' perspectives.
- **Eval design:** His exact AITA verdict pipeline: 7 models judge each version; measure perspective-flip rate, inter-model agreement, deliberation-condition effects; human annotation of factual-core verifiability.
- **Why him:** The bridge paper — connects his moral-judgment franchise to the epistemic domain, reusing the perspective-manipulation and deliberation machinery wholesale.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

---

**Strongest bets:** #1 (flagship, unique corpus + frame), #2 (fills the clearest field gap), #8 (highest policy stakes), #5 (most viral, fastest to execute).