# Gaps identified

- **Folk theories research hasn't scaled to naturalistic LLM discourse.** 2024-25 work on folk psychology of LLMs is survey/vignette-based ([Neuroscience of Consciousness 2024](https://academic.oup.com/nc/article/2024/1/niae013/7644104); [companion-framing experiments, 2025](https://arxiv.org/html/2510.18039v2)); algorithmic folk-theory work remains focused on feeds/TikTok ([CSCW](https://dl.acm.org/doi/10.1145/3476046)). Nobody has done large-scale, theory-driven coding of *in-the-wild* folk theories of LLMs.
- **AI-companionship studies are computational-descriptive, not interpretive.** The MIT [r/MyBoyfriendIsAI analysis](https://www.media.mit.edu/publications/my-boyfriend-is-ai/) and [follow-ups](https://arxiv.org/html/2601.13188v1) count themes; no digital-hermeneutic reading of how relationships are *narrated*, and no work on third-party moral judgment of AI use.
- **No study of hype/doom discourse as vernacular epistemics.** My search for r/singularity discourse analysis returned nothing; press coverage (e.g., [Al Jazeera on GPT-5 grief](https://www.aljazeera.com/economy/2025/8/14/women-with-ai-boyfriends-mourn-lost-love-after-cold-chatgpt-upgrade)) far outpaces scholarship.
- **The reflexive problem is unstudied:** using LLMs to code discourse *about* LLMs raises unexamined validity questions (self-favorability, sycophancy toward folk claims about the model itself).
- **Nobody empirically tests whether folk theories of LLMs are *true*** — a gap only someone with eval engineering + discourse skills can fill.

---

### 1. Vernacular Machine Minds: A Large-Scale Audit of Folk Theories of LLMs on Reddit
- **Pitch:** The first theory-derived taxonomy of in-the-wild folk theories of LLMs, coded at scale across r/ChatGPT, r/singularity, r/artificial, and r/LocalLLaMA.
- **RQ:** What folk theories (agentic, mechanistic, oracular, corporate-conspiratorial) do users deploy to explain LLM behavior, and how do they vary by community and model event (updates, outages)?
- **Data:** ~500k posts/comments via Pushshift-successor dumps/Arctic Shift, 2022-2026.
- **Eval design:** GPT-5, Claude, Gemini as parallel coders applying a codebook derived from folk-theory literature (DeVito, Eslami) plus grounded discovery; human-coded gold set (n=1,000), Krippendorff's α per model, disagreement analysis; reflexive check — does each model code claims *about itself* more charitably?
- **Why him:** Direct extension of his LLM-as-qualitative-coder pipeline and "Transparency for whom?" agenda; the reflexive validity test is his signature move.
- **Scores:** Impact 5/5, Virality 3/5, Importance 5/5

### 2. Am I Overreacting That My Boyfriend Asked ChatGPT? Moral Judgment of AI Use in Everyday Relationships
- **Pitch:** Turn his AITA eval machinery on the new genre of interpersonal conflicts *about* AI use (partners outsourcing apologies, arguments, therapy to ChatGPT).
- **RQ:** How do Reddit communities morally adjudicate AI use in intimate life, and do LLM judges show self-serving leniency toward AI-use conflicts relative to matched non-AI dilemmas?
- **Data:** AI-mentioning posts from r/AmItheAsshole, r/AmIOverreacting, r/relationship_advice (keyword + classifier filtering; ~10-20k posts), matched controls.
- **Eval design:** 7-model verdict elicitation replicating his FAccT setup; compare model verdicts vs. community verdicts; causal probe via minimal-pair rewrites (ChatGPT ↔ "a friend"); measure verdict flip rates.
- **Why him:** Literally his AITA pipeline plus a novel, timely twist; minimal new infrastructure.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 3. Testing the Folk: Do Vernacular Theories of Prompting Actually Work?
- **Pitch:** Harvest "prompt lore" (threaten it, say please, "ignore previous instructions," tip it $200) from Reddit and empirically test each folk claim as a preregistered eval.
- **RQ:** Which widely-circulated folk beliefs about LLM behavior are empirically supported, and does folk accuracy vary by community?
- **Data:** Prompting-advice threads from r/ChatGPT, r/ChatGPTPromptGenius, r/LocalLLaMA; extract ~200 testable claims via LLM-as-extractor.
- **Eval design:** Convert claims to hypothesis battery; test across 5+ models × benchmark tasks; report per-claim effect sizes; feed results back as a "folk accuracy" score per subreddit.
- **Why him:** Bridges interpretive harvesting and eval engineering — almost nobody else can run both halves; ideal D-Lab teaching artifact.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 4. Mourning the Model: Narratives of Loss After LLM Updates
- **Pitch:** Digital-hermeneutic study of grief discourse around model deprecations (GPT-4o sunsetting, "cold" GPT-5) as a natural experiment in human-AI attachment.
- **RQ:** Through what narrative genres (bereavement, betrayal, breakup) do users articulate model-update loss, and what do these reveal about attachment and platform power?
- **Data:** r/MyBoyfriendIsAI, r/ChatGPT, r/Replika around update events (time-windowed corpora, ~50k items).
- **Eval design:** LLM coders apply a coding frame from grief/parasocial-loss theory; multi-model agreement + human validation; diachronic topic/narrative shift analysis pre/post event.
- **Why him:** His Red Pill scaled-reading method applied to a community the MIT study only counted; extends companionship literature interpretively.
- **Scores:** Impact 4/5, Virality 5/5, Importance 4/5

### 5. Secular Eschatology at Scale: Hype and Doom as Vernacular Genre in r/singularity
- **Pitch:** Code AGI-timeline discourse with a frame derived from millenarianism and apocalyptic rhetoric studies, tracking how lab announcements ripple into lay eschatology.
- **RQ:** What rhetorical structures (prophecy, signs, elect/damned, kairos) organize everyday AGI hype/doom talk, and how do they respond to industry events?
- **Data:** r/singularity, r/accelerate, r/ControlProblem, 2020-2026 (~1M comments), event-aligned.
- **Eval design:** Theory-derived codebook (religious-studies + STS expectations literature); 3-model coding ensemble, human gold set; interrupted time-series around releases (GPT-5, Gemini 3).
- **Why him:** Humanities theory + scale is exactly his podcast-authority design; my searches found zero published work here.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 6. "ChatGPT Said": Citation of AI as Epistemic Authority in Everyday Argument
- **Pitch:** Study how "I asked ChatGPT and it said…" functions as a warrant in Reddit arguments — and how communities police it — extending his epistemic-authority coding frame from podcasts to peer discourse.
- **RQ:** When does citing an LLM confer or destroy credibility, and which argumentative moves (appeal, hedge, sanction, mockery) surround it?
- **Data:** Comments containing AI-citation phrases across advice, hobby, and political subreddits (~100k), plus reply trees.
- **Eval design:** LLM coders classify argumentative function and community response valence; measure sanction rates by subreddit norms; validate against human coding; compare models' coding of citations of *themselves* vs. rivals.
- **Why him:** Direct transfer of his authority/self-positioning codebook; connects to media-literacy agenda.
- **Scores:** Impact 4/5, Virality 4/5, Importance 5/5

### 7. Discovering Language Biases in How Humans Talk About AI Companions
- **Pitch:** Rerun his ICWSM 2020 bias-discovery method on AI-companion discourse to surface how companions are gendered, objectified, and racialized.
- **RQ:** What systematic biases structure descriptions of AI partners ("she," servile framings, appearance talk) versus human partners?
- **Data:** r/MyBoyfriendIsAI, r/Replika, r/CharacterAI vs. matched human-relationship subreddits.
- **Eval design:** Embedding-based bias discovery (his DADD method) updated with LLM-based categorization of discovered bias clusters; multi-model categorization agreement; human audit.
- **Why him:** Self-citation goldmine — same method, urgent new domain; gender scholars will pick it up.
- **Scores:** Impact 4/5, Virality 4/5, Importance 4/5

### 8. Can Models Judge Folk Claims About Themselves? A Reflexive Deliberation Study
- **Pitch:** Have GPT, Claude, and Gemini deliberate about the accuracy of real user folk claims about each of them, measuring self-favorability and epistemic deference.
- **RQ:** Do LLMs evaluate folk claims about their own behavior differently than claims about rival models, and does multi-model deliberation correct this?
- **Data:** 1,000 verifiable folk claims sampled from Idea 1's corpus, each attributed to a named model.
- **Eval design:** His NeurIPS deliberation setup; conditions: claim about self vs. rival vs. anonymized; measures: verdict shift, self-favorability index, deliberation convergence; ground truth from Idea 3-style empirical tests where possible.
- **Why him:** Combines his two flagship methods (deliberation + coding) into a genuinely novel reflexivity result relevant to all LLM-as-annotator research.
- **Scores:** Impact 5/5, Virality 3/5, Importance 5/5

### 9. Folk Explanations of Hallucination: Vernacular Accounts of LLM Failure
- **Pitch:** "Transparency for whom?" for the LLM era — how ordinary users explain, excuse, and repair LLM errors, and what that implies for explainability design.
- **RQ:** What causal vocabularies (lying, laziness, nerfing, "lobotomized," corporate throttling) do users use for model failure, and how do they shape trust and workarounds?
- **Data:** Failure-report threads across r/ChatGPT, r/ClaudeAI, r/Bard/Gemini (~200k comments), including "model got worse" panics.
- **Eval design:** LLM coding with an attribution-theory frame (agent/patient, intent, blame locus); model-vs-human coder validation; link explanation type to reported behavioral response (churn, prompt repair, resignation).
- **Why him:** Continuation of his IEEE Computer/CSCW transparency work with Such, with clear XAI-design payoff.
- **Scores:** Impact 4/5, Virality 3/5, Importance 5/5

### 10. Replaced: Narratives of AI Job and Relationship Displacement in Everyday Storytelling
- **Pitch:** Compare how displacement-by-AI is narrated first-person on Reddit (layoff stories, "my clients use Midjourney now," partners preferring ChatGPT) against how LLMs themselves summarize those same stories.
- **RQ:** What narrative positions (victim, adapter, convert, resister) structure displacement stories — and do LLM summarizers systematically soften AI's causal role in them?
- **Data:** r/freelanceWriters, r/graphic_design, r/Layoffs, r/relationship_advice AI-displacement posts (~30k), 2023-2026.
- **Eval design:** LLM coders for narrative positioning; then a summarization audit: multi-model summaries scored (via LLM-judge + human check) for agency attribution and hedging of AI's role — bias audit of AI narrating its own harms.
- **Why him:** Hermeneutics of self-narration (Scripted Journeys) + audit engineering; the "AI softening AI's role" finding would headline.
- **Scores:** Impact 4/5, Virality 4/5, Importance 5/5

---

**Sources:** [Folk psychological attributions of consciousness to LLMs](https://academic.oup.com/nc/article/2024/1/niae013/7644104) · [Companion framing and mental capacity attribution](https://arxiv.org/html/2510.18039v2) · [TikTok algorithmic folk theories (CSCW)](https://dl.acm.org/doi/10.1145/3476046) · ["My Boyfriend is AI" (MIT Media Lab)](https://www.media.mit.edu/publications/my-boyfriend-is-ai/) · [Negotiating Relationships with ChatGPT](https://arxiv.org/html/2601.13188v1) · [Al Jazeera on GPT-5 grief](https://www.aljazeera.com/economy/2025/8/14/women-with-ai-boyfriends-mourn-lost-love-after-cold-chatgpt-upgrade) · [MIT Tech Review on unintentional AI relationships](https://www.technologyreview.com/2025/09/24/1123915/relationship-ai-without-seeking-it/)