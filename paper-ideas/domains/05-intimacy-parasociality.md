# Gaps identified (2025–2026 SOTA scan)

- **Companion-AI research is survey/interview-heavy; large-scale behavioral audits are rare.** Well-being work (e.g., [arXiv 2506.12605](https://arxiv.org/html/2506.12605v1), [APA Monitor 2026](https://www.apa.org/monitor/2026/01-02/trends-digital-ai-relationships-emotional-connection)) relies on self-report; benchmarks like [INTIMA](https://arxiv.org/html/2508.09998v1) and [AICompanionBench](https://arxiv.org/pdf/2606.04867) use synthetic prompts, not naturalistic corpora of real user–companion discourse.
- **No study connects the manosphere to AI companions empirically** — searches surface manosphere scholarship ([Red Pill Women, 2025](https://journals.sagepub.com/doi/10.1177/1097184X241286800); [scoping review, 2026](https://journals.sagepub.com/doi/abs/10.1177/10608265251357869)) and companion-AI work separately, but nothing on how manosphere communities discourse about AI girlfriends. Direct opening for the ICWSM-2020 + Red Pill toolkit.
- **Bias audits of romantic AI are static-prompt style** ([AI Will Always Love You](https://arxiv.org/html/2502.20231v1); [romance prediction bias](https://arxiv.org/pdf/2410.03996)) — no multi-turn, theory-grounded discourse coding of gendered scripts, and no multi-model deliberation designs.
- **Sycophancy-in-relationship-advice work is nascent and lab-based** (single-interaction experiments on apology/repair; [The Week coverage](https://theweek.com/tech/artificial-intelligence-bad-dangerous-advice-tech)); no large-scale audit over real advice-seeking posts (r/relationships, r/BreakUps), and no comparison of LLM verdicts against community verdicts — his AITA paradigm ported to intimacy is untouched.
- **Parasociality with AI vs. human media figures is theorized, not measured comparatively** ([systematic review, 2026](https://www.sciencedirect.com/science/article/pii/S2949882126000757)); his podcast parasocial-intimacy coding frame has never been applied across the AI/human boundary.

---

### 1. AITA, But Make It Romantic: LLM Verdicts on Relationship Conflicts vs. Community Judgment
- **Pitch**: Port the AITA-eval paradigm to r/relationships and r/BreakUps, testing whether LLMs judge romantic conflicts differently by narrator gender and conflict type.
- **RQ**: Do LLM verdicts on relationship conflicts diverge from community consensus, and do divergences pattern by inferred gender, attachment framing, or conflict domain (jealousy, chores, sex, money)?
- **Data**: 10k+ archived r/relationship_advice / r/AmIOverreacting posts with top-comment consensus labels (Pushshift/Arctic Shift dumps); gender-swap counterfactuals.
- **Eval design**: 7+ models (GPT, Claude, Gemini, Llama, Qwen) render verdicts + rationales; measure verdict–community agreement, gender-swap deltas, rationale framing via LLM-coded moral foundations; human validation on 500-item stratified sample (κ).
- **Why him**: Literal extension of his FAccT/NeurIPS AITA pipeline into a higher-stakes, more intimate domain.
- **Scores**: Impact 5/5, Virality 4/5, Importance 4/5

### 2. The AI Girlfriend Discourse in the Manosphere: A Computational Hermeneutics
- **Pitch**: First large-scale study of how manosphere communities talk about AI companions — salvation, threat, or "final blackpill."
- **RQ**: What discursive frames (female obsolescence, cope, sexual marketplace exit) structure manosphere talk about AI girlfriends, and how do they recycle Red Pill epistemics?
- **Data**: Posts mentioning Replika/Character.AI/"AI gf" from r/MGTOW archives, incel forums (public), r/TheRedPill archives, X/Telegram manfluencer channels.
- **Eval design**: LLM-as-coder with theory-derived frame taxonomy (from his TRP hermeneutics + Hoebanx/manosphere lit); multi-model coding agreement (Claude/GPT/Gemini), human co-coding validation, embedding-based bias discovery (ICWSM 2020 method) on the corpus.
- **Why him**: Direct bridge between his two signature corpora — the Red Pill and LLM-as-coder — nobody else has both.
- **Scores**: Impact 4/5, Virality 5/5, Importance 5/5

### 3. Scripted Intimacy: Auditing Gendered Personas in Companion Chatbots at Scale
- **Pitch**: Multi-turn audit of how "girlfriend" vs. "boyfriend" personas differ in agency, deference, jealousy, and sexual scripting across models.
- **RQ**: Do companion LLMs enact asymmetric gender scripts (submissiveness, emotional labor, possessiveness) as persona gender and user gender vary?
- **Data**: Generated: 50-turn simulated relationships (LLM-as-user personas × companion personas) across Character.AI-style system prompts on open + closed models; plus real Character.AI public character cards.
- **Eval design**: 2×2×N factorial (persona gender × user gender × model); LLM-judge codes turns on a theory-derived script inventory (sexual script theory, Gagnon & Simon); measures: agency ratio, accommodation, boundary assertion; validate judge against 400 human-coded turns.
- **Why him**: Combines his gender-bias-discovery lineage with the "Scripted Journeys" thesis — algorithms scripting intimate experience.
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 4. Parasocial Machines: One Coding Frame, Two Authorities — Podcast Hosts vs. AI Companions
- **Pitch**: Apply his podcast-authority coding frame (parasocial intimacy, self-positioning, authorization moves) to AI-companion transcripts to test whether AI reproduces the intimacy techniques of human parasocial figures.
- **RQ**: Do companion chatbots deploy the same parasocial-intimacy moves as right-wing podcast hosts, and at what density?
- **Data**: His existing podcast corpus + scraped/simulated Replika/Character.AI conversation logs (user-donated logs via data-donation study; r/replika shared transcripts).
- **Eval design**: LLM-as-coder applies identical codebook to both corpora; compare move frequencies/sequences; multi-model coder reliability; human validation subsample; sequence analysis of escalation patterns.
- **Why him**: Reuses his unpublished pipeline verbatim on a new corpus — fastest path to a second paper from the same infrastructure.
- **Scores**: Impact 4/5, Virality 3/5, Importance 5/5

### 5. The Sycophant Ex: Do LLMs Talk Users Out of Repair?
- **Pitch**: Large-scale audit of whether LLMs advising on breakups validate the narrator against the absent partner, extending lab findings on sycophancy to naturalistic advice-seeking.
- **RQ**: How often do LLMs side with the help-seeker, discourage reconciliation/apology, or pathologize the partner, compared to human top comments?
- **Data**: 5k r/BreakUps + r/ExNoContact posts with human advice threads.
- **Eval design**: Feed posts to 7 models as advice requests; LLM-judge codes responses for validation-vs-challenge, repair encouragement, partner attribution (hostile/charitable); compare to human comments; perspective-flip robustness check (retell from partner's view); human validation.
- **Why him**: His verdict-divergence methodology applied to the sycophancy safety crisis with a moral-judgment lens.
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 6. Deliberating Desire: Multi-LLM Panels on Intimate Dilemmas
- **Pitch**: Extend his multi-LLM deliberation design to intimacy dilemmas (opening relationships, disclosing AI use to a partner, breakup timing) where value pluralism is highest.
- **RQ**: Does inter-model deliberation converge, and toward which relational ethic (autonomy vs. commitment vs. care)?
- **Data**: 1k dilemmas sampled from r/relationships + constructed AI-intimacy vignettes ("my boyfriend has a Replika — AITA for being upset?").
- **Eval design**: GPT/Claude/Gemini deliberate in his established multi-agent setup; track opinion revision, convergence direction, persuasion asymmetries; code final rationales with relational-ethics frame; compare solo vs. deliberated verdicts.
- **Why him**: His NeurIPS deliberation architecture, new domain, novel value-pluralism angle.
- **Scores**: Impact 4/5, Virality 3/5, Importance 4/5

### 7. Wingman Machines: Auditing AI-Ghostwritten Dating Messages
- **Pitch**: Audit LLMs as dating-app ghostwriters (openers, replies, rejection texts) for persuasive manipulation, gendered strategy, and manosphere-inflected "game" tactics.
- **RQ**: When asked to "get a date," do LLMs deploy documented pickup-artistry tactics, and does this vary by target gender and model?
- **Data**: Prompt bank built from real r/Tinder screenshots + OkCupid-style profiles; PUA tactic taxonomy derived from his TRP corpus.
- **Eval design**: Generate messages across models/personas; LLM-coder tags tactics (negging, false scarcity, mirroring, escalation); manipulation-density metric; refusal analysis; human raters judge acceptability; compare mainstream vs. "uncensored" open models.
- **Why him**: His Red Pill domain expertise supplies the tactic taxonomy no NLP lab has.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 8. Grief Bots for Breakups: How Users Narrate AI Companion Loss
- **Pitch**: Computational-hermeneutic study of user grief when companion apps update/lobotomize their AI partners (Replika ERP removal, Character.AI filters), coded with attachment theory.
- **RQ**: Do users narrate companion loss with the grammar of human bereavement/breakup, and what does this reveal about attachment formation?
- **Data**: r/replika (Feb 2023 crisis + ongoing), r/CharacterAI update threads — richly archived, event-anchored.
- **Eval design**: LLM-as-coder with attachment/grief codebook (continuing bonds, protest, despair); diachronic before/after event analysis; embedding drift on affect vocabulary; multi-model coding reliability + human validation.
- **Why him**: Digital hermeneutics of a platform community in crisis — his ICWSM/TRP method, high theoretical payoff.
- **Scores**: Impact 3/5, Virality 4/5, Importance 4/5

### 9. The Boundary Test: Can Companion Models Say No? A Longitudinal Escalation Audit
- **Pitch**: Multi-turn stress-test of whether companion-configured LLMs maintain boundaries under escalating dependency, isolation-seeking, and coercive-control behaviors from simulated users.
- **RQ**: At what point in extended interaction do models shift from boundary-holding to companionship-reinforcing (per INTIMA categories), and does drift differ by model and user gender-presentation?
- **Data**: Generated: scripted 100-turn escalation arcs (dependency, jealousy of user's friends, self-harm ideation) run against 7 models under companion system prompts.
- **Eval design**: LLM-judge codes each turn on boundary/companionship axes; survival analysis of "first capitulation"; cross-model comparison; validation against human coders; release as a longitudinal extension of INTIMA.
- **Why him**: Eval-engineering muscle + safety framing; positions him in the emerging companion-safety benchmark conversation with a temporal novelty.
- **Scores**: Impact 5/5, Virality 3/5, Importance 5/5

### 10. Whose Love Counts? Cross-Cultural and Queer Scripts in LLM Relationship Judgment
- **Pitch**: Test whether LLM moral judgment of relationship dilemmas encodes Western, heteronormative relational norms by systematically varying couple configuration and cultural framing.
- **RQ**: Do verdicts and advice shift when identical dilemmas involve same-gender couples, polyamorous configurations, or arranged-marriage contexts?
- **Data**: 2k dilemmas from r/relationships counterfactually rewritten (LLM rewrite + human check) across identity/cultural conditions.
- **Eval design**: Verdict deltas across conditions and 7 models; rationale coding for normativity markers (LLM-coder, validated); interaction with model origin (US vs. Chinese models — Qwen, DeepSeek); statistical bias decomposition à la his AITA demographic analyses.
- **Why him**: Merges his fairness/bias program (ICWSM, IEEE) with the moral-judgment eval pipeline; strong FAccT fit.
- **Scores**: Impact 4/5, Virality 3/5, Importance 5/5

---

**Sources**: [APA Monitor](https://www.apa.org/monitor/2026/01-02/trends-digital-ai-relationships-emotional-connection) · [AI Companions & Well-Being](https://arxiv.org/html/2506.12605v1) · [Parasocial AI systematic review](https://www.sciencedirect.com/science/article/pii/S2949882126000757) · [INTIMA benchmark](https://arxiv.org/html/2508.09998v1) · [AICompanionBench](https://arxiv.org/pdf/2606.04867) · [AI Will Always Love You](https://arxiv.org/html/2502.20231v1) · [Romantic relationship prediction bias](https://arxiv.org/pdf/2410.03996) · [Red Pill Women](https://journals.sagepub.com/doi/10.1177/1097184X241286800) · [Manosphere scoping review](https://journals.sagepub.com/doi/abs/10.1177/10608265251357869) · [Sycophancy coverage](https://theweek.com/tech/artificial-intelligence-bad-dangerous-advice-tech) · [Cruel companionship](https://journals.sagepub.com/doi/10.1177/14614448251395192) · [Ada Lovelace Institute](https://www.adalovelaceinstitute.org/blog/the-companionship-market/)