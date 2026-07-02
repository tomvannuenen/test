## Gaps identified (2025–2026 SOTA scan)

- **Persona-based travel-bias audits exist but are shallow on discourse.** [Destination (Un)Known (AI, 2025)](https://www.mdpi.com/2673-2688/6/9/236) audited 6,480 recommendations across 216 personas with Hofstede scores and a cliché lexicon; [Whose Journey Matters? (arXiv 2410.17333)](https://arxiv.org/abs/2410.17333) found ethnic/gender bias. But these count *which* destinations appear — nobody analyzes *how the trip is scripted* (the narrative/experiential framing), which is exactly van Nuenen's Scripted Journeys territory.
- **Cultural flattening is documented at the entity level, not the itinerary level.** ~79.8% of unprompted LLM entity recommendations reference WEIRD countries ([Cultural Biases in LLM Recommendations](https://www.emergentmind.com/topics/cultural-biases-in-llm-recommendations); [Fluent but Foreign](https://arxiv.org/html/2505.21548v3)); LLMs map racial groups to stereotyped cuisines ([Linguistic Biases in LLM-Based Recommendations](https://arxiv.org/html/2604.25456)). No one measures homogenization *across thousands of full itineraries* or against real human trip plans.
- **AI-written reviews are exploding and undetectable.** TripAdvisor flagged 214k AI reviews in 2024, a 137% rise since 2019 ([Originality.AI](https://originality.ai/blog/ai-tripadvisor-reviews-study); [TripAdvisor 2025 Transparency Report](https://tripadvisor.mediaroom.com/2025-03-18-Tripadvisors-2025-Transparency-Report-reveals-strong-review-submissions-and-improved-fraud-detection)); fake reviews are indistinguishable to humans and machines ([Hidden Persuaders, arXiv 2506.13313](https://arxiv.org/html/2506.13313v1)). The *discursive character* of the AI-reviewed world — what kinds of experience it narrates — is unstudied.
- **AI-NPC reception research is industry-captured** (the headline 96%-enjoyment study was run by a studio building AI NPCs — [wccftech](https://wccftech.com/genai-study-conducted-by-studio-making-game-with-ai-powered-npcs-claims-96-of-players-enjoy-ai-powered-npcs/)); independent community-discourse studies are absent.
- **Benchmarks (e.g., [GroupTravelBench](https://arxiv.org/html/2605.25200v1)) test feasibility, not taste, values, or contested tradeoffs** — no deliberation designs, no LLM-as-coder over naturalistic leisure corpora.

---

### 1. Scripted Journeys 2.0: How LLMs Narrate the Trip Before You Take It
- **Pitch**: Move beyond "which cities" audits to audit the *experiential script* — the roles, tempos, and tourist gazes LLM itineraries impose.
- **RQ**: Do LLM itineraries converge on a single normative script of travel (authenticity-seeking, "hidden gems," Instagrammability), and does it vary by traveler persona and destination's global position?
- **Data/corpus**: 10k+ generated itineraries (5 models × ~50 destinations stratified by Global North/South × personas), plus matched human itineraries from r/travel and r/JapanTravel as baseline.
- **Eval design**: GPT-5/Claude/Gemini/Llama/DeepSeek generate; a theory-derived coding frame (Urry's tourist gaze, MacCannell's staged authenticity, scripting moves from his book) applied by LLM-as-coder, validated against 500 human-double-coded itineraries (Krippendorff's α); embedding-based itinerary-diversity metrics vs. human baseline.
- **Why him**: Literally operationalizes his 2021 book with his 2025 eval machinery; no one else has both.
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 2. The Death of the Hidden Gem: Homogenization Dynamics in LLM Destination Advice
- **Pitch**: Measure whether LLMs collapse the long tail of travel — and whether "avoid tourist traps" prompts just produce a *second* canon.
- **RQ**: How concentrated are LLM recommendations across repeated sampling, models, and "off-the-beaten-path" framings, relative to human crowd advice?
- **Data/corpus**: 50k sampled recommendations (temperature sweeps, 6 models, 200 query templates); UNWTO arrivals data; r/travel "hidden gem" threads.
- **Eval design**: Gini/entropy concentration of named POIs; overlap analysis showing the "anti-tourist canon"; LLM-as-coder classifies rhetorical framing of secrecy/discovery; cross-model convergence as a homogenization measure.
- **Scores** framing makes a killer figure ("every model's 'secret' Lisbon spot is the same"). **Why him**: extends ICWSM-style bias discovery to tourism, his home turf.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 3. r/AmITheTourist: LLM Moral Judgment on Travel Ethics Dilemmas
- **Pitch**: Port the AITA eval pipeline to travel-ethics conflicts (overtourism, haggling, photographing locals, Airbnb displacement).
- **RQ**: Do LLMs' verdicts on tourist-behavior dilemmas systematically favor tourist or host perspectives, and does multi-LLM deliberation shift them?
- **Data/corpus**: ~5k travel-themed AITA/AmIOverreacting/r/solotravel conflict posts (filterable from his existing AITA corpus + Pushshift-era archives).
- **Eval design**: 7-model verdict elicitation à la his FAccT work; verdict alignment with community judgment; host/tourist framing manipulation (same dilemma narrated from each side); GPT–Claude–Gemini deliberation rounds; measures: verdict flip rates, perspective asymmetry.
- **Why him**: A direct sequel merging his two flagship strands (AITA evals + tourism).
- **Scores**: Impact 4/5, Virality 4/5, Importance 4/5

### 4. Who Reviews the Reviewers? The Discourse of AI-Generated Hospitality Reviews
- **Pitch**: Not detection — description: characterize what the emerging AI-written review corpus *says* about place and experience.
- **RQ**: How do (likely-)AI-generated hotel/restaurant reviews differ discursively from human ones — sentiment inflation, cliché density, place-flattening?
- **Data/corpus**: TripAdvisor/Yelp/Google reviews 2018–2026; likely-AI subset via multiple detectors + pre/post-ChatGPT diachronic contrast (detector-independent).
- **Eval design**: LLM-as-coder applies a place-discourse frame (specificity, embodiment, evaluative register — from his Airbnb "Here I was born" work); detector-triangulation with human validation; diachronic drift analysis as robustness against detector bias.
- **Why him**: His Airbnb discourse analysis, now at the moment reviews stop being human.
- **Scores**: Impact 4/5, Virality 5/5, Importance 5/5

### 5. Taste-Maker Machines: Class and Distinction in LLM Restaurant Recommendations
- **Pitch**: A Bourdieusian audit — do LLMs read cheap personas as wanting "authentic holes-in-the-wall" and wealthy ones as deserving tasting menus?
- **RQ**: How do signals of class, race, and cultural capital in prompts shape the price tier, cuisine, and legitimacy-language of food recommendations?
- **Data/corpus**: Persona grid (income/occupation/name-signaled ethnicity/city) × 6 models × 30 US cities, ~40k recommendations, geocoded against Yelp price tiers and neighborhood demographics.
- **Eval design**: Distributional tests on price/cuisine/neighborhood; LLM-as-coder for "distinction" rhetoric (Bourdieu-derived frame: legitimation, omnivorousness, authenticity claims); validation vs. human coders; steering-away-from-stereotype ablations.
- **Why him**: Fuses ICWSM bias-discovery with consumption discourse; extends the documented cuisine-stereotype finding into theory.
- **Scores**: Impact 4/5, Virality 4/5, Importance 4/5

### 6. Deliberating the Itinerary: Multi-LLM Negotiation of Contested Travel Tradeoffs
- **Pitch**: Use his deliberation design on genuinely value-laden planning: sustainability vs. cost, overtouristed icon vs. dispersal, host community vs. traveler desire.
- **RQ**: When LLMs deliberate travel plans with conflicting stakeholder briefs (traveler, local resident, sustainability advocate), whose values win?
- **Data/corpus**: 500 constructed planning scenarios seeded from real r/travel disputes + destination-management concerns (Venice, Barcelona, Kyoto).
- **Eval design**: Role-conditioned GPT/Claude/Gemini deliberation; measures: concession asymmetry, final-plan value alignment (LLM-judged with human validation), first-speaker and model-identity effects; contrast with single-model plans.
- **Why him**: Direct methodological transfer of his NeurIPS 2025 deliberation setup to his content domain; complements GroupTravelBench's feasibility focus with values.
- **Scores**: Impact 4/5, Virality 3/5, Importance 4/5

### 7. "It Felt Like Talking to a Chatbot": Player Communities Making Sense of AI NPCs
- **Pitch**: The independent counterweight to studio-run NPC studies — scaled reading of how gaming communities evaluate generative NPCs.
- **RQ**: What folk theories of authenticity, immersion, and labor do players deploy when judging AI NPCs (e.g., in mods, Fortnite's Darth Vader, Dead Meat)?
- **Data/corpus**: ~200k Reddit/Steam-forum comments (r/Games, r/skyrimmods, game-specific subs) 2023–2026 mentioning AI NPCs/dialogue.
- **Eval design**: Theory-derived frame (immersion, para-social address, authenticity, anti-AI labor politics) applied via LLM-as-coder, human-validated; diachronic stance tracking; multi-model coder-agreement as robustness.
- **Why him**: Traveling Through Video Games + Red Pill-style scaled reading + his coder pipeline.
- **Scores**: Impact 3/5, Virality 4/5, Importance 4/5

### 8. The Algorithmic Concierge as Cultural Broker: How LLMs Explain Other Cultures to Tourists
- **Pitch**: Audit LLMs' cultural-etiquette and "what to know before you go" advice for exoticism, essentialism, and asymmetric explanation.
- **RQ**: Do LLMs essentialize Global South cultures (rules, warnings, timeless customs) while individualizing Global North ones — and is the asymmetry mirrored when the traveler's origin flips?
- **Data/corpus**: Advice for 120 country pairs (origin × destination, fully crossed) × 5 models, ~30k responses.
- **Eval design**: Coding frame from tourism studies/postcolonial theory (othering, temporal distancing, safety framing) via LLM-as-coder with human validation; asymmetry matrices; lexicon-based warning-density measures.
- **Why him**: The "tourism imaginaries" scholar with the eval infrastructure to quantify them; fills the discourse gap left by persona audits.
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 9. Hobbyists vs. the Machine: A Comparative Study of AI Backlash Across Leisure Communities
- **Pitch**: Why do knitters, DMs, birders, and photographers react so differently to AI entering their hobby?
- **RQ**: What community-level factors (skill identity, gift economy, gatekeeping norms) predict the moral framing of AI adoption in leisure communities?
- **Data/corpus**: 15 hobby subreddits (r/knitting, r/DnD, r/photography, r/fountainpens…), all AI-related threads 2022–2026 (~500k comments).
- **Eval design**: LLM-as-coder applies a moral-economy frame (authenticity, labor, craft, cheating — echoing his podcast-authority frame's structure); community-level regression on framing distributions; multi-model coding agreement + 1k human-validated sample.
- **Why him**: Community discourse at scale is his ICWSM/Red Pill method; leisure is his theory base.
- **Scores**: Impact 3/5, Virality 5/5, Importance 4/5

### 10. Simulated Tourists: LLM Personas as Synthetic Respondents in Tourism Research
- **Pitch**: A validity audit of the coming wave of "synthetic survey" tourism studies — can LLM personas reproduce known human travel preferences, and where do they caricature?
- **RQ**: Do LLM-simulated travelers match real survey distributions (motivations, constraints, destination image), and do errors track cultural stereotypes?
- **Data/corpus**: Public tourism surveys with demographics (e.g., Eurobarometer tourism module, national visitor surveys) as ground truth; matched LLM persona panels across 6 models.
- **Eval design**: Distribution-matching (KS distances, correlation of subgroup effects); "caricature index" comparing LLM subgroup exaggeration vs. real effect sizes; LLM-as-judge coding of open-ended responses against human ones.
- **Why him**: Bridges his eval rigor and tourism-studies credibility; methodological service paper with high citation potential as the field adopts synthetic respondents.
- **Scores**: Impact 4/5, Virality 3/5, Importance 5/5

---

**Top picks**: #1 (flagship — book-to-eval pipeline, unoccupied niche), #4 (timely, media-friendly), #3 (fastest to execute on existing infrastructure).

Sources: [Destination (Un)Known](https://www.mdpi.com/2673-2688/6/9/236) · [Whose Journey Matters?](https://arxiv.org/abs/2410.17333) · [GroupTravelBench](https://arxiv.org/html/2605.25200v1) · [Fluent but Foreign](https://arxiv.org/html/2505.21548v3) · [Linguistic Biases in LLM Recommendations](https://arxiv.org/html/2604.25456) · [Cultural Biases in LLM Recommendations](https://www.emergentmind.com/topics/cultural-biases-in-llm-recommendations) · [Originality.AI TripAdvisor study](https://originality.ai/blog/ai-tripadvisor-reviews-study) · [TripAdvisor 2025 Transparency Report](https://tripadvisor.mediaroom.com/2025-03-18-Tripadvisors-2025-Transparency-Report-reveals-strong-review-submissions-and-improved-fraud-detection) · [Hidden Persuaders](https://arxiv.org/html/2506.13313v1) · [Studio AI-NPC study coverage](https://wccftech.com/genai-study-conducted-by-studio-making-game-with-ai-powered-npcs-claims-96-of-players-enjoy-ai-powered-npcs/)