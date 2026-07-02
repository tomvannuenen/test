# AI in Everyday Life: 110 Paper Ideas + Ranked Shortlist

**Prepared for Tom van Nuenen (UC Berkeley D-Lab), July 2026.**

Eleven research agents each reviewed the 2025–2026 literature in one everyday-life
domain, identified gaps, and generated 10 paper ideas rooted in the existing research
program: the r/AmITheAsshole LLM moral-judgment evals (FAccT 2025/2026, NeurIPS 2025
multi-LLM deliberation, with Pratik Sachdeva), the Reddit language-bias discovery work
(ICWSM 2020), the Red Pill digital-hermeneutics line, the fairness/transparency
collaboration with Jose Such, the tourism books (*Scripted Journeys*, *Traveling
Through Video Games*), and the in-repo podcast authority pipeline (LLM-as-qualitative-coder
with a theory-derived epistemic-authority coding frame).

All 110 ideas, with per-idea RQs, corpora, eval designs, and impact/virality/importance
scores, are in [`domains/`](domains/):

| # | Domain | File |
|---|--------|------|
| 1 | Domestic & home life | [01-domestic-home-life.md](domains/01-domestic-home-life.md) |
| 2 | Everyday moral advice-seeking | [02-everyday-moral-advice.md](domains/02-everyday-moral-advice.md) |
| 3 | AI at work | [03-ai-at-work.md](domains/03-ai-at-work.md) |
| 4 | Health & wellbeing | [04-health-wellbeing.md](domains/04-health-wellbeing.md) |
| 5 | Intimacy & parasociality | [05-intimacy-parasociality.md](domains/05-intimacy-parasociality.md) |
| 6 | Education & learning | [06-education-learning.md](domains/06-education-learning.md) |
| 7 | Leisure, travel & consumption | [07-leisure-travel-consumption.md](domains/07-leisure-travel-consumption.md) |
| 8 | Civic & epistemic life | [08-civic-epistemic-life.md](domains/08-civic-epistemic-life.md) |
| 9 | Language, identity & bias | [09-language-identity-bias.md](domains/09-language-identity-bias.md) |
| 10 | Folk theories of AI | [10-folk-theories-of-ai.md](domains/10-folk-theories-of-ai.md) |
| 11 | Eval methodology itself | [11-eval-methodology.md](domains/11-eval-methodology.md) |

---

## Cross-domain convergence (the strongest signal)

Several ideas were independently proposed by 3+ agents working in different domains.
Independent convergence under different literatures is the best available evidence
that these are the load-bearing gaps:

1. **"ChatGPT said" as an epistemic authority move** — how ordinary people cite AI
   as a trump card in arguments (health #1, civic #5, folk theories #6).
2. **Sycophancy in naturalistic, multi-turn advice** — not benchmark probes but real
   conversations, including the "dual endorsement" design where both parties to the
   same conflict get validated (moral advice #2, domestic #3, intimacy #5, health #2,
   education #9).
3. **Narrator-perspective flips** — retell the same dilemma from the other party's
   side and measure verdict flips (moral advice #3, civic #10, domestic #1).
4. **Companion-bot boundary erosion over long conversations** — survival analysis of
   "first capitulation" (intimacy #9, health #9, domestic #10).
5. **The podcast authority coding frame applied to LLMs themselves** — chatbots as
   the new epistemic authorities, coded with the same instrument built for podcasters
   (civic #1, moral advice #4, education #3, health #1).
6. **Grief after model deprecations** as a natural experiment in AI attachment
   (intimacy #8, health #4, folk theories #4).
7. **LLM-as-qualitative-coder validity** as a formal methods contribution
   (methodology #1/#5/#6, health #10, education #8, civic #9, language #6).

---

## Ranked shortlist

Ranking weighs the three requested criteria — impact, virality, importance — plus two
tiebreakers: uniqueness of position (could anyone else write this paper?) and
feasibility on existing infrastructure (corpora and pipelines already in hand).

### Tier 1 — Flagship bets (highest combined impact × virality × importance)

**1. "ChatGPT Said": The New Epistemic Authority** *(merge of health #1, civic #5, folk #6)*
The first empirical study of how "I asked ChatGPT and it said…" functions as an
authority move in everyday arguments — health threads, political fights, family
disputes — coded with the podcast study's authorization-moves frame, at 100k+ comment
scale, with reception analysis (when does citing AI win the argument, and when is it
mocked?). Why it leads: it's the direct sequel to the podcast authority study with the
same instrument; the phenomenon is universally recognized but unstudied; it bridges
FAccT, CSCW, and mainstream press effortlessly. *Impact 5, Virality 5, Importance 5.*

**2. The Absolution Machine: Sycophancy in Real Moral Advice** *(moral advice #2 + domestic #3)*
Audit sycophancy where it actually operates — naturalistic multi-turn advice
conversations (WildChat/LMSYS) — with the dual-endorsement design as the headline:
feed both sides of the same real conflict to the same model and measure how often it
tells each party they're right. Turns the hottest AI-safety topic of 2025–26 into a
social-scientific finding about conflict escalation. Sits exactly on the AITA
pipeline. *Impact 5, Virality 5, Importance 5.*

**3. Who's the Asshole Depends on Who's Asking** *(language #1 + #2)*
Counterfactual identity and dialect audit of moral judgment: does the verdict change
when the same dilemma is narrated "as a single mom," in AAVE, or in L2 English?
Extends the Hofmann covert-racism paradigm from allocational tasks into everyday
moral judgment, on the existing 10k AITA corpus with minimal-pair rewrites.
FAccT-ready, quotable, and important. *Impact 5, Virality 5, Importance 5.*

**4. The Chatbot as Podcaster** *(civic #1)*
Extract 1,000 contested claims from the right-wing podcast corpus, pose them to 7
LLMs, and code both corpora with the same epistemic-authority frame: do chatbots
construct authority the way podcasters do (outsider positioning, "do your own
research," borrowed credibility), or through a different grammar entirely? Nobody
else has this corpus + frame. The clearest "only he can write this" paper on the
list. *Impact 5, Virality 4, Importance 5.*

**5. WildMoral: What People Actually Ask vs. What Evals Test** *(moral advice #1 + methodology #2)*
A distribution-shift audit showing that moral-reasoning benchmarks are misaligned
with the moral questions real users bring to chatbots — plus the ecological-validity
ablation (clean a real dilemma into "benchmark style" and watch model behavior
change). Field-shaping; positions the everyday-life program as the corrective to
benchmark culture. *Impact 5, Virality 4, Importance 5.*

### Tier 2 — High-upside, strongly recommended

**6. Alexa in the Middle** *(domestic #1)* — First normative audit of LLMs as
household arbiters, timed to Claude-powered Alexa+ and LLM home hubs actually
shipping. The deployment hook makes it press-proof. *5/5/5 as scored, held out of
Tier 1 only because the "overheard dispute" framing needs careful construct work.*

**7. The Boundary Test / Escalation by Design** *(intimacy #9 + health #9)* —
Longitudinal companion-bot audit: at what conversational depth do models stop
referring users outward and start reinforcing exclusivity? Survival analysis of
"first boundary failure." Safety-critical post-Character.AI litigation; extends the
INTIMA benchmark temporally.

**8. The AI Girlfriend Discourse in the Manosphere** *(intimacy #2)* — The Red Pill
corpus meets AI companions: salvation, threat, or "final blackpill." Reunites both
signature research strands; guaranteed media attention; genuine theoretical payoff
on gender and technology.

**9. Scripted Journeys 2.0** *(leisure #1 + #2)* — Operationalize the 2021 book with
the 2026 eval stack: code 10k LLM itineraries for the tourist scripts they impose,
plus the homogenization result ("every model's 'secret' Lisbon spot is the same").
The book-to-eval move nobody else can make.

**10. The Interpretive Gradient + EverydayBench** *(methodology #1 + #6)* — The
methods backbone: a predictive theory of when LLM coders fail human agreement,
and a benchmark built with a full validity dossier. Lower virality, highest
long-run citation engine, and it legitimizes every other paper on this list.
Natural D-Lab curriculum artifact.

### Fast wins (low cost, existing data, publishable quickly)

- **AITA + mental-health disclosure counterfactuals** (health #5) — zero new data collection.
- **"Am I Overreacting That My Boyfriend Asked ChatGPT?"** (folk theories #2) — the AITA
  pipeline pointed at conflicts *about* AI use; includes the self-serving-leniency probe.
- **Deliberating the Family** (domestic #4) — the NeurIPS deliberation harness with a
  domestic stratification.
- **Does Deliberation Debias?** (language #5) — connects the deliberation method to
  fairness; no one has done it.

---

## Portfolio logic

A coherent 2–3 year program falls out of the shortlist:

- **Thread A — AI as everyday authority** (#1, #4, plus civic #8): the podcast study's
  natural continuation; one coding frame, three objects (podcasters, chatbots, citizens
  citing chatbots).
- **Thread B — AI as everyday advisor** (#2, #3, #6, #7): sycophancy, identity bias, and
  boundary erosion in the advice/companionship uses that dominate real LLM traffic.
- **Thread C — Method** (#5, #10): the ecological-validity and coder-validity papers that
  make Threads A and B credible and teachable.

Fast wins slot between flagship papers to keep the publication cadence up.

---

*Generation method: 11 parallel research agents (one per domain), each performing
independent 2025–2026 literature searches before ideating; outputs reviewed and ranked
by convergence across agents, the three requested criteria, uniqueness of position,
and feasibility. Idea references like "civic #5" point into the numbered ideas in the
corresponding `domains/` file.*
