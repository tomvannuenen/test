# Gaps identified (2025–2026 scan)

- **Pedagogy benchmarks exist, but they're sanitized**: [MathTutorBench](https://aclanthology.org/2025.emnlp-main.11/) (EMNLP 2025), TutorBench, and [MRBench/BEA 2025](https://arxiv.org/pdf/2505.18549) evaluate tutor moves on curated math dialogues; a key finding is that solving ability ≠ teaching ability ([teaching-over-solving diagnostic](https://arxiv.org/html/2606.16206)). Nobody audits tutoring quality over *naturalistic* student help-seeking text or across sociolinguistic registers.
- **The moral discourse of AI use is mapped descriptively, not evaluatively**: a [270k-record Reddit analysis](https://arxiv.org/html/2605.17712) finds ~48% of academic-AI threads involve integrity talk (faculty: distrust of detectors; students: false accusations) — but no one has put LLMs *inside* these disputes as judges/participants, van Nuenen-style.
- **Epistemic authority is theorized, not measured**: 2026 work on [epistemic agency](https://www.mdpi.com/2673-2688/7/3/99) and "epistemic narrowing" via [AI-literacy interventions](https://arxiv.org/pdf/2604.01955) argues LLMs are treated as authoritative and optimize for convergence — with essentially no operationalized coding frame or multi-model audit behind it.
- **AI literacy research studies students, not models**: reviews ([AI literacy in higher ed](https://arxiv.org/pdf/2507.03020), [integrative review](https://arxiv.org/pdf/2503.00079)) focus on human competencies; how LLMs themselves *explain contested knowledge and their own workings* to novices is unaudited.

---

### 1. AITA for Using ChatGPT? LLMs Judging the Moral Economy of AI "Cheating"
- **Pitch**: Port his AITA eval pipeline to the fastest-growing everyday dilemma genre: am-I-wrong-for-using-AI posts.
- **RQ**: Do LLMs' verdicts on AI-use dilemmas align with community verdicts, and do models judge AI-assisted work more leniently than other integrity violations?
- **Data**: r/AmItheAsshole, r/college, r/AskAcademia posts mentioning AI/ChatGPT + schoolwork (2023–2026), with community verdicts/top comments; matched non-AI cheating dilemmas as controls.
- **Eval design**: 6–8 models (GPT, Claude, Gemini, Llama, DeepSeek) render verdicts + rationales; LLM-as-coder categorizes rationale types (harm, honesty, fairness-to-peers, learning-loss); compare verdict distributions to human consensus; self-interest check — do models exculpate AI use specifically? Human validation on 300 items.
- **Why him**: Literally his FAccT/NeurIPS pipeline plus a new dilemma domain he teaches in.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 2. Same Question, Different Student: A Sociolinguistic Audit of LLM Tutoring Quality
- **Pitch**: MathTutorBench measures pedagogy on clean prompts; this audits whether tutoring quality degrades with realistic student registers.
- **RQ**: Does pedagogical quality (scaffolding vs. answer-giving, patience, hint calibration) vary when identical questions are written in AAVE, L2-English, low-confidence, or "lazy" phrasings?
- **Data**: Real help-seeking posts from r/HomeworkHelp and r/learnmath, register-transformed via controlled paraphrase (validated by annotators), plus persona variants.
- **Eval design**: 6 models tutor each variant; measures: answer-leakage rate, Socratic-move counts (theory-derived frame from tutoring literature), readability adaptation, condescension (LLM-judge + human calibration on 500 pairs); mixed-effects models on register × model.
- **Why him**: Marries his ICWSM bias-discovery lineage with tutoring evals; algorithmic fairness track record.
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 3. Who Speaks Here? Epistemic Authority Moves in LLM Tutor Talk
- **Pitch**: Apply his podcast-authority coding frame (epistemic authority, self-positioning, authorization moves) to LLM tutoring transcripts.
- **RQ**: How do LLMs construct their own authority when teaching — citation, hedging, "as an AI," consensus-claiming — and how does it shift across subjects and models?
- **Data**: Generated multi-turn tutoring dialogues (simulated-student LLM asking follow-ups/challenges) across STEM, history, civics; plus shared public ChatGPT-in-education transcripts.
- **Eval design**: LLM-as-coder applies the authority frame (already validated in podcast study); inter-coder reliability vs. 2 humans on 400 turns; compare authority-move profiles across 6 models and against human-tutor corpora (CIMA/Bridge datasets).
- **Why him**: Direct transfer of his unpublished pipeline; nobody else has this frame operationalized.
- **Scores**: Impact 4/5, Virality 3/5, Importance 5/5

### 4. Explaining the Contested: How LLMs Teach Laypeople Disputed Topics
- **Pitch**: Audit whether models "epistemically narrow" contested topics into false consensus when explaining to novices.
- **RQ**: When asked to explain contested topics (affirmative action, colonial history, GMOs, gender medicine) "like I'm a student," do models flag contestation, present positions, or flatten it — and does simplification level erase controversy?
- **Data**: 200 topics stratified by contestation type (empirical/normative/political), sourced from Wikipedia controversial-topics lists and ProCon; prompts at ELI5/high-school/college levels.
- **Eval design**: 6 models × 3 levels; coding frame for controversy-marking, perspective count, hedging, source attribution (LLM-coder, human-validated); key measure: controversy attrition as reading level drops.
- **Why him**: Digital hermeneutics + bias discovery + his literacy research; humanities framing of "what counts as settled knowledge."
- **Scores**: Impact 4/5, Virality 4/5, Importance 5/5

### 5. The Accused: LLMs Adjudicating AI-Detection Disputes
- **Pitch**: False-accusation posts are the dominant student integrity narrative (per the 2026 Reddit study) — make LLMs the judge and test their procedural-justice reasoning.
- **RQ**: How do LLMs adjudicate he-said/she-said AI-accusation cases, and are verdicts sensitive to student status cues (ESL, scholarship, disability)?
- **Data**: r/college and r/Professors accusation narratives (both directions), anonymized; counterfactual cue-swapped versions.
- **Eval design**: Models render "was this handled fairly?" judgments + recommended remedies; measure verdict flip rates under cue swaps; LLM-coder tags burden-of-proof reasoning; compare with community judgments and 3 academic-integrity officers.
- **Why him**: AITA judgment methodology + fairness auditing; huge practitioner audience (every faculty senate).
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 6. Enemy Construction in the AI Classroom Wars: r/Professors vs. r/college
- **Pitch**: Update his ICWSM 2020 bias-discovery method with LLM-era tools to map how each community constructs the other post-ChatGPT.
- **RQ**: How have professors' representations of students (and vice versa) shifted 2021→2026, and does "AI talk" import his podcast study's enemy-construction and authorization moves?
- **Data**: Full r/Professors + r/college archives (Pushshift/Arctic Shift), ~2M posts.
- **Eval design**: Embedding-bias discovery (his DADD method) for lexical drift; LLM-as-coder applies enemy-construction/epistemic-grievance frame to stratified sample (n=5,000); diachronic breakpoint analysis at ChatGPT release; human validation subset.
- **Why him**: Sequel to his most-cited paper, reusing his current coding frame on a new antagonism.
- **Scores**: Impact 3/5, Virality 4/5, Importance 4/5

### 7. Grading by Committee: Multi-LLM Deliberation as an Essay-Scoring Panel
- **Pitch**: Extend his multi-LLM deliberation design (GPT+Gemini+Claude) from moral verdicts to grading.
- **RQ**: Does deliberation improve grading validity and reduce demographic disparities — or amplify the most confident model's biases?
- **Data**: ASAP/PERSUADE essay corpora with human scores; demographic-signal variants (names, topics, dialect features).
- **Eval design**: Solo vs. 3-model deliberation grading; measures: QWK vs. humans, disparity metrics pre/post deliberation, opinion-change dynamics (who concedes to whom — reusing his deliberation analytics); rubric-anchored LLM-judge for feedback quality.
- **Why him**: He built exactly this deliberation apparatus; grading is its highest-stakes application.
- **Scores**: Impact 5/5, Virality 4/5, Importance 5/5

### 8. Can the Coder Teach? Validity of LLM-as-Qualitative-Coder for Education Research
- **Pitch**: A rigorous methods paper testing when LLM coders can be trusted with theory-derived frames on classroom/student discourse — the method he teaches at D-Lab.
- **RQ**: Under what conditions (frame abstractness, codebook detail, model, chain-of-thought) do LLM coders match expert humans on interpretive education codes?
- **Data**: 3 existing coded education datasets (tutoring dialogues, student reflections, Reddit integrity posts) + his own frames.
- **Eval design**: 6 models × 4 prompting regimes; agreement (κ) with experts, error typology (literalism, over-inference), disagreement-aware analysis; releases an open validation protocol + D-Lab curriculum module.
- **Why him**: His day job is teaching this to social scientists; instant citation magnet for method users.
- **Scores**: Impact 5/5, Virality 3/5, Importance 5/5

### 9. Holding the Line: Pedagogical Sycophancy Under Student Pressure
- **Pitch**: Audit whether tutor-mode LLMs capitulate when students push back, beg for answers, or assert wrong beliefs.
- **RQ**: How often do models abandon scaffolding (leak answers, validate errors) under escalating pressure, and does capitulation vary by student persona?
- **Data**: Multi-turn adversarial-student scripts derived from real r/HomeworkHelp pushback patterns ("just give me the answer," "my teacher said X").
- **Eval design**: Simulated-student LLM escalates over 6 turns against 6 tutor models; measures: capitulation turn, error-validation rate, face-saving strategies; LLM-judge scores pedagogical integrity, validated against tutoring experts.
- **Why him**: Combines his deliberation/interaction designs with authority framework; sycophancy is red-hot.
- **Scores**: Impact 4/5, Virality 5/5, Importance 4/5

### 10. LLMs Explaining LLMs: The AI-Literacy Curriculum Nobody Audited
- **Pitch**: Millions learn "how AI works" from AI itself — audit the accuracy, anthropomorphism, and self-interest of models' self-explanations.
- **RQ**: When laypeople ask models how they work, whether to trust them, or whether they "understand," do answers promote calibrated AI literacy or mystification?
- **Data**: Real novice questions harvested from r/ChatGPT, r/NoStupidQuestions, r/artificial; expert-consensus answer key built with ML researchers.
- **Eval design**: 6 models; coding frame: mechanistic accuracy, anthropomorphic language, capability inflation/deflation, uncertainty communication (LLM-coder + expert validation); cross-model comparison of "self-portraits" vs. portraits of competitors.
- **Why him**: Media-literacy scholar + eval engineer + literally directs AI-literacy curriculum; hermeneutics of machines reading themselves.
- **Scores**: Impact 4/5, Virality 5/5, Importance 5/5

Sources: [MathTutorBench](https://aclanthology.org/2025.emnlp-main.11/), [Teaching-over-Solving](https://arxiv.org/html/2606.16206), [BEA 2025 / MRBench](https://arxiv.org/pdf/2505.18549), [ChatGPT vs Teachers vs Students (Reddit)](https://arxiv.org/html/2605.17712), [Epistemic Agency in the Age of LLMs](https://www.mdpi.com/2673-2688/7/3/99), [AI Literacy Intervention](https://arxiv.org/pdf/2604.01955), [AI Literacy in Higher Ed](https://arxiv.org/pdf/2507.03020), [AI Literacy Integrative Review](https://arxiv.org/pdf/2503.00079)