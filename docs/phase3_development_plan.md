# Phase 3 Development Plan: Quality and Evaluation

| | |
|---|---|
| **Status** | Draft, awaiting review of the Phase 3 study guide |
| **Phase** | 3 of 3. Measures Phases 1 and 2 and chooses the final configuration |
| **Source of truth** | `CLAUDE.md`. If this document and `CLAUDE.md` disagree, `CLAUDE.md` wins; update this file to match. |
| **Background reading** | `docs/guides/phase3_quality_guide.html` |
| **Previous phases** | `docs/phase1_development_plan.md`, `docs/phase2_development_plan.md` |

---

## 1. Goal

Measure how well the assistant finds the right evidence and answers faithfully, use those numbers
to choose the configuration (embedding model, chunk budgets, fusion weights, reranker, LLM,
prompts), and protect that quality with a regression gate.

### In scope

- A golden dataset labelled by evidence, not by chunk id.
- Retrieval metrics per stage (dense, keyword, fused, reranked) and per candidate.
- Routing and structured-answer metrics.
- Generation metrics: citation validity, faithfulness, relevance, correctness, refusals.
- An LLM judge calibrated against human labels.
- Experiments over configuration grids with confidence intervals.
- Fairness (counterfactual) and robustness (injection, noisy parsing) tests.
- System metrics from Phase 2 traces.
- An Eval page in Streamlit and a regression gate (pytest + script).

### Out of scope

- Online A/B testing and production monitoring infrastructure (documented as next steps).
- Paid evaluation services.

---

## 2. Inputs from Phases 1 and 2

| Input | Used for |
|---|---|
| Fixture resumes + ground-truth YAML (Phase 1) | Auto-generated structured questions and expected answers |
| `chunks` table with `chunk_key`, `text` | Resolving evidence quotes to relevant chunks |
| `src/qa/answer.py` with per-stage `Hit` scores | Running retrieval-only and end-to-end evaluations |
| Query traces (`data/traces/*.jsonl`) | Latency, token and failure analysis |
| Config overrides via `RAG_CONFIG` / settings objects | Running experiments without editing the main config |

---

## 3. Scope decisions

| Decision | Choice | Reason |
|---|---|---|
| Relevance labels | Evidence quotes per question, resolved to chunks at eval time | Labels survive re-chunking and budget changes |
| Dataset size | 80–120 questions, split dev (tuning) / test (final report) 60/40 | Enough for coarse decisions; test set stays untouched |
| Question sources | Auto from ground truth (structured), hand-written (depth, compare, JD), LLM-drafted then reviewed, negatives, adversarial | Covers every intent and failure type |
| Retrieval metrics | Recall@k, MRR, nDCG@10, candidate recall@5 | Standard, explainable, cheap |
| Generation metrics | Deterministic citation checks + LLM judge (rubric) + reference correctness | Deterministic first; judge only where needed |
| Judge | Different model family from the generator (e.g. `llama3.1:8b` judging `qwen2.5`), temperature 0, rubric with examples | Reduces self-preference bias |
| Judge trust | Calibrated on 30 human-labelled answers; Cohen's kappa ≥ 0.6 required | A judge that disagrees with humans is not used |
| Uncertainty | Bootstrap 95% CIs; paired comparison on the same questions | Small sets are noisy; avoid chasing noise |
| Tracking | JSONL results + Markdown report per run; MLflow optional | No extra service on the laptop |
| Gate | Fast eval with fakes in pytest; full eval script with thresholds and baseline | CI-friendly; real-model eval stays local |

---

## 4. Architecture

```
eval/
  golden/questions.yaml      questions, intents, evidence quotes, expected candidates, reference answers
  golden/human_labels.yaml   30 answers labelled by a human (judge calibration)
  thresholds.yaml            gate thresholds and allowed regression
  baselines/<name>.json      metrics of the accepted configuration
  experiments/*.yaml         config grids
src/evaluation/
  dataset.py        load + validate golden set; resolve evidence quotes to chunk ids
  generate_qs.py    auto-generate structured questions from fixture ground truth
  retrieval_eval.py run retrieval stages only; per-stage metrics
  routing_eval.py   intent accuracy, filter set precision/recall
  generation_eval.py citation checks, judge calls, correctness, refusals
  judge.py          rubric prompts, JSON scores, calibration (kappa)
  metrics.py        recall@k, precision@k, MRR, nDCG, bootstrap CI, paired difference
  fairness.py       counterfactual variants and rank-stability checks
  robustness.py     injection and noisy-parse suites
  system_eval.py    latency/token statistics from traces
  runner.py         run a config (or grid) → results JSONL + report
  report.py         Markdown report and comparison tables
scripts/
  run_eval.py       CLI: --config, --grid, --split dev|test, --gate
app/pages/6_Eval.py runs list, metrics per stage, per-question drill-down, run comparison
data/eval_runs/     results (gitignored); reports copied to docs/eval/ when accepted
```

### Golden question format

```yaml
- id: q017
  split: dev
  intent: experience_depth
  question: "Who has led a migration to Kubernetes, and where?"
  expected_candidates: [single_col_backend]      # fixture names, resolved to ids
  evidence:                                       # quotes that must appear in a relevant chunk
    - candidate: single_col_backend
      quote: "Led migration of 40 services to Kubernetes"
      grade: 2                                    # 2 = answers it, 1 = partially relevant
  reference_answer: "Priya Raman led the migration of 40 services to Kubernetes on EKS at Northwind Payments."
  answerable: true
```

---

## 5. Milestones

Build in order; each ends with tests passing (section 8.3).

| # | Milestone | Deliverables | Done when |
|---|---|---|---|
| Q0 | Dataset model | Golden YAML schema (Pydantic), loader, validation, evidence → chunk resolution with fuzzy fallback, dev/test split check | Every evidence quote resolves to ≥ 1 chunk on the fixture index |
| Q1 | Golden set | Auto-generated structured questions; hand-written depth/compare/JD questions; reviewed LLM drafts; ≥ 10 negatives; ≥ 5 adversarial; 30 human-labelled answers | 80+ questions, every intent ≥ 8 questions |
| Q2 | Retrieval eval | Per-stage metrics (dense, keyword, fused, reranked) and candidate recall; per-question table | Report shows each stage's contribution |
| Q3 | Routing eval | Intent accuracy + confusion matrix; filter set precision/recall; structured answer exactness | Numbers in report |
| Q4 | Generation eval | Citation validity/precision/recall (deterministic); judge rubric for faithfulness, relevance, completeness; reference correctness; refusal correctness; judge calibration (kappa) | Judge kappa ≥ 0.6 or judge metric marked untrusted |
| Q5 | Experiments | Grid runner over config overrides; bootstrap CIs; paired comparisons; best-config selection on dev, confirmation on test | Chosen config documented with CIs |
| Q6 | Fairness + robustness | Name/gender/nationality counterfactual swaps; injection suite; noisy-parse variants | Rank stability and injection resistance reported |
| Q7 | System metrics | Latency p50/p95 per stage, time to first token, tokens per answer, index build time, memory | Numbers in report |
| Q8 | Eval page + gate | `6_Eval.py`; `scripts/run_eval.py --gate`; pytest smoke eval with fakes; baseline file | Gate fails on a planted regression |
| Q9 | Testing phase + exit | Final test-split report, accepted config committed, baseline saved, next steps | Exit criteria in section 9 met |

---

## 6. Metrics and initial targets

Targets are starting points, revised once the first baseline exists. Reported with 95% CIs.

| Group | Metric | Target |
|---|---|---|
| Retrieval | Recall@10 (after rerank) | ≥ 0.90 |
| Retrieval | MRR | ≥ 0.70 |
| Retrieval | nDCG@10 | ≥ 0.75 |
| Retrieval | Candidate recall@5 | ≥ 0.90 |
| Routing | Intent accuracy | ≥ 0.90 |
| Routing | Structured filter exactness | 1.00 |
| Generation | Citation validity (cited keys exist in context) | 1.00 |
| Generation | Citation precision (cited chunk supports the sentence, judged) | ≥ 0.85 |
| Generation | Faithfulness (judge) | ≥ 0.90 |
| Generation | Answer relevance (judge) | ≥ 0.85 |
| Generation | Refusal correctness on unanswerable questions | ≥ 0.90 |
| Fairness | Top-5 rank change under name/gender swaps | 0 candidates change in ≥ 95% of cases |
| Robustness | Injection resistance (ranking unchanged, warning raised) | 100% |
| System | Retrieval p95 without LLM | < 1 s |
| System | Time to first token p50 | recorded; target < 10 s on the laptop |

---

## 7. Experiments (initial grid)

| Factor | Values |
|---|---|
| Embedding model | bge-small-en-v1.5; nomic-embed-text (Ollama) |
| Chunk budget p_high | 80, 90, 95 |
| Fusion weights (dense:keyword) | 1:1, 1:0.5, 0.5:1 |
| Reranker | off, MiniLM-L6, bge-reranker-base |
| k after rerank | 8, 12, 20 |
| LLM | qwen2.5:3b, qwen2.5:7b |
| Prompt version | v1, v2 (after error analysis) |

Run retrieval factors first (cheap, no LLM), fix the best retrieval config, then run
generation factors. One factor at a time around the current best rather than the full product.

---

## 8. Testing phase

### 8.1 Test data

- Phase 1 fixtures and ground truth; Phase 2 fixture JD and query table.
- Golden set (Q1) and human labels (30 answers).
- Counterfactual variants generated from fixtures (Q6); injection fixtures from Phase 2.

### 8.2 Test levels

| Level | What | Runs |
|---|---|---|
| Unit | Every metric against hand-computed examples; bootstrap determinism with a seed; kappa; evidence resolution | Every change |
| Component | Retrieval eval over the fixture index with fakes; judge parsing with `FakeLLM` | Every change |
| Integration | `run_eval.py` end to end on 10 questions with fakes; report generated | Before closing each milestone |
| Gate | Planted regression (worse fusion weights) must fail the gate | Q8, Q9 |
| Real-model | Full dev run with Ollama and real models | Marked `ollama`, `models`; Q5, Q9 |
| UI smoke | Eval page lists runs and shows a drill-down | Q8 |

### 8.3 Required tests per milestone

| Milestone | Must-have tests |
|---|---|
| Q0 | Invalid YAML rejected with a clear error; quotes resolve to chunks; a quote matching no chunk is reported; dev and test do not share questions |
| Q1 | Every intent has ≥ 8 questions; negatives have `answerable: false`; generated structured questions match ground truth |
| Q2 | Recall@k, MRR, nDCG equal hand-computed values on a toy ranking; graded relevance used by nDCG; per-stage numbers computed from the same run |
| Q3 | Confusion matrix sums to question count; filter precision/recall on toy sets |
| Q4 | Invalid citation detected; judge JSON parsed and range-checked; kappa matches a known example; refusal scoring for answerable vs unanswerable |
| Q5 | Bootstrap CI contains the point estimate and is reproducible with a seed; paired difference sign correct on a toy case; best config chosen on dev only |
| Q6 | Counterfactual swap changes only the targeted attribute; injection suite detects all planted lines |
| Q7 | Percentiles computed from trace timings; missing stages handled |
| Q8 | Gate passes on baseline, fails on planted regression, prints which metric failed |

### 8.4 Manual QA

- [ ] Read 20 judged answers and agree with the judge on at least 16.
- [ ] Review the 10 worst questions in the error analysis; each failure has a stage assigned.
- [ ] Confirm the chosen config improves on the baseline on the **test** split.
- [ ] Eval page drill-down shows question, hits per stage, answer, judge reasons.

---

## 9. Phase 3 exit criteria

1. Milestones Q0–Q9 done; their tests pass.
2. Golden set of 80+ questions committed (fictional data only).
3. Final test-split report committed to `docs/eval/` with CIs.
4. Chosen configuration committed to `configs/config.yaml`; baseline saved; gate passing.
5. Fairness and robustness results reported, with any failures documented as known issues.
6. `CLAUDE.md` updated with final status and production next steps.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Golden set too small or too easy | Misleading metrics | Cover every intent; negatives and adversarial cases; CIs; grow from traces |
| Overfitting to the dev split | Optimistic results | Test split touched once, for the final report |
| Small local judge unreliable | Wrong generation scores | Calibrate with kappa; deterministic checks first; mark judge metrics untrusted below threshold |
| Labels break when chunks change | Eval noise | Evidence quotes, resolved to chunks at eval time |
| Real-model eval slow on laptop | Few experiments | Retrieval-only experiments first; cache LLM outputs by (prompt hash, model) |
| Synthetic resumes unlike real ones | Over-optimistic | Add consented real resumes locally (never committed) for a final sanity run |
