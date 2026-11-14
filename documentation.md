# Evaluation changes: what changed and why

This document records how the evaluation in [src/eval/evaluate.py](src/eval/evaluate.py) changed in November 2026, what motivated each change, and what the changes turned up. Current numbers are in the README's [Evaluation](README.md#evaluation) section.

## Current state (2026-11-14)

**The eval**
- 22 questions × 3 attempts × 3 configs, scored by judge v2 (gpt-4o-mini, temperature 0; sees the retrieved excerpts).
- Metrics: pass^1, ≥1 of 3 and pass^3 (section 1).
- Code: split into modules under `src/eval/` (section 8).

**Checks on the judge**
- Robustness test: 3/3 as expected (section 3).
- Human validation: 4/6 exact, 6/6 within ±1, on 3 tuned items (sections 2, 4).
- Per-question audit: 14/19 auditable verdicts trustworthy. Order effects are the main problem (section 6).

**What the results support**
- top_k=5 beats top_k=3.
- The critic is **not** clearly more or less consistent than running without it (section 7 revises section 5).
- No result has uncertainty intervals yet.

**Next steps:** a plan to add intervals, a paraphrase audit condition, aggregated judging, fresh human grades and an optional second judge has been drafted and is waiting for approval (Open items).

## Starting point

The original eval ran each of the 22 questions in [data/eval_set.json](data/eval_set.json) **once** per configuration. A GPT-4o-mini judge scored faithfulness and relevance from 1 to 5. Three configurations were logged to MLflow:

| Config (judge v1, single attempt) | Avg faithfulness |
|---|---|
| top_k=3, critic off | 3.32 |
| top_k=5, critic off | 3.86 |
| top_k=5, critic on | 4.18 |

Taken at face value, this said both larger top_k and the critic improve faithfulness. It had three blind spots:

1. **Averages hide inconsistency.** One run per question can't tell an agent that is always right from one that is right some of the time.
2. **The judge had never been checked against a human.** Nothing showed its scores meant what they claimed.
3. **Nobody had checked whether the judge could be gamed.** An answer that says nothing could, in principle, score well.

The changes below address these in order. Each one landed as its own commit, citing the paper that motivated it.

---

## 1. pass^k: measuring consistency

**Commit:** `25f84c1`

**Motivation:** τ-bench ([Yao et al., 2024, arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) showed that single-attempt success badly overstates reliability. GPT-4o solved about half its tasks on one attempt, but under 25% when it had to succeed on all 8 attempts.

**What changed**
- Each question now runs 3 times per configuration.
- An attempt is a success only if it scores 5/5 on both faithfulness and relevance.
- Three rates are logged to MLflow, alongside the existing averages:
  - `pass1_rate`: mean per-attempt success (τ-bench's pass^1).
  - `pass_any3_rate`: the question succeeded at least once in 3 attempts.
  - `pass3_rate`: the question succeeded on all 3 attempts (pass^3).
- Runs are suffixed `_x3`, so the old single-attempt runs stay intact for reference.

**Naming note:** "succeeded at least once in k attempts" is usually called **pass@k**, not pass^1. Both are logged, under separate names, so they aren't confused. pass@3 is always ≥ pass^1 ≥ pass^3, and the gap between pass@3 and pass^3 is the part of the success rate that depends on luck.

## 2. Validating the judge against human grades

**Commits:** `c6fcf2f` (infrastructure), `33f6d6a` (human grades)

**Motivation:**
- The ABC checklist ([Zhu et al., NeurIPS 2025, arXiv:2507.02825](https://arxiv.org/abs/2507.02825)) found outcome-validity flaws, such as graders accepting wrong answers, in 7 of 10 popular agent benchmarks.
- Mind2Web 2 (Gou et al., NeurIPS 2025) reached 99% judge accuracy by validating its judge against human labels.

**What changed**
- [data/judge_validation.json](data/judge_validation.json) holds a 3-question holdout, one question per category:
  - q03 (method)
  - q06 (result)
  - q21 (comparison)
- For each question it stores:
  - the agent's answer (top_k=5, critic on)
  - the retrieved excerpts the answer was built from
  - the judge's scores
  - **human scores from the repo owner**
- Every eval run re-judges the stored answers with the current judge, compares them with the human scores, prints the agreement, and attaches it to each MLflow run as the run note.
- `--build-validation` regenerates the answers. It refuses to overwrite a file that already contains human grades.

**On the human scores:** the scores in this file come from a person, not from an LLM. During this work Claude also rated the three answers, but those ratings were deliberately kept out of the file. Filling it with model ratings would reduce "judge vs. human" to "LLM vs. LLM".

**First result (judge v1): 4/6 scores matched exactly, 5/6 within ±1.** The meaningful disagreement was q21:
- The answer named the wrong paper, but every claim in it was supported by the retrieved text.
- The human gave faithfulness 4. The judge gave 2, treating "wrong" as "fabricated".

## 3. Judge robustness against degenerate answers

**Commit:** `daaac72`

**Motivation:**
- Zheng et al., *Cheating Automatic LLM Benchmarks* ([ICLR 2025, arXiv:2410.07137](https://arxiv.org/abs/2410.07137)): a "null model" that gives the same response to every prompt topped AlpacaEval 2.0 with an 86.5% win rate.
- Kapoor et al., HAL (ICLR 2026): agents exploited benchmarks, for example by looking answers up instead of solving the tasks.

**What changed**
Before the main eval, the judge scores three degenerate answers to a seeded random eval question:

| Case | Expected |
|---|---|
| Empty answer | faithfulness and relevance ≤ 2 |
| One-word "unknown" | faithfulness and relevance ≤ 2 |
| "The optimal method is XGBoost with a learning rate of 0.001." (nothing like it is in the corpus) | faithfulness ≤ 2 |

Any score above 3 prints `Judge robustness: <case> scored <n>, may not catch gaming`. Results go to [data/judge_robustness.json](data/judge_robustness.json) and into the MLflow run note.

**First result (judge v1): the empty answer scored 5/5 on both dimensions.** The judge's reasoning described the reference answer's content, so it had graded the reference rather than the (non-existent) answer. The one-word and made-up answers were correctly scored 1/1.

## 4. Judge v2: fixing what the checks found

**Commit:** `935b3a6`

Checks 2 and 3 traced back to the same flaw. Judge v1 was given only the **titles** of the retrieved papers, never their text. So it could not:
- check whether a claim was actually supported (hence q21), or
- tell a real answer apart from a restatement of the reference (hence the empty answer scoring 5/5).

**What changed (`JUDGE_VERSION = "v2-context"`)**
- The judge sees the retrieved excerpts: the same evidence the agent had.
- The two dimensions are now defined separately:
  - **Faithfulness** is judged *only* against the excerpts. A wrong-but-supported answer is still faithful, and honestly saying "the excerpts don't cover this" is faithful too.
  - **Correctness** against the reference counts toward relevance.
- Empty or whitespace-only answers get 1/1 by rule, without calling the judge.
- All judging goes through a single function, `score_answer()`, so the eval, the validation and the robustness test all use exactly the same path.

**Result**
- **Robustness:** 3/3 as expected. Without the rule, the v2 judge alone also scores an empty answer 1/1.
- **Human agreement:** still 4/6 exact, but now 6/6 within ±1 (up from 5/6). q21 faithfulness moved from 2 to 3, against the human's 4.

**Caveats**
- The v2 prompt was written *after* seeing the three holdout items, so agreement on them is optimistic. A fair test needs freshly graded questions the prompt wasn't tuned on.
- Three questions (6 scores) is a spot-check. It can't support a judge-accuracy percentage.

## 5. Full re-run and what changed in the conclusions

**Commit:** `4b7dd94` (22 questions × 3 configs × 3 attempts, judge v2, about 20 minutes)

| Config | pass^1 | ≥1 of 3 | pass^3 | Avg faith | Avg rel |
|---|---|---|---|---|---|
| top_k=3, critic off | 44% | 45% | 41% | 4.18 | 3.74 |
| top_k=5, critic off | 55% | 55% | 55% | 4.39 | 3.95 |
| top_k=5, critic on | 59% | 68% | 50% | 4.35 | 4.14 |

**Conclusions that held**
- top_k=5 is better than top_k=3.

**Conclusions that changed**
- **"The critic improves faithfulness" no longer holds.** Under judge v1 it looked like 3.86 → 4.18. Under v2, faithfulness is flat (4.39 vs 4.35) and the critic's benefit shows up in relevance (3.95 → 4.14). The v1 gain was most likely an artefact of a judge that couldn't see the evidence.
- **The critic trades consistency for peak quality.** It gives the best pass^1 and best-of-3, but a lower pass^3 than running without it. Everything else runs at temperature 0, and without the critic pass^1 equals pass^3 almost exactly. The critic is the main source of run-to-run variation. Single-run evaluation would never have shown this.
  - *Revised in section 7: this conclusion did not survive closer inspection.*

**New findings**
- **Questions that fail in every config:**
  - q03: "LRLSTM" never matches local retrieval, so the agent falls back to arXiv.
  - q21, q22: cross-paper comparison questions.
  - q07, q08, q12, q14.

  These are retrieval problems and are the clearest targets for improving the agent itself.

**Comparability:** v1 and v2 scores can't be compared directly. v2 runs log a `judge_version` parameter in MLflow. Runs without it, including the old single-attempt runs with no `_x3` suffix, used v1.

## 6. Judge audit: per-item verdict reliability

**Motivation:** Kumar, Rathore & Moitra, *All Verdicts are Not Equal: Rethinking LLM Judge Reliability* ([arXiv:2610.12083](https://arxiv.org/abs/2610.12083), AACL-IJCNLP 2026) report three things:
- judge verdicts can change across identical reruns, even at temperature 0;
- presentation order can flip verdicts;
- reliability is item-specific, not a property of the judge model.

They propose a **trustworthy verdict rate**: the share of verdicts that are reproducible, order-invariant and accurate. pass^k repeats the agent but judges each answer only once, so it couldn't detect any of this.

**What changed (`--judge-audit`)**

Judge prompt v2 itself was **not** changed. The audit takes attempt 1 of each question from `topk5_critic_on_x3` and doesn't re-run the agent.

**Judgements per answer**

| Condition | What it tests |
|---|---|
| 5× normal prompt | reproducibility |
| 5× answer-first variant | order invariance |
| 5× on the human-graded answer (holdout items only) | accuracy |

- **Answer-first variant:** built by reordering the live prompt's own sections, so the wording is guaranteed identical; an assertion fails if the prompt layout changes.
- **Accuracy:** checked on the answer the human actually graded. For q03 and q21 that is a different answer text from the eval's attempt 1.

**Trustworthy** means:
- all normal verdicts agree,
- all answer-first verdicts agree with them,
- and, for holdout items, every judgement is within ±1 of the human score.

**Results**

| Measure | Result |
|---|---|
| Trustworthy verdict rate | 14/19 auditable (74%); 14/22 (64%) counting unverifiable as untrusted |
| Order invariance | 74%. The answer-first order flipped verdicts on q11, q14, q16, q18 and q20. The biggest swings were q16, faithfulness 5 → 1 (relevance 5 → 2), and q14, faithfulness 2 → 5. |
| Reproducibility | 100% in the first audit run, 95% in the second. In the second, q12 faithfulness split 2/3 and q16 relevance split 4/5 across identical calls. Both MLflow runs are kept. |
| Cross-session drift | q05, q12 and q16 scored differently from the score the same judge gave the same input during the eval run a few hours earlier. q05 went from success to fail. |
| Accuracy | q06 and q21 stay within ±1 of the human scores on every run. q03 couldn't be audited (below). |
| pass^1 / pass^3 (critic on) | 59% / 50% on all 22, vs 60% / 57% on the 14 trustworthy questions |

The audit confirms all three of the paper's claims on this eval:
- Order is the biggest effect.
- Temperature 0 does not guarantee identical verdicts, especially across sessions.
- The untrustworthy verdicts are concentrated on specific questions rather than spread evenly.

Restricting to trustworthy questions slightly narrows the gap between pass^1 and pass^3. Part of the critic's apparent inconsistency may therefore be judge noise rather than agent behaviour. With 14 questions, that difference is not conclusive.

**Limitations**
- **3 questions couldn't be audited** (q03, q04, q17). They went through the arXiv fallback, and the eval log didn't store the retrieved excerpts, so their context can't be rebuilt. For the other 19 the context is rebuilt exactly, because the agent retrieved once with the original question and FAISS retrieval is deterministic.
- **Eval logs now store each attempt's retrieved excerpts**, so a future audit after a fresh eval run covers all 22.
- **5 repetitions per condition is small.** The reproducibility numbers in particular moved between the two audit runs.

**Not done in this change:** fixing the order sensitivity. Possible fixes are judging in both orders and requiring agreement, or averaging across orders. Either would change judge behaviour and needs its own validation.

## 7. Reassessing "the critic is the main source of variation"

**Commit:** documentation only, in the same commit as section 8. No API calls: the numbers come from the stored logs in `data/eval_runs/*_x3.json`.

**Motivation:** the section 5 conclusion rested on an assumption: with the critic off, the temperature-0 pipeline gives nearly the same answer every time, so any pass^1-to-pass^3 gap must come from the critic. The judge audit (section 6) showed the judge isn't fully deterministic either. That made it worth checking the assumption directly, by comparing the 3 stored answer texts per question.

**Result**

| Config | Questions where all 3 attempts had identical text | Questions whose verdict varied across attempts |
|---|---|---|
| top_k=3, critic off | 4/22 | q10 (text differed) |
| top_k=5, critic off | 12/22 | none |
| top_k=5, critic on | 5/22 | q05 (identical text); q03, q16, q20 (text differed) |

- **The assumption is false.** With the critic off, the answer text still changes across attempts on 10 to 18 of 22 questions. The verdicts happened to stay stable, but the agent is not near-deterministic.
- **None of the 4 questions behind the critic-on gap is a clean case of the agent varying:**
  - q05: identical text, varying verdict, so this is pure judge variation.
  - q16, q20: both untrusted in the audit, because answer-first order flips their verdicts.
  - q03: arXiv fallback, unauditable.
- 50% vs 55% pass^3 is a one-question difference (11 vs 12 questions) with no interval.

**Conclusion change:** "the critic trades consistency for peak quality" is withdrawn. The current statement is that the critic is not clearly more or less consistent than running without it. Separating agent from judge variance properly needs aggregated judging on stored answers (see Open items).

**Caveat:** identical text with differing verdicts proves the judge varied. Differing text with differing verdicts doesn't prove the agent varied, because the judge can vary there too.

## 8. Eval code split into modules

**Commit:** see git log, "Split eval code into modules". It's a refactor only, so there's no paper. It prepares for the planned changes, which each touch one part of the eval.

**What changed:** the single 700-line `src/eval/evaluate.py` became:

| File | Contents |
|---|---|
| `settings.py` | paths, eval topic, pass^k parameters, success rule; loads `.env` |
| `judge.py` | `JudgeScore`, prompt v2, answer-first variant, `format_judge_context()`, `score_answer()`, `judge_once()` |
| `pass_k.py` | `CONFIGS`, `run_eval()`, `print_summary()` |
| `validation.py` | holdout build and agreement report, `load_graded_items()` |
| `robustness.py` | degenerate-answer test |
| `audit.py` | judge audit |
| `evaluate.py` | CLI only (same commands and flags as before) |

**Validation (no behaviour change)**
- **Static comparison, no API calls.** I compared the new modules with the pre-refactor file. All of these were identical:
  - system prompt, user prompt and answer-first prompt text
  - `JudgeScore` schema
  - judge model, version and temperature
  - topic, attempts and success threshold
  - file paths, `CONFIGS`, validation ids, robustness cases and seed, audit parameters
  - context formatting, the empty-answer rule, and the success rule for all 25 score pairs
- **Live end-to-end run.** I ran `evaluate.py --limit 1` against a throwaway MLflow database:
  - robustness 3/3 and validation 4/6 exact, 6/6 within ±1, the same as before
  - q01 5/5 on all 3 attempts in all 3 configs
  - eval logs now include the retrieved excerpts

  The robustness file regenerated with identical scores; only one free-text `reasoning` sentence was worded differently, a reminder that the judge's output text isn't deterministic either. Test outputs were deleted and the committed data files kept.

## Open items

A plan covering most of these has been drafted and is waiting for approval:
- **Uncertainty:** bootstrap 95% intervals on all metrics, and paired intervals for the critic on/off and top_k comparisons.
- **Fresh eval run** with context logging, so every question can be audited and re-judged exactly.
- **Paraphrase condition in the audit:** same rubric, different wording.
- **Aggregated judge (v3):** majority over several calls, across repeats and/or prompt variants (order, paraphrase). Then re-judge the stored attempts to separate agent variance from judge variance.
- **Human agreement:** add weighted kappa, and grade 3–5 fresh questions by hand, reported separately from the 3 tuned ones.
- **Optional second judge** for cross-judge agreement.

Also open:
- **Stronger robustness cases:** one canned answer for all 22 questions (a true null model), and an answer containing text aimed at the judge.
- **Agent traces:** inspect the arXiv-fallback answers for shortcuts, such as finding the very paper a question is about (HAL's lesson).
- **Retrieval failures:** fix q03 and the comparison questions q21/q22, then re-run the eval.
- **Eval set:** still tied to the battery SOC corpus. A topic-agnostic one is on the README roadmap.
- **Paper figures:** verify the figures quoted from the cited papers against the papers themselves.
