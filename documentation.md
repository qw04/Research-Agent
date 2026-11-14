# Evaluation changes: what changed and why

This document records how the evaluation in [src/eval/evaluate.py](src/eval/evaluate.py) changed in November 2026, what motivated each change, and what the changes turned up. Current numbers are in the README's [Evaluation](README.md#evaluation) section.

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

**New findings**
- **Questions that fail in every config:**
  - q03: "LRLSTM" never matches local retrieval, so the agent falls back to arXiv.
  - q21, q22: cross-paper comparison questions.
  - q07, q08, q12, q14.

  These are retrieval problems and are the clearest targets for improving the agent itself.

**Comparability:** v1 and v2 scores can't be compared directly. v2 runs log a `judge_version` parameter in MLflow. Runs without it, including the old single-attempt runs with no `_x3` suffix, used v1.

## Open items

- Grade 2–3 fresh questions by hand, to test judge v2 on items it wasn't tuned against.
- Fix the retrieval failures above, then re-run the eval to measure the effect.
- Consider a non-zero temperature, or several judge samples, if pass^k should also capture variation in generation and judging, not just in the critic.
- The eval set is still tied to the battery SOC corpus. A topic-agnostic eval set is on the README roadmap.
