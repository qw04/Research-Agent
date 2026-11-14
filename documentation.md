# Evaluation changes: what changed and why

This document records how the evaluation in [src/eval/evaluate.py](src/eval/evaluate.py) changed in November 2026, what motivated each change, and what the changes turned up. Current numbers are in the README's [Evaluation](README.md#evaluation) section.

## Current state (2026-11-14)

**The eval**
- 22 questions × 3 attempts × 3 configs. Two agent runs exist: run 1 (`*_x3`) and run 2 (`*_x3_ctx`, which logs each answer's context).
- Headline numbers: run 2 re-judged with **judge v3-ensemble9**, a majority of 9 calls across the normal, answer-first and paraphrased prompts (section 12).
- Metrics: pass^1, ≥1 of 3 and pass^3 (section 1), each with bootstrap 95% intervals (section 9).
- Code: modules under `src/eval/` (section 8).

**Checks on the judge**
- Robustness test: 3/3 for both v2 and v3 (sections 3, 13).
- Human validation on 3 tuned items: v2 4/6 exact, 6/6 within ±1; v3 3/6 exact, 5/6 within ±1 (section 13).
- Per-question audit on run 2: 17/22 verdicts trustworthy. Order and wording effects are the main problem (section 11).

**What the results support**
- **No config difference in pass rates** is distinguishable from question sampling.
- **The critic lowers faithfulness** in run 2: −0.24 [−0.49, −0.04] with v3. It didn't in run 1.
- **The critic adds run-to-run variation on specific questions** (section 13 revises section 7), but its pass^3 difference interval still touches zero.

**Not done, by decision:** fresh human grades, kappa, and a second judge model.

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
- *Revised again in section 13: with judge noise mostly removed, the critic does add variation on specific questions.*

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

## 9. Uncertainty: bootstrap intervals

**Commit:** `514ffed`

**Motivation:** with 22 questions, one question moves pass^3 by about 4.5 points, so differences between point estimates can't be read as real on their own. Yagubyan, *The Coin Flip Judge?* ([arXiv:2606.13685](https://arxiv.org/abs/2606.13685), preprint) recommends reporting uncertainty for LLM-judged results.

**What changed**
- New `src/eval/stats.py` and `--stats`, with no API calls.
- For each config: a percentile bootstrap over **questions** (10,000 resamples, seed 0) for pass^1, ≥1 of 3, pass^3 and the mean scores.
- A **paired** bootstrap, reusing the same resampled questions, for critic on − off and top_k 5 − 3.
- The identical-text breakdown from section 7, now computed by the code.
- Each attempt now logs its `judge_version`, and `--suffix` names new runs so they don't overwrite old ones.
- Output goes to `data/eval_stats<suffix>.json` and an MLflow `stats<suffix>` run (point estimates, `ci_low`, `ci_high`).

**Result on run 1 (v2):** no paired difference excludes zero.
- top_k 5 − 3, pass^1: +0.106 [+0.000, +0.242]. This is the closest to a real difference.
- Critic on − off, pass^3: −0.045 [−0.182, +0.091].

**Caveat:** the intervals cover only which questions are in the set. They don't include judge noise or the agent's randomness beyond the 3 attempts. Conformal / interval evaluation (Sheng et al., EMNLP 2025) would need a much larger human-labelled calibration set, so it's left as future work.

## 10. Fresh eval run with context logging (run 2)

**Commit:** `cafeeba`

**Motivation:** re-judging stored answers exactly needs the excerpts each answer was built from. Run 1 didn't store them, and the 3 arXiv-fallback questions couldn't be rebuilt.

**What changed:** the same eval (judge v2, same configs) was run again with `--suffix _ctx`. Every attempt stores its retrieved excerpts, and all 22 questions are auditable.

**Result (v2, single call)**

| Config | pass^1 | ≥1 of 3 | pass^3 | Faith | Rel |
|---|---|---|---|---|---|
| top_k=3, critic off | 44% | 50% | 41% | 4.14 | 3.76 |
| top_k=5, critic off | 55% | 55% | 55% | 4.39 | 3.88 |
| top_k=5, critic on | 53% | 59% | 45% | 4.26 | 4.00 |

- **Critic configs moved between runs:** the critic-on pass^1 fell from 59% (run 1) to 53%, and ≥1 of 3 from 68% to 59%. The critic-off configs replicated almost exactly.
- **Only one paired difference excludes zero:** critic on − off, average faithfulness −0.136 [−0.318, −0.015].
- **Robustness and validation** gave the same results as before (3/3; 4/6 exact, 6/6 within ±1).

## 11. Wording sensitivity: a paraphrased-prompt audit condition

**Commit:** `238b7b6`

**Motivation:** Yagubyan ([arXiv:2606.13685](https://arxiv.org/abs/2606.13685)) reports that meaning-preserving paraphrases of the judge prompt changed a quarter of gpt-4o-mini's majority outcomes. The first audit (section 6) only tested section order.

**What changed**
- `judge.py` gains a paraphrased prompt. It keeps the same rubric, rules, section order and output schema (including the field descriptions), but every sentence is reworded. It was written for this project and hasn't been independently reviewed.
- The audit now judges each answer 5 times under each of three conditions: normal, answer-first and paraphrase.
- Trustworthy now also requires **paraphrase invariance**.
- Reported per condition:
  - consistency (all 5 verdicts agree);
  - majority-flip rates against the normal prompt;
  - on the human-graded items, a pointwise adaptation of Kumar et al.'s **T = C × (A − flip/2)**. Here C is agreement with the normal-prompt majority, flip is the answer-first disagreement rate, and A is the share of judgements within ±1 of the human score. The paper defines T for pairwise judging, and it needs human labels, so it's only computed on those 3 items.
- New audits write `data/judge_audit_<run>.json`. The first audit's `data/judge_audit.json` is unchanged.

**Result on run 2, critic on (all 22 questions auditable)**

| Measure | Result |
|---|---|
| Trustworthy | 17/22 (77%) |
| Consistency within a condition | normal 91%, answer-first 100%, paraphrase 100% |
| Majority verdict differs from normal | answer-first 18%, paraphrase 9% |
| Adapted T | 0.87 on 3 items |
| Untrustworthy | q03 and q18 (normal prompt not reproducible; q03 also flips under paraphrase), q11 and q14 (order flips), q16 (order and paraphrase flip) |

**Interpretation:** repeating the same prompt almost always gives the same verdict. Changing its order or wording is what moves verdicts. This shaped the choice of aggregation in section 12.

## 12. Aggregated judge (v3): majority of N

**Commit:** `e90c83f`

**Motivation:**
- Single calls are unreliable per item (sections 6 and 11).
- Yagubyan recommends multi-trial aggregation and position randomisation.
- Peng et al., *RuVerBench* ([arXiv:2606.29920](https://arxiv.org/abs/2606.29920), EMNLP 2026), find majority voting helps rubric judging, with diminishing returns.

**What changed**
- `score_answer_aggregated()` judges each answer N times (N must be odd, so there are no ties).
- **Success** is the majority of the N per-call verdicts. Faithfulness and relevance are reported as means and never thresholded, because requiring a mean of 5.0 would mean every call had to agree.
- **Modes:**
  - `repeat`: the normal prompt N times.
  - `ensemble`: cycles normal, answer-first and paraphrase, so N must be a multiple of 3.
- **CLI:** `--judge-n` and `--judge-mode` apply to the eval, the robustness test and the validation report.
- **Logging:** MLflow gets `judge_n`, `judge_mode` and `judge_version`; per-attempt logs keep every call.
- **Versioning:** N > 1 is versioned `v3-<mode><N>`. N=1 is exactly the v2 judge; I verified this live, and the validation check reproduced 4/6 and 6/6. v3 robustness results go to `data/judge_robustness_<version>.json`, so the v2 file stays as it was.

**Choice of N = 9, ensemble:** within-condition repeats were 91–100% unanimous, so repeating one prompt would barely change anything. Order and wording flipped 9–18% of verdicts, so varying them is what matters. N=3 would give each prompt one vote. N=9 also guards against the normal prompt's own non-reproducibility (q03, q18). At gpt-4o-mini prices it costs well under $1 per full re-judge.

## 13. Separating agent variance from judge variance

**Commit:** `cbd2d4a`

**Motivation:** the README's critic conclusion depended on how much of the gap between pass^1 and pass^3 came from the judge rather than the agent:
- Kumar et al. ([arXiv:2610.12083](https://arxiv.org/abs/2610.12083)) show reliability is item-specific.
- Haldar & Hockenmaier, *Rating Roulette* ([arXiv:2510.27106](https://arxiv.org/abs/2510.27106)), show LLM judges have low intra-rater reliability.

Section 7 had withdrawn the claim on judge-noise grounds without testing it directly.

**What changed:** `src/eval/rejudge.py` and `--rejudge`. Every stored run-2 attempt (198 in total) is re-scored with v3-ensemble9, with no agent calls (1,782 judge calls). The original single-call scores are kept beside the new ones, and intervals are computed on the result (`*_x3_ctx_v3ens9`).

**Result**

| Config | pass^1 | ≥1 of 3 | pass^3 | Faith | Rel | Verdicts changed vs v2 |
|---|---|---|---|---|---|---|
| top_k=3, critic off | 47% [27, 68] | 50% | 45% [23, 68] | 4.30 | 3.75 | 4/66 |
| top_k=5, critic off | 55% [32, 73] | 55% | 55% [32, 73] | 4.50 | 3.96 | 0/66 |
| top_k=5, critic on | 52% [33, 70] | 64% | 41% [23, 64] | 4.26 | 3.98 | 5/66 |

**Paired differences**
- **Critic on − off:**
  - pass^1: −3 pp [−15, +6]
  - pass^3: −14 pp [−27, 0]
  - faithfulness: **−0.24 [−0.49, −0.04]**, which excludes zero
  - relevance: +0.02 [−0.21, +0.28]
- **top_k 5 − 3:** every interval includes zero.

**Where the critic-on gap comes from**
- With the critic **off**, there is no gap between pass^1 and pass^3 (55% / 55%).
- With it **on**, the gap remains after aggregation (52% / 41%). It comes from five questions: q03, q04, q17, q18 and q20.
- All five had differing answer text across attempts, and in every one of those attempts the critic rejected the draft and the reviser rewrote it.
- **q18 and q20 are the clearest cases.** With the critic off, they produced identical text and verdicts all three times. With it on, they produced three different rewrites, and one q20 rewrite got 0 of 9 votes.
- q03, q04 and q17 also go through the arXiv fallback, which varies on its own. Even so, their verdicts were stable with the critic off.

**Judge checks with v3**
- Robustness: 3/3.
- Human validation: **3/6 exact, 5/6 within ±1**, worse than v2's 4/6 and 6/6. The answer-first and paraphrased prompts pull q21's faithfulness down to 2, against the human's 4. Aggregating doesn't fix the judge's disagreement with the human on that item.

**Conclusions that held**
- No pass-rate difference between configs is distinguishable from question sampling.
- top_k=5 ≥ top_k=3 on point estimates in every run.

**Conclusions that changed**
- **"The critic is not clearly more or less consistent" (section 7) is revised.** With judge noise mostly removed, the critic does add run-to-run variation, concentrated on questions where it rejects the draft and the rewrites differ. The pass^3 difference interval [−27, 0] still touches zero, and several attempts are borderline (3–6 of 9 votes). So this is a question-level finding, not a demonstrated config-level difference.
- **The critic lowering faithfulness is new.** It excludes zero in run 2 with both judges (v2 −0.14, v3 −0.24) but didn't in run 1 (−0.05 [−0.23, +0.08]). So it has replicated in one of two agent runs.

**Caveats**
- v3 reduces judge noise but doesn't remove it.
- The human-validation set is 3 tuned items.
- No second judge model has been compared.
- "Run 2" is one more sample of the agent, not a fixed ground truth.

## Open items

Not done, by decision:
- **Human agreement:** weighted kappa, and grading 3–5 fresh questions not used for tuning.
- **Second judge model** for cross-judge agreement.

Also open:
- **Re-run the agent several more times** (or more attempts per question) to test whether the critic's faithfulness drop and its question-level variation replicate.
- **Stronger robustness cases:** one canned answer for all 22 questions (a true null model), and an answer containing text aimed at the judge.
- **Agent traces:** inspect the arXiv-fallback answers for shortcuts, such as finding the very paper a question is about (HAL's lesson).
- **Retrieval failures:** fix q03 and the comparison questions q21/q22, then re-run the eval.
- **Eval set:** still tied to the battery SOC corpus. A topic-agnostic one is on the README roadmap.
- **Paper figures:** verify the figures quoted from the cited papers against the papers themselves.
