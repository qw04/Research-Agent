"""
Shared settings and small helpers for the eval package: file locations, the
eval topic, the pass^k parameters, and the success rule every module uses.

Imported by every other module in src/eval/, so it loads .env first.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths are relative to the repo root (run the eval from there).
EVAL_SET_FILE = Path("data/eval_set.json")
RUNS_DIR = Path("data/eval_runs")
VALIDATION_FILE = Path("data/judge_validation.json")
ROBUSTNESS_FILE = Path("data/judge_robustness.json")
AUDIT_FILE = Path("data/judge_audit.json")

MLFLOW_EXPERIMENT = "research-agent-eval"

# The hand-built eval set (data/eval_set.json) was written against the
# battery SOC corpus, so the eval runs with that topic. Change this together
# with the eval set when evaluating a different corpus.
EVAL_TOPIC = "battery state-of-charge (SOC) estimation"

N_ATTEMPTS = 3   # runs per question per configuration (pass^k with k=3)
PASS_SCORE = 5   # an attempt succeeds iff faithfulness >= this AND relevance >= this


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))


def attempt_succeeded(faithfulness: int, relevance: int) -> bool:
    return faithfulness >= PASS_SCORE and relevance >= PASS_SCORE
