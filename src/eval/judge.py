"""
The LLM judge: scores an answer's faithfulness and relevance (1-5 each).

All judging in the eval goes through score_answer() (or judge_once() for the
audit's prompt variants), so the eval, the human-validation check, the
robustness test and the audit use the same prompt, model and empty-answer rule.

Judge v2 (2026-11-14). v1 only saw the retrieved sources' TITLES, so it could
not check whether a claim was actually backed by the text. Two failures
followed from that: the robustness test showed it scoring an empty answer 5/5
(it graded the reference facts instead of the answer), and the human
validation holdout showed it marking a wrong-but-supported answer as
unfaithful (q21: human 4, judge 2). v2 sees the retrieved excerpts and keeps
the two dimensions separate: faithfulness = supported by the excerpts;
correctness vs the reference = relevance. Scores from v2 are not directly
comparable with the earlier single-attempt runs, which used v1.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from eval.settings import EVAL_TOPIC

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_VERSION = "v2-context"

judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


class JudgeScore(BaseModel):
    faithfulness: int = Field(
        description="1-5: is every claim in the answer backed by the retrieved "
                     "sources, with no fabricated or unsupported claims? "
                     "5 = fully grounded, 1 = mostly fabricated."
    )
    relevance: int = Field(
        description="1-5: does the answer cover the key facts listed in the "
                     "reference, and actually address the question asked? "
                     "5 = fully covers the reference facts, 1 = off-topic or missing them."
    )
    reasoning: str = Field(description="One sentence justifying both scores.")


judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading a RAG system's answer to a question about research "
     "papers on {topic}. You are given the excerpts the system retrieved, "
     "which are the only evidence it had. Score two separate things, 1-5 each:\n"
     "- faithfulness: is every claim in the answer supported by the retrieved "
     "excerpts, with nothing fabricated? Judge this ONLY against the excerpts, "
     "not against the reference facts. An answer that misses the point or picks "
     "the wrong paper, but whose every claim is backed by the excerpts, is still "
     "faithful. A plausible-sounding number or detail that does not appear in "
     "the excerpts is a faithfulness failure even if it is close to correct. "
     "Honestly saying the excerpts don't contain the answer is faithful.\n"
     "- relevance: does the answer actually address the question and cover the "
     "reference facts? Missing or contradicting the reference facts lowers "
     "relevance.\n"
     "Grade only the text in the 'System's answer' section. If it is empty or "
     "says nothing substantive, it covers none of the reference facts."),
    ("human",
     "Question: {question}\n\n"
     "Reference facts a good answer should cover:\n{expected_answer}\n\n"
     "Retrieved excerpts (the system's only evidence):\n{context}\n\n"
     "System's answer:\n<<<\n{answer}\n>>>\n\n"
     "Score faithfulness and relevance."),
])
judge_chain = judge_prompt | judge_llm.with_structured_output(JudgeScore)

# Answer-first variant (used by the audit's order-invariance check), built from
# the live judge prompt's own text so the wording is guaranteed identical and
# only the section order changes.
_human_sections = judge_prompt.messages[1].prompt.template.split("\n\n")
assert [s.split("\n")[0][:10] for s in _human_sections] == [
    "Question: ", "Reference ", "Retrieved ", "System's a", "Score fait",
], "judge prompt layout changed; update the answer-first variant"
judge_prompt_answer_first = ChatPromptTemplate.from_messages([
    ("system", judge_prompt.messages[0].prompt.template),
    ("human", "\n\n".join(_human_sections[i] for i in (0, 3, 1, 2, 4))),
])
judge_chain_answer_first = judge_prompt_answer_first | judge_llm.with_structured_output(JudgeScore)


def format_judge_context(chunks: list[dict]) -> str:
    """chunks: [{"text": ..., "source": optional title}] -> numbered excerpts."""
    if not chunks:
        return "(nothing retrieved)"
    return "\n\n".join(
        f"[Excerpt {i}]" + (f" (from: {c['source']})" if c.get("source") else "") + f"\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def _is_empty(answer: str) -> bool:
    return not answer or not answer.strip()


def score_answer(question: str, expected_answer: str, answer: str, chunks: list[dict]) -> JudgeScore:
    """Single entry point for judging. Empty / whitespace-only answers are scored
    1/1 deterministically without calling the judge (the v1 judge gave them 5/5)."""
    if _is_empty(answer):
        return JudgeScore(faithfulness=1, relevance=1,
                          reasoning="Empty answer: scored 1/1 by rule, judge not called.")
    return judge_chain.invoke({
        "topic": EVAL_TOPIC,
        "question": question,
        "expected_answer": expected_answer,
        "context": format_judge_context(chunks),
        "answer": answer,
    })


def judge_once(chain, question: str, expected: str, answer: str, chunks: list[dict]) -> tuple[int, int]:
    """Judge with an explicit chain (normal or a variant); same empty-answer
    rule as score_answer(). Returns (faithfulness, relevance)."""
    if _is_empty(answer):
        return 1, 1
    s = chain.invoke({"topic": EVAL_TOPIC, "question": question, "expected_answer": expected,
                      "context": format_judge_context(chunks), "answer": answer})
    return s.faithfulness, s.relevance
