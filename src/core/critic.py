"""
Deliberately scoped to one critic role + one revision pass, downstream of
the LangGraph agent in graph.py — not a full multi-agent overhaul. This is
architecturally distinct from LangGraph's graph-based control flow: it's
role-based multi-agent orchestration (a "second opinion" reviewer), which
is the point of adding it as a second framework rather than just another
LangGraph node.
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

load_dotenv()

# Cheap model: reviewing/revising a single already-generated answer is a
# lighter task than the original generation, so the same cost rationale
# used for graph.py's utility_llm applies here.
CRITIC_MODEL = "gpt-4o-mini"


class Critique(BaseModel):
    approved: bool = Field(
        description="True only if every claim in the draft answer is fully "
                     "supported by the source excerpts and citations are accurate."
    )
    feedback: str = Field(
        description="If not approved, specific, actionable feedback on what is "
                     "unsupported, missing, or miscited. Empty string if approved."
    )


# Used when no topic is supplied, so the critic still works as a generic
# grounding checker rather than assuming any particular domain.
DEFAULT_TOPIC = "academic research"


def _make_critic_agent(topic: str) -> Agent:
    # Built per call (not at module level) because the backstory embeds the
    # user-provided research topic, which varies between runs.
    return Agent(
        role="Fact-Checking Critic",
        goal=(
            "Verify that a draft answer is fully supported by the retrieved source "
            "excerpts, and flag any unsupported, fabricated, or miscited claims."
        ),
        backstory=(
            f"You are a meticulous research reviewer for a research assistant "
            f"focused on {topic}. You check draft answers claim by claim against "
            "the source excerpts they cite, and you never let an unsupported "
            "claim pass."
        ),
        llm=CRITIC_MODEL,
        verbose=False,
    )


def _make_reviser_agent(topic: str) -> Agent:
    return Agent(
        role="Answer Reviser",
        goal=(
            "Revise a draft answer to address specific critic feedback while staying "
            "strictly grounded in the provided source excerpts."
        ),
        backstory=(
            f"You rewrite answers from a research assistant focused on {topic} "
            "to fix issues a critic identified, without introducing new claims "
            "that aren't backed by the source excerpts."
        ),
        llm=CRITIC_MODEL,
        verbose=False,
    )


def _critique_task(agent: Agent, question: str, answer: str, context: str) -> Task:
    return Task(
        description=(
            f"Question: {question}\n\nDraft answer:\n{answer}\n\n"
            f"Source excerpts the answer must be grounded in:\n{context}\n\n"
            "Check every claim in the draft answer against the source excerpts. "
            "Approve only if every claim is supported and citations are accurate."
        ),
        expected_output="A structured critique with an approval decision and feedback.",
        agent=agent,
        output_pydantic=Critique,
    )


def _revision_task(agent: Agent, question: str, answer: str, context: str, feedback: str) -> Task:
    return Task(
        description=(
            f"Question: {question}\n\nOriginal draft answer:\n{answer}\n\n"
            f"Critic feedback:\n{feedback}\n\n"
            f"Source excerpts:\n{context}\n\n"
            "Rewrite the answer to address the feedback, staying strictly grounded "
            "in the source excerpts and preserving the [Source: <title>] citation "
            "format used in the original answer."
        ),
        expected_output="A revised, fully grounded answer with citations.",
        agent=agent,
    )


def review_answer(
    question: str,
    answer: str,
    context: str,
    topic: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run the critic agent against a draft answer. If approved, return it
    unchanged. If not, run exactly one revision pass with the reviser agent
    and return the revised answer — no further critique loop, per the v2
    plan's "one critic role, one revision loop" scope.

    `topic` is the research topic the user supplied up front; it frames both
    agents' personas. Falls back to DEFAULT_TOPIC when not given."""
    topic = topic or DEFAULT_TOPIC
    critic_agent = _make_critic_agent(topic)

    critique_crew = Crew(
        agents=[critic_agent],
        tasks=[_critique_task(critic_agent, question, answer, context)],
        process=Process.sequential,
        verbose=verbose,
    )
    critique: Critique = critique_crew.kickoff().pydantic

    if verbose:
        print(f"[critic] approved={critique.approved} feedback={critique.feedback!r}")

    if critique.approved:
        return {"answer": answer, "approved": True, "feedback": critique.feedback, "revised": False}

    reviser_agent = _make_reviser_agent(topic)
    revision_crew = Crew(
        agents=[reviser_agent],
        tasks=[_revision_task(reviser_agent, question, answer, context, critique.feedback)],
        process=Process.sequential,
        verbose=verbose,
    )
    revised_answer = str(revision_crew.kickoff())

    return {"answer": revised_answer, "approved": False, "feedback": critique.feedback, "revised": True}


if __name__ == "__main__":
    demo_context = (
        "[Source: Support Vector Machines Used to Estimate the Battery State of Charge]\n"
        "SVM regression was used to estimate SOC with an RMSE of 2.3% on the test set."
    )
    demo_answer = (
        "SVM regression achieves near-perfect SOC estimation with 0% error "
        "[Source: Support Vector Machines Used to Estimate the Battery State of Charge]."
    )
    result = review_answer(
        question="How accurate is SVM regression for SOC estimation?",
        answer=demo_answer,
        context=demo_context,
        topic="battery state-of-charge estimation",
        verbose=True,
    )
    print(result)
