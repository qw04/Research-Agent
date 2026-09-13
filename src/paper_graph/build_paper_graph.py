"""
Offline pipeline: extract a structured profile per paper (techniques used,
novelty claim, baseline methods compared against, headline result) and
build a graph over the corpus from those profiles.

Two edge types come out of this, per the "method-lineage + technique/concept"
scope decision:
  - paper --USES--> technique          (what the paper's method is built from)
  - paper --COMPARES_AGAINST--> X      (X is either another paper in this
                                         corpus, if the stated baseline matches
                                         that paper's own technique, or a
                                         generic technique node otherwise)

This is a one-time offline pass (not run per-query) — src/graph.py's
paper_relationships() tool reads the resulting data/paper_graph.json rather
than re-extracting anything live, so querying it at chat time is cheap and
deterministic.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

CHUNKS_FILE = Path("data/chunks.json")
OUTPUT_FILE = Path("data/paper_graph.json")
MODEL = "gpt-4o-mini"

# Seeding the prompt with known techniques from this corpus encourages the
# LLM to reuse the same canonical name across papers instead of inventing
# near-duplicate variants (e.g. "LSTM" vs "Long Short-Term Memory network"),
# which would otherwise silently break the edge-matching step below.
KNOWN_TECHNIQUES = [
    "SVM", "LSTM", "CNN", "U-Net", "Extreme Learning Machine", "Random Forest",
    "Temporal Convolutional Network", "Sparse Autoencoder", "NARX",
    "Kalman Filter", "Unscented Kalman Filter", "Gravitational Search Algorithm",
    "Barnacle Mating Optimizer", "Transfer Learning", "Multiple Linear Regression",
    "BPNN", "RBFNN", "Coulomb Counting",
]

llm = ChatOpenAI(model=MODEL, temperature=0)


class PaperProfile(BaseModel):
    techniques: list[str] = Field(
        description="Canonical names of the core ML/optimization techniques this "
                     "paper's method is built from. Reuse one of these canonical "
                     f"names whenever applicable: {', '.join(KNOWN_TECHNIQUES)}. "
                     "Otherwise use a short, consistent canonical name of your own."
    )
    novelty: str = Field(
        description="One to two sentences describing what this paper claims is "
                     "new or different about its approach compared to prior work, "
                     "based only on the excerpts given."
    )
    baselines: list[str] = Field(
        description="Canonical names of prior/comparison methods this paper "
                     "explicitly benchmarks its results against in the excerpts "
                     "(e.g. ['BPNN', 'RBFNN']). Use the same canonical-name "
                     "convention as techniques. Empty list if none are mentioned."
    )
    key_result: str = Field(
        description="One sentence summarizing the paper's headline quantitative "
                     "result (e.g. the RMSE/MAE it achieves), grounded only in "
                     "the excerpts given."
    )


profile_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract a structured profile of a research paper's method from "
     "excerpts of that paper. Base every field only on the excerpts provided "
     "— do not use outside knowledge of the paper or invent numbers."),
    ("human", "Paper title: {title}\n\nExcerpts:\n{excerpts}\n\nExtract the profile."),
])
profile_chain = profile_prompt | llm.with_structured_output(PaperProfile)


def gather_excerpts(texts: list[str], max_excerpts: int = 8) -> str:
    keyword_re = re.compile(
        r"RMSE|MAE|MAPE|R2|R-square|accuracy|compar|baseline|than|outperform|versus|propose|novel",
        re.I,
    )
    picked = [texts[0]]  # first chunk is usually title/abstract
    for t in texts[1:]:
        if len(picked) >= max_excerpts:
            break
        if keyword_re.search(t):
            picked.append(t)
    return "\n\n---\n\n".join(p[:700] for p in picked)


# The LLM doesn't always use the same acronym-vs-full-name form for a
# technique when it appears as another paper's baseline (e.g. "SVM" as a
# technique vs "Support Vector Machine" as someone else's baseline) — expand
# known acronyms to a common form before matching so those still link up.
ALIASES = {
    "svm": "support vector machine",
    "rf": "random forest",
    "cnn": "convolutional neural network",
    "kf": "kalman filter",
    "ekf": "extended kalman filter",
    "ukf": "unscented kalman filter",
    "lstm": "long short term memory",
    "elm": "extreme learning machine",
    "gsa": "gravitational search algorithm",
    "bpnn": "back propagation neural network",
    "rbfnn": "radial basis function neural network",
    "tcn": "temporal convolutional network",
    "pf": "particle filter",
    "ocv": "open circuit voltage",
    "ann": "artificial neural network",
    "anns": "artificial neural network",
}


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def canonicalize(s: str) -> str:
    return ALIASES.get(normalize(s), normalize(s))


def names_match(a: str, b: str) -> bool:
    ca, cb = canonicalize(a), canonicalize(b)
    if not ca or not cb:
        return False
    return ca == cb or ca in cb or cb in ca


def build_profiles() -> dict[str, dict]:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c["text"])

    profiles = {}
    for source, texts in by_source.items():
        excerpts = gather_excerpts(texts)
        profile: PaperProfile = profile_chain.invoke({"title": source, "excerpts": excerpts})
        profiles[source] = profile.model_dump()
        print(f"  {source[:70]}... techniques={profile.techniques} baselines={profile.baselines}")
    return profiles


def build_graph(profiles: dict[str, dict]) -> dict:
    papers = list(profiles.keys())
    edges = []
    technique_nodes: list[str] = []  # canonical surface forms, first-seen wins

    def resolve_technique(name: str) -> str:
        for existing in technique_nodes:
            if names_match(name, existing):
                return existing
        technique_nodes.append(name)
        return name

    # Register every paper's own technique names as canonical nodes first, so
    # that baseline mentions of the same technique (possibly phrased
    # differently, e.g. "RBFNN" vs "Radial Basis Function Neural Networks")
    # merge into that canonical node instead of creating a near-duplicate.
    for profile in profiles.values():
        for tech in profile["techniques"]:
            resolve_technique(tech)

    for paper, profile in profiles.items():
        for tech in profile["techniques"]:
            canonical = resolve_technique(tech)
            edges.append({"from": paper, "to": canonical, "type": "uses"})

        for baseline in profile["baselines"]:
            # does this baseline match another paper's own technique set?
            matched_papers = [
                other for other in papers
                if other != paper
                and any(names_match(baseline, t) for t in profiles[other]["techniques"])
            ]
            if matched_papers:
                for other in matched_papers:
                    edges.append({"from": paper, "to": other, "type": "compares_against_paper"})
            else:
                canonical = resolve_technique(baseline)
                edges.append({"from": paper, "to": canonical, "type": "compares_against_technique"})

    return {
        "papers": profiles,
        "technique_nodes": sorted(technique_nodes),
        "edges": edges,
    }


if __name__ == "__main__":
    print("Extracting per-paper profiles...")
    profiles = build_profiles()

    print("\nBuilding graph...")
    graph = build_graph(profiles)

    OUTPUT_FILE.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_FILE} — {len(graph['papers'])} papers, "
          f"{len(graph['technique_nodes'])} technique nodes, {len(graph['edges'])} edges")
