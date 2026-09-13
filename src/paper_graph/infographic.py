"""
Standalone infographic designer: extracts every model/baseline result
(model name, dataset or test condition, metric, value) mentioned in a
corpus of research papers, then renders a grid infographic — one panel per
paper, grouped bar chart of model x dataset for that paper's dominant
metric, proposed method highlighted against its baselines.

Deliberately self-contained and domain-agnostic (no battery/SOC-specific
wording baked into the prompts) and CLI-configurable, so it can be pointed
at a different corpus of papers later without editing code — just:

    python src/infographic.py --chunks path/to/chunks.json --output-dir path/to/out

Expects the same chunks.json shape produced by src/ingest.py: a JSON list
of {"source": <paper title>, "text": <chunk text>, ...}.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gpt-4o-mini"
KEYWORD_RE = re.compile(
    r"RMSE|MAE|MAPE|R2|R-square|accuracy|F1|precision|recall|error|score|dataset|benchmark",
    re.I,
)

llm = ChatOpenAI(model=MODEL, temperature=0)


class ModelResult(BaseModel):
    model_name: str = Field(
        description="Name of the model/method this result is for — either the "
                     "paper's own proposed method, or a baseline it compares against."
    )
    is_proposed: bool = Field(
        description="True if this is the paper's own proposed method, False if "
                     "it's a baseline/comparison method from prior work."
    )
    dataset: str = Field(
        description="The dataset, benchmark, or experimental test condition this "
                     "result was measured on, exactly as the paper names it (e.g. a "
                     "named public dataset, or a test cycle/protocol if the paper "
                     "uses its own experimental setup instead of a public dataset)."
    )
    metric_name: str = Field(description="Name of the metric reported, e.g. RMSE, MAE, Accuracy, F1.")
    metric_value: str = Field(
        description="The reported value exactly as stated, including units/percent "
                     "signs. Do not compute, convert, or round it yourself."
    )


class PaperResults(BaseModel):
    results: list[ModelResult] = Field(
        description="Every distinct model-dataset-metric result mentioned in the "
                     "excerpts, for both the paper's own method and any baselines "
                     "it compares itself against. Empty list if none are present."
    )


extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract quantitative model comparison results from excerpts of a "
     "research paper. Base every result only on numbers and names that "
     "literally appear in the excerpts — never estimate, infer, or invent a "
     "value. If the same model is reported on more than one dataset or test "
     "condition, extract a separate result for each.\n\n"
     "metric_value must be an absolute number explicitly stated in the text "
     "(e.g. '0.76%', '0.0042', '1.1'). Two things to never do:\n"
     "1. Never record a vague qualitative phrase as metric_value (e.g. 'high "
     "accuracy', 'state-of-the-art', 'lower than X', 'not specified') — if no "
     "explicit absolute number is given for a model, omit that result entirely "
     "rather than inventing or approximating one.\n"
     "2. Never use a relative/comparative statement (e.g. 'improved by "
     "approximately 60% compared with the standard LSTM') as if it were the "
     "comparison target's own absolute value — that percentage describes the "
     "size of the improvement, not the baseline's metric. Only record it if "
     "the excerpt also separately states the baseline's own absolute number; "
     "otherwise omit that baseline's result."),
    ("human", "Paper title: {title}\n\nExcerpts:\n{excerpts}\n\nExtract all results."),
])
extract_chain = extract_prompt | llm.with_structured_output(PaperResults)


# A chunk containing an actual digit-percent/decimal pattern (e.g. "0.76%",
# "0.0042") is much more likely to be an actual results table row than one
# that just mentions a metric name in passing — prioritize those first so
# the excerpt budget doesn't get used up on narrative mentions instead.
NUMERIC_RESULT_RE = re.compile(r"\d+\.\d+\s*%|\b0\.\d{2,4}\b")


def gather_excerpts(texts: list[str], max_excerpts: int = 12) -> str:
    picked = [texts[0]]  # first chunk is usually title/abstract
    remaining = texts[1:]

    numeric_hits = [t for t in remaining if KEYWORD_RE.search(t) and NUMERIC_RESULT_RE.search(t)]
    keyword_only_hits = [t for t in remaining if KEYWORD_RE.search(t) and t not in numeric_hits]

    for t in numeric_hits + keyword_only_hits:
        if len(picked) >= max_excerpts:
            break
        picked.append(t)

    return "\n\n---\n\n".join(p[:900] for p in picked)


def extract_all_results(chunks_path: Path) -> dict[str, list[dict]]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c["text"])

    all_results = {}
    for source, texts in by_source.items():
        excerpts = gather_excerpts(texts)
        parsed: PaperResults = extract_chain.invoke({"title": source, "excerpts": excerpts})
        all_results[source] = [r.model_dump() for r in parsed.results]
        print(f"  {source[:70]}... {len(parsed.results)} result(s)")
    return all_results


def _numeric_value(raw: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", raw)
    return float(match.group()) if match else None


def render_infographic(all_results: dict[str, list[dict]], output_path: Path) -> None:
    # Only keep results whose metric_value actually parses to a number —
    # qualitative extraction slip-throughs ("state-of-the-art", "not
    # specified") get dropped here rather than plotted as a misleading
    # zero-height bar.
    plottable = {}
    for paper, results in all_results.items():
        rows_with_values = [r for r in results if _numeric_value(r["metric_value"]) is not None]
        if rows_with_values:
            plottable[paper] = rows_with_values

    skipped = set(all_results) - set(plottable)
    if skipped:
        print(f"Skipping {len(skipped)} paper(s) with no numeric results to plot:")
        for p in skipped:
            print(f"  - {p[:70]}")

    papers = list(plottable.keys())
    n = len(papers)
    if n == 0:
        print("No results with extractable data — nothing to render.")
        return

    cols = 2
    rows = -(-n // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, paper in zip(axes, papers):
        results = plottable[paper]

        # chart the paper's dominant metric only — mixing units (RMSE% vs
        # MAPE vs R2) on one axis would misrepresent the comparison
        metric_counts = Counter(r["metric_name"] for r in results)
        dominant_metric = metric_counts.most_common(1)[0][0]
        rows_for_metric = [r for r in results if r["metric_name"] == dominant_metric]

        models = list(dict.fromkeys(r["model_name"] for r in rows_for_metric))
        datasets = list(dict.fromkeys(r["dataset"] for r in rows_for_metric))
        proposed = {r["model_name"] for r in rows_for_metric if r["is_proposed"]}

        value_lookup = {(r["model_name"], r["dataset"]): _numeric_value(r["metric_value"]) for r in rows_for_metric}

        x = np.arange(len(models))
        width = 0.8 / max(len(datasets), 1)

        for i, dataset in enumerate(datasets):
            values = [value_lookup.get((m, dataset)) for m in models]
            offsets = x + i * width - (width * (len(datasets) - 1) / 2)
            colors = ["#4C72B0" if m in proposed else "#DD8452" for m in models]
            # skip bars with no value at this offset rather than drawing a
            # phantom zero-height one
            plot_offsets = [o for o, v in zip(offsets, values) if v is not None]
            plot_values = [v for v in values if v is not None]
            plot_colors = [c for c, v in zip(colors, values) if v is not None]
            alpha = max(0.35, 0.9 - 0.1 * i)
            ax.bar(plot_offsets, plot_values, width=width, label=dataset,
                   color=plot_colors, edgecolor="white", alpha=alpha)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(dominant_metric, fontsize=9)
        ax.set_title(paper[:60] + ("..." if len(paper) > 60 else ""), fontsize=9, fontweight="bold")
        if len(datasets) > 1:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes[n:]:
        ax.axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="Proposed method"),
        plt.Rectangle((0, 0), 1, 1, color="#DD8452", label="Baseline"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Model vs. baseline results by paper and dataset", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build a model/baseline results infographic from a paper corpus.")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"),
                         help="Path to a chunks.json-style corpus (list of {source, text}).")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                         help="Directory to write infographic_data.json and infographic.png into.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run LLM extraction even if a cached infographic_data.json exists.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "infographic_data.json"
    image_path = args.output_dir / "infographic.png"

    if data_path.exists() and not args.force:
        print(f"Using cached {data_path} (pass --force to re-extract)")
        all_results = json.loads(data_path.read_text(encoding="utf-8"))
    else:
        print("Extracting model/baseline results per paper...")
        all_results = extract_all_results(args.chunks)
        data_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Wrote {data_path}")

    render_infographic(all_results, image_path)


if __name__ == "__main__":
    main()
