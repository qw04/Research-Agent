"""
Renders data/paper_graph.json (built by src/paper_graph/build_paper_graph.py) as:
  - data/paper_graph.png   — static image via networkx + matplotlib
  - data/paper_graph.mmd   — Mermaid flowchart source, for embedding in a
                              GitHub README (Mermaid renders natively there;
                              it does not in Notion, so use the PNG for that)

Paper nodes and technique nodes are drawn/labeled differently, and the three
edge types (uses / compares_against_paper / compares_against_technique) get
distinct styles so the method-lineage edges (paper-to-paper) stand out from
plain technique usage.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

GRAPH_FILE = Path("data/paper_graph.json")
PNG_FILE = Path("data/paper_graph.png")
MMD_FILE = Path("data/paper_graph.mmd")


def load_graph() -> dict:
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


def short_label(title: str, max_words: int = 5) -> str:
    words = title.split()
    label = " ".join(words[:max_words])
    return label + ("..." if len(words) > max_words else "")


def render_png(data: dict) -> None:
    G = nx.DiGraph()

    papers = list(data["papers"].keys())
    techniques = data["technique_nodes"]
    G.add_nodes_from(papers, kind="paper")
    G.add_nodes_from(techniques, kind="technique")

    uses_edges = [(e["from"], e["to"]) for e in data["edges"] if e["type"] == "uses"]
    paper_edges = [(e["from"], e["to"]) for e in data["edges"] if e["type"] == "compares_against_paper"]
    tech_edges = [(e["from"], e["to"]) for e in data["edges"] if e["type"] == "compares_against_technique"]
    G.add_edges_from(uses_edges)
    G.add_edges_from(paper_edges)
    G.add_edges_from(tech_edges)

    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=100)

    fig, ax = plt.subplots(figsize=(18, 13))

    nx.draw_networkx_nodes(G, pos, nodelist=papers, node_color="#4C72B0",
                            node_shape="o", node_size=1400, ax=ax, label="Paper")
    nx.draw_networkx_nodes(G, pos, nodelist=techniques, node_color="#DD8452",
                            node_shape="s", node_size=700, ax=ax, label="Technique")

    nx.draw_networkx_edges(G, pos, edgelist=uses_edges, edge_color="#999999",
                            style="solid", width=1.0, alpha=0.6, arrowsize=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=tech_edges, edge_color="#DD8452",
                            style="dotted", width=1.2, alpha=0.7, arrowsize=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=paper_edges, edge_color="#C44E52",
                            style="dashed", width=2.0, alpha=0.9, arrowsize=15, ax=ax)

    paper_labels = {p: short_label(p) for p in papers}
    tech_labels = {t: t for t in techniques}
    nx.draw_networkx_labels(G, pos, labels=paper_labels, font_size=7,
                             font_weight="bold", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=tech_labels, font_size=6, ax=ax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=12, label="Paper"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#DD8452", markersize=10, label="Technique"),
        plt.Line2D([0], [0], color="#999999", linestyle="solid", label="uses"),
        plt.Line2D([0], [0], color="#DD8452", linestyle="dotted", label="compares against (technique)"),
        plt.Line2D([0], [0], color="#C44E52", linestyle="dashed", linewidth=2, label="compares against (paper)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title("Paper dependency / technique graph — battery SOC estimation corpus", fontsize=13)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=150)
    plt.close(fig)
    print(f"Wrote {PNG_FILE}")


def _mermaid_id(name: str, prefix: str, counter: dict) -> str:
    if name not in counter:
        counter[name] = f"{prefix}{len(counter) + 1}"
    return counter[name]


def _mermaid_escape(label: str) -> str:
    return re.sub(r'["\[\]]', "", label)


def render_mermaid(data: dict) -> None:
    paper_ids: dict[str, str] = {}
    tech_ids: dict[str, str] = {}

    lines = ["flowchart LR"]

    for paper in data["papers"]:
        pid = _mermaid_id(paper, "P", paper_ids)
        lines.append(f'  {pid}["{_mermaid_escape(short_label(paper))}"]')
    for tech in data["technique_nodes"]:
        tid = _mermaid_id(tech, "T", tech_ids)
        lines.append(f'  {tid}("{_mermaid_escape(tech)}")')

    def node_id(name: str) -> str:
        return paper_ids.get(name) or tech_ids.get(name)

    for e in data["edges"]:
        a, b = node_id(e["from"]), node_id(e["to"])
        if e["type"] == "uses":
            lines.append(f"  {a} -->|uses| {b}")
        elif e["type"] == "compares_against_technique":
            lines.append(f"  {a} -.->|baseline| {b}")
        else:  # compares_against_paper
            lines.append(f"  {a} ==>|compares vs| {b}")

    MMD_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {MMD_FILE}")


if __name__ == "__main__":
    graph = load_graph()
    render_png(graph)
    render_mermaid(graph)
