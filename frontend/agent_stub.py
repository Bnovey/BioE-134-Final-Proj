"""Stub MCP agent that simulates tool dispatch without a real LLM/server."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str] = field(default_factory=list)


# keyword → (tool_name, response_text)
_RULES: list[tuple[list[str], str, str]] = [
    (
        ["normalize", "count", "ratio", "rna", "dna"],
        "normalize_counts",
        "I ran **normalize_counts** on your library. DNA counts were depth-normalized using total mapped reads, "
        "and RNA/DNA log₂ ratios were computed per element. Median library size: ~18 reads/element. "
        "Elements with DNA counts < 5 were flagged as low-coverage.",
    ),
    (
        ["qc", "quality", "flagged", "coverage", "depth"],
        "run_qc",
        "**run_qc** complete. 94.2% of barcodes passed mapping. 12 elements flagged for low DNA coverage (<5 reads). "
        "Barcode collision rate: 0.8%. Overall library complexity looks good — Gini coefficient = 0.31.",
    ),
    (
        ["active", "activity", "call", "classify", "inactive"],
        "call_activity",
        "**call_activity** identified **89 active CREs** out of 300 (29.7%). "
        "Activity threshold: log₂(RNA/DNA) > 1.0, FDR < 5% vs. negative control scrambled sequences. "
        "Mean activity score of active elements: log₂ ratio = 2.41.",
    ),
    (
        ["plot", "visualize", "show", "graph", "chart", "histogram", "distribution"],
        "generate_plot",
        "**generate_plot** rendered. Navigate to the **QC & Plots** tab to view the activity score distribution, "
        "volcano plot, and per-element read depth. Plots update dynamically with your uploaded data.",
    ),
    (
        ["motif", "tf", "transcription factor", "binding", "meme", "homer", "fimo"],
        "motif_enrichment",
        "**motif_enrichment** (HOMER-style scan) found 3 significantly enriched motifs in active vs. inactive CREs:\n"
        "- **SP1** — enriched 2.8×, p = 0.0012\n"
        "- **AP1 (FOSL2)** — enriched 2.1×, p = 0.0041\n"
        "- **NRF1** — enriched 1.9×, p = 0.0089\n\n"
        "No significant depletion motifs detected.",
    ),
    (
        ["annotate", "chromatin", "encode", "roadmap", "overlap", "enhancer", "promoter"],
        "annotate_elements",
        "**annotate_elements** overlapped active CREs with ENCODE chromatin states:\n"
        "- Active Enhancer: 38%\n- Weak Enhancer: 29%\n- Promoter-flanking: 18%\n- Other: 15%\n\n"
        "Active elements are significantly enriched in active enhancer states (OR = 3.2, p < 0.001).",
    ),
    (
        ["variant", "allele", "snp", "eqtl", "regulatory", "effect"],
        "variant_effect",
        "**variant_effect** compared allele-matched pairs. 7 elements showed significant allele-specific activity "
        "(|Δlog₂| > 0.5, FDR < 10%). Top hit: CRE_0042 — reference allele 1.8× more active than alternate. "
        "3 of 7 variants overlap known eQTLs.",
    ),
    (
        ["stat", "test", "pvalue", "fdr", "deseq", "glm", "model"],
        "run_statistics",
        "**run_statistics** applied a negative binomial GLM (DESeq2-style) to model count variability. "
        "Dispersion estimates stabilized after 200 iterations. Size factors computed per sample. "
        "Results are available in the **Results** tab with FDR-corrected p-values.",
    ),
    (
        ["help", "what", "can you", "tool", "available", "capability"],
        "list_tools",
        "I have access to the following MCP tools:\n\n"
        "| Tool | Description |\n"
        "|------|-------------|\n"
        "| `normalize_counts` | Map barcodes, aggregate per element, compute log₂ RNA/DNA |\n"
        "| `run_qc` | Library quality metrics, coverage, barcode collision |\n"
        "| `call_activity` | Statistical testing vs. negative controls |\n"
        "| `generate_plot` | Activity distributions, volcano plots, heatmaps |\n"
        "| `motif_enrichment` | TF motif enrichment in active CREs |\n"
        "| `annotate_elements` | Overlap with ENCODE/Roadmap chromatin states |\n"
        "| `variant_effect` | Allele-specific activity and regulatory variant flagging |\n"
        "| `run_statistics` | GLM-based differential activity testing |\n\n"
        "Ask me to run any of these or describe what you want to analyze.",
    ),
]

_FALLBACK = AgentResponse(
    text=(
        "I'm not sure which tool to call for that. Try asking about: "
        "**QC**, **normalization**, **activity calling**, **motif enrichment**, "
        "**variant effects**, **annotation**, or type **help** to see all available tools."
    ),
    tools_called=[],
)


def query_agent(prompt: str, has_data: bool = True) -> AgentResponse:
    if not has_data:
        return AgentResponse(
            text="No data loaded yet. Please upload a CRE-seq file on the **Upload** tab first.",
            tools_called=[],
        )

    lower = prompt.lower()
    matches: list[tuple[int, str, str]] = []
    for keywords, tool, response in _RULES:
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0:
            matches.append((score, tool, response))

    if not matches:
        return _FALLBACK

    matches.sort(key=lambda x: x[0], reverse=True)
    # occasionally call a secondary tool to look realistic
    tools_called = [matches[0][1]]
    if len(matches) > 1 and matches[1][0] >= matches[0][0]:
        tools_called.append(matches[1][1])

    response_text = matches[0][2]
    if len(matches) > 1 and random.random() < 0.3:
        response_text += f"\n\nI also invoked **{matches[1][1]}** as a secondary step."
        tools_called.append(matches[1][1])

    return AgentResponse(text=response_text, tools_called=list(dict.fromkeys(tools_called)))
