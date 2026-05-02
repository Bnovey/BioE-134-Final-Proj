# CRE-seq Analyzer

**BioEng 134 Final Project — Individual Submission**
**Author:** Arjun Gurjar (`arjungurjar2006@gmail.com`)

---

## My Contributions

I built the **Streamlit frontend** and the **MCP Chat integration** for this project. The backend analysis modules (QC library, activity calling, motif enrichment, stats) were built by teammates; I designed and implemented the UI layer that exposes all of that functionality to end users.

### Functions I wrote (`frontend/app.py`)

| Function / Section | What it does |
|---|---|
| `_load_results()` | Reads `activity_results.tsv` from disk, normalises column names |
| `_run_async(coro)` | Runs an async coroutine from Streamlit's synchronous context |
| `_load_paper_excerpt()` | Loads the Agarwal 2025 paper excerpt for the agent system prompt |
| `_build_system_prompt()` | Constructs the Claude agent system prompt with pipeline context |
| `_extract_charts()` | Parses MCP tool results and builds inline Plotly charts (volcano, motif bar) |
| `_agent_turn()` | Full async Claude + MCP agent loop with tool discovery, execution, and chart extraction |
| Upload page | 4-step pipeline UI: association → counting → activity, with per-step status tracker, skip-association toggle, activity calling threshold controls |
| Chat page | Claude + MCP chat interface with suggested prompt chips, tool call log expander |
| QC & Plots page | 4-tab dashboard: Library QC (8 checks, threshold controls, CSV export), Activity Plots, Motif Analysis (run-from-UI enrichment), Variant Effects (run-from-UI delta scores) |
| Results page | Summary metrics, live search/filter, computed enrichment summary, CSV download |
| Help page | Pipeline diagram, tool reference, file glossary, FAQ |
| Sidebar | Navigation, session status, two-step reset confirmation |

### MCP tools I added (`creseq_mcp/server.py`)

| Tool | What it does |
|---|---|
| `tool_variant_delta_scores` | Computes Δlog₂ (mutant − reference) for all variant families; writes `variant_delta_scores.tsv` |
| `tool_export_qc_html` | Runs all 8 QC checks and writes a self-contained HTML report |

---

## Project Overview

CRE-seq (cis-regulatory element sequencing) measures the transcriptional activity of thousands of DNA regulatory elements in parallel using a lentiMPRA assay. This tool takes raw FASTQ barcode reads and outputs activity scores for each element with statistical significance, motif annotations, and variant effect sizes.

**Reference:** Agarwal et al. 2025, *Nature* — "Massively parallel characterization of transcriptional regulatory elements"

---

## Architecture

```
FASTQ reads
    │
    ▼
1. Association  ─── mappy alignment + STARCODE barcode clustering
    │                    → mapping_table.tsv
    ▼
2. DNA Counting ─── count plasmid barcodes per oligo
    │                    → plasmid_counts.tsv
    ▼
3. RNA Counting ─── count RNA barcodes (multiple replicates, parallel)
    │                    → rna_counts.tsv
    ▼
4. Activity     ─── log₂(RNA/DNA), z-score vs. negative controls, BH-FDR
                     → activity_results.tsv

Streamlit UI  ◄──────────────────────── MCP Server (32 tools)
     │                                       │
     └─── Claude (claude-sonnet-4-6) ────────┘
          AsyncAnthropic + stdio MCP client
```

---

## Installation

```bash
# Clone and create conda environment
git clone <repo-url>
cd BioE-134-Final-Proj
conda env create -f environment.yml
conda activate creseq

# Install the MCP package in editable mode
pip install -e .
```

**Dependencies:** Python 3.11+, mappy, starcode (for association step), anthropic, mcp, streamlit, pandas, numpy, scipy, statsmodels, plotly, matplotlib, seaborn, pyjaspar, biopython.

---

## Running the App

```bash
cd BioE-134-Final-Proj
streamlit run frontend/app.py
```

Set your Anthropic API key for the Chat agent:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## UI Pages

### 📤 Upload
Run the 4-step pipeline by providing file paths to your FASTQ data. Toggle "Skip association" if you already have a mapping table. Adjust activity calling thresholds (FDR, log₂FC cutoff, min DNA count) in the expander before running.

### 📊 QC & Plots
- **Library QC** — run all 8 checks (barcode complexity, oligo recovery, synthesis errors, collisions, uniformity, plasmid depth, GC bias, variant family coverage) with adjustable thresholds. Export results as CSV.
- **Activity Plots** — log₂(RNA/DNA) histogram, volcano plot, active/inactive breakdown by category.
- **Motif Analysis** — run TF motif enrichment directly from the UI or via Chat agent.
- **Variant Effects** — compute Δlog₂ (mutant − reference) scores from the UI.

### 📋 Results
Searchable, filterable element table with live text search, activity filter, and FDR sort. Computed enrichment summary (active rate by category, top elements, top motifs). Download as CSV.

### 💬 Chat
Claude + MCP agent with suggested prompt chips ("Run library QC", "Show volcano plot", "Annotate motifs", etc.) and an expandable tool call log showing which MCP tools were invoked for each response.

### 📖 Help
Pipeline diagram, tool reference table, output file glossary, and FAQ.

---

## MCP Tools

The MCP server exposes 32 registered tools to the Claude agent:

**Library QC (9 tools):** `tool_barcode_complexity`, `tool_oligo_recovery`, `tool_synthesis_error_profile`, `tool_barcode_collision_analysis`, `tool_barcode_uniformity`, `tool_plasmid_depth_summary`, `tool_gc_content_bias`, `tool_oligo_length_qc`, `tool_variant_family_coverage`, `tool_library_summary_report`

**Stats & Activity (7 tools):** `tool_normalize_activity`, `tool_call_active_elements_full`, `tool_rank_cre_candidates`, `tool_motif_enrichment_summary`, `tool_aggregate_fastq_counts_to_elements`, `tool_count_barcodes_from_fastq`, `tool_prepare_rag_context`

**Literature (4 tools):** `tool_search_pubmed`, `tool_search_jaspar_motif`, `tool_search_encode_tf`, `tool_literature_search_for_motifs`, `tool_interpret_literature_evidence`

**Plotting (1 tool):** `tool_plot_creseq` — volcano, ranked_activity, replicate_correlation, annotation_boxplot, motif_dotplot

**Analysis (added by me — 2 tools):** `tool_variant_delta_scores`, `tool_export_qc_html`

---

## Team Contributions

| Member | Contribution |
|---|---|
| Arjun Gurjar | Streamlit frontend, MCP Chat integration, `tool_variant_delta_scores`, `tool_export_qc_html` |
| Bowman Novey | Backend pipeline (association, counting, activity calling), MCP server wiring, library QC module, integration |
| Sarrah Rose | Activity calling module, motif enrichment, plotting |
| Zach Rao | Stats & RAG tools (normalization, ranking, PubMed/JASPAR/ENCODE search) |

---

## Citation

Agarwal, H. et al. "Massively parallel characterization of transcriptional regulatory elements." *Nature* (2025). DOI: 10.1038/s41586-024-08430-9
