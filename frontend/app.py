"""CRE-seq Analysis Tool — Streamlit frontend."""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import anthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pathlib import Path

UPLOAD_DIR = Path.home() / "Desktop" / "creseq_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(UPLOAD_DIR)
PROJECT_ROOT = str(Path(__file__).parent.parent)
PAPER_PATH = Path(PROJECT_ROOT) / "creseq_mcp" / "data" / "papers" / "agarwal2025_lentimpra.txt"

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRE-seq Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state defaults ────────────────────────────────────────────────────
def _load_results() -> pd.DataFrame:
    p = UPLOAD_DIR / "activity_results.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        if "oligo_id" in df.columns and "element_id" not in df.columns:
            df = df.rename(columns={"oligo_id": "element_id"})
        return df
    return pd.DataFrame()

if "data" not in st.session_state:
    st.session_state.data: pd.DataFrame = _load_results()
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run: bool = (UPLOAD_DIR / "activity_results.tsv").exists()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages: list = []

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 CRE-seq Analyzer")
    st.caption("BioEng 134 · Final Project")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📤 Upload", "📊 QC & Plots", "📋 Results", "💬 Chat"],
        label_visibility="collapsed",
    )
    st.divider()
    n_elements = len(st.session_state.data)
    if n_elements:
        st.info(f"**{n_elements:,} elements** loaded from activity_results.tsv")
    else:
        st.warning("No results yet — complete the Upload pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload":
    st.header("Upload CRE-seq Data")
    st.caption("Enter local file paths — no upload needed since the app runs on your machine.")

    def _resolve(s: str) -> Path | None:
        s = s.strip()
        if not s:
            return None
        p = Path(s).expanduser()
        return p if p.exists() else None

    # ── Skip association ─────────────────────────────────────────────────────
    skip_assoc = st.toggle("Skip association — use existing mapping table")
    if skip_assoc:
        _sk_col1, _sk_col2 = st.columns(2)
        with _sk_col1:
            existing_mapping_str = st.text_input(
                "Mapping table path *",
                placeholder="~/Desktop/creseq_outputs/mapping_table.tsv",
                help="mapping_table.tsv produced by a previous association run.",
            )
        with _sk_col2:
            existing_manifest_str = st.text_input(
                "Design manifest path (optional)",
                placeholder="~/Desktop/creseq_outputs/design_manifest.tsv",
                help="design_manifest.tsv from the same run. Used for oligo category labels in activity analysis.",
            )
        existing_mapping_path = _resolve(existing_mapping_str)
        existing_manifest_path = _resolve(existing_manifest_str)
        if existing_mapping_str.strip() and not existing_mapping_path:
            st.error(f"Mapping table not found — {existing_mapping_str.strip()}")
        if existing_manifest_str.strip() and not existing_manifest_path:
            st.warning(f"Design manifest not found — {existing_manifest_str.strip()}")
    st.divider()

    # ── Association inputs (hidden when skipping) ────────────────────────────
    if not skip_assoc:
        st.subheader("Association (barcode → oligo mapping)")
        col1, col2 = st.columns(2)
        with col1:
            assoc_r1_str = st.text_input(
                "Association R1 FASTQ *",
                placeholder="~/Desktop/creseq_test_data/assoc_R1.fastq.gz",
                help="R1 oligo reads — used for alignment to the design FASTA.",
            )
        with col2:
            design_fasta_str = st.text_input(
                "Design FASTA *",
                placeholder="~/Desktop/creseq_test_data/reference.fa",
                help="FASTA of all designed oligo sequences — used for alignment.",
            )

        col1b, col2b = st.columns(2)
        with col1b:
            assoc_bc_str = st.text_input(
                "Barcode index FASTQ *",
                placeholder="~/Desktop/creseq_test_data/assoc_bc.fastq.gz",
                help="15bp i5 barcode reads (ENCODE: separate index file). Required unless barcodes are embedded in R1 headers (older MiSeq data).",
            )
        with col2b:
            labels_path_str = st.text_input(
                "Labels TSV (optional)",
                placeholder="~/Desktop/creseq_test_data/labels.tsv",
                help="TSV with oligo_id + designed_category columns. If omitted all oligos are labelled 'other'.",
            )
    else:
        assoc_r1_str = assoc_bc_str = design_fasta_str = labels_path_str = ""

    # ── Counting inputs ──────────────────────────────────────────────────────
    st.subheader("Counting")
    col3, col4 = st.columns(2)
    with col3:
        dna_path_str = st.text_input(
            "DNA Counting FASTQ *",
            placeholder="~/Desktop/creseq_test_data/dna_rep1.fastq.gz",
            help="15 bp barcode reads from your plasmid pool.",
        )
    with col4:
        rna_paths_str = st.text_input(
            "RNA FASTQs * (comma-separated)",
            placeholder="~/Desktop/creseq_test_data/rna_rep1.fastq.gz, ~/Desktop/creseq_test_data/rna_rep2.fastq.gz",
            help="One path per replicate, separated by commas.",
        )

    # ── Advanced options ─────────────────────────────────────────────────────
    with st.expander("Advanced options"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            assoc_r2_str = st.text_input(
                "Association R2 FASTQ (optional)",
                placeholder="~/Desktop/creseq_test_data/assoc_R2.fastq.gz",
                help="Paired-end R2 oligo reads — improves alignment rate when provided.",
            )
        with adv_col2:
            bc_len = st.number_input("Barcode length (bp)", min_value=6, max_value=30, value=15)
        adv_col3, adv_col4, adv_col5 = st.columns(3)
        min_cov    = adv_col3.number_input("Min reads per barcode", min_value=1, max_value=20, value=3)
        min_frac   = adv_col4.number_input("Min mapping fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        mapq_thr   = adv_col5.number_input("Min MAPQ", min_value=0, max_value=60, value=20)

    # ── Activity calling options ──────────────────────────────────────────────
    with st.expander("Activity calling options"):
        act_col1, act_col2, act_col3 = st.columns(3)
        act_fdr_threshold = act_col1.number_input(
            "FDR threshold",
            min_value=0.001, max_value=0.20, value=0.05, step=0.005, format="%.3f",
            help="Benjamini–Hochberg FDR cutoff for calling an element active. Lower = more stringent.",
        )
        act_log2fc_min = act_col2.number_input(
            "Min log₂ fold-change",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.1f",
            help="Elements below this log₂(RNA/DNA) are not called active regardless of p-value.",
        )
        act_min_dna = act_col3.number_input(
            "Min DNA count per element",
            min_value=0, max_value=100, value=10,
            help="Elements with fewer than this many plasmid reads are excluded from activity calling.",
        )

    # ── Resolve and validate paths ───────────────────────────────────────────
    assoc_r1_path  = _resolve(assoc_r1_str)
    assoc_r2_path  = _resolve(assoc_r2_str)
    assoc_bc_path  = _resolve(assoc_bc_str)
    design_fasta   = _resolve(design_fasta_str)
    labels_path    = _resolve(labels_path_str)
    dna_path       = _resolve(dna_path_str)
    rna_paths      = [p for s in rna_paths_str.split(",") if (p := _resolve(s)) is not None] if rna_paths_str.strip() else []

    if not skip_assoc:
        for label, val, raw in [
            ("Association R1", assoc_r1_path, assoc_r1_str),
            ("Design FASTA",   design_fasta,  design_fasta_str),
            ("DNA FASTQ",      dna_path,      dna_path_str),
        ]:
            if raw.strip() and not val:
                st.error(f"{label}: file not found — {raw.strip()}")

        for opt_label, opt_val, opt_raw in [
            ("Association R2", assoc_r2_path, assoc_r2_str),
            ("Labels TSV",     labels_path,   labels_path_str),
        ]:
            if opt_raw.strip() and not opt_val:
                st.warning(f"{opt_label}: file not found — {opt_raw.strip()}")

        if assoc_bc_str.strip() and not assoc_bc_path:
            st.error(f"Barcode index FASTQ: file not found — {assoc_bc_str.strip()}")

        if rna_paths_str.strip():
            for s in rna_paths_str.split(","):
                if s.strip() and not _resolve(s):
                    st.error(f"RNA FASTQ not found: {s.strip()}")

        # Check whether R1 headers contain embedded barcodes (i5 format)
        _r1_has_header_bc = False
        if assoc_r1_path:
            try:
                import gzip as _gz
                _opener = _gz.open if str(assoc_r1_path).endswith(".gz") else open
                with _opener(assoc_r1_path, "rt") as _fh:
                    _hdr = _fh.readline()
                    _parts = _hdr.split()
                    if len(_parts) >= 2 and "+" in _parts[1].split(":")[-1]:
                        _r1_has_header_bc = True
            except Exception:
                pass

        _bc_required = not _r1_has_header_bc
        if _bc_required and not assoc_bc_path:
            st.info("Barcode index FASTQ required — R1 headers don't contain embedded barcodes (ENCODE format).")
    else:
        _r1_has_header_bc = False

    # ── Process button ───────────────────────────────────────────────────────
    if skip_assoc:
        ready = bool(existing_mapping_path and dna_path and len(rna_paths) > 0)
        if not ready:
            missing = [n for n, v in [
                ("Mapping table", existing_mapping_path),
                ("DNA FASTQ",     dna_path),
                ("RNA FASTQs",    rna_paths or None),
            ] if not v]
            if missing:
                st.info(f"Still needed: {', '.join(missing)}")
    else:
        ready = bool(
            assoc_r1_path and design_fasta and dna_path and len(rna_paths) > 0
            and (assoc_bc_path or _r1_has_header_bc)
        )
        if not ready:
            missing = [n for n, v in [
                ("Association R1",     assoc_r1_path),
                ("Barcode index FASTQ", assoc_bc_path or (_r1_has_header_bc or None)),
                ("Design FASTA",       design_fasta),
                ("DNA FASTQ",          dna_path),
                ("RNA FASTQs",         rna_paths or None),
            ] if not v]
            if missing:
                st.info(f"Still needed: {', '.join(missing)}")

    if st.button("▶ Process all files", type="primary", use_container_width=True, disabled=not ready):
        from creseq_mcp.processing.association import run_association
        from creseq_mcp.processing.counting import process_dna_counting, process_rna_counting
        from creseq_mcp.qc.activity import normalize_and_compute_ratios, call_activity

        STEPS = ["Association", "DNA Counting", "RNA Counting", "Activity Analysis"]
        step_status = {s: ("⏳", "—") for s in STEPS}  # (icon, elapsed)

        status_table = st.empty()

        def _render_status():
            rows = "| Step | Status | Time |\n|---|---|---|\n"
            for s in STEPS:
                icon, elapsed = step_status[s]
                rows += f"| {s} | {icon} | {elapsed} |\n"
            status_table.markdown(rows)

        _render_status()

        try:
            # ── Step 1: Association ──────────────────────────────────────────
            step_status["Association"] = ("🔄 Running…", "—")
            _render_status()
            _t0 = time.time()
            if skip_assoc:
                import shutil as _shutil
                _dest = UPLOAD_DIR / "mapping_table.tsv"
                if existing_mapping_path.resolve() != _dest.resolve():
                    _shutil.copy(existing_mapping_path, _dest)
                if existing_manifest_path:
                    _mdest = UPLOAD_DIR / "design_manifest.tsv"
                    if existing_manifest_path.resolve() != _mdest.resolve():
                        _shutil.copy(existing_manifest_path, _mdest)
                assoc_stats = None
            else:
                assoc_stats = run_association(
                    fastq_r1=assoc_r1_path,
                    design_fasta=design_fasta,
                    outdir=UPLOAD_DIR,
                    fastq_r2=assoc_r2_path,
                    fastq_bc=assoc_bc_path,
                    labels_path=labels_path,
                    min_cov=int(min_cov),
                    min_frac=float(min_frac),
                    mapq_threshold=int(mapq_thr),
                )
            step_status["Association"] = ("✅", f"{time.time()-_t0:.1f}s")
            _render_status()

            # ── Steps 2+3: Counting (parallel) ───────────────────────────────
            step_status["DNA Counting"] = ("🔄 Running…", "—")
            step_status["RNA Counting"] = ("🔄 Running…", "—")
            _render_status()
            from concurrent.futures import ThreadPoolExecutor
            _mapping_table = UPLOAD_DIR / "mapping_table.tsv"
            _bc_len = int(bc_len)
            _t1 = time.time()
            with ThreadPoolExecutor(max_workers=2) as _ex:
                _dna_fut = _ex.submit(
                    process_dna_counting, dna_path, _mapping_table, UPLOAD_DIR,
                    barcode_len=_bc_len,
                )
                _rna_fut = _ex.submit(
                    process_rna_counting, rna_paths, _mapping_table, UPLOAD_DIR,
                    barcode_len=_bc_len,
                )
                dna_stats = _dna_fut.result()
                rna_stats = _rna_fut.result()
            _count_elapsed = f"{time.time()-_t1:.1f}s"
            step_status["DNA Counting"] = ("✅", _count_elapsed)
            step_status["RNA Counting"] = ("✅", _count_elapsed)
            _render_status()

            # ── Step 4: Activity analysis ─────────────────────────────────────
            step_status["Activity Analysis"] = ("🔄 Running…", "—")
            _render_status()
            _t3 = time.time()
            manifest_path = UPLOAD_DIR / "design_manifest.tsv"
            _oligo_df, norm_summary = normalize_and_compute_ratios(
                UPLOAD_DIR / "plasmid_counts.tsv",
                UPLOAD_DIR / "rna_counts.tsv",
                manifest_path if manifest_path.exists() else None,
            )
            if int(act_min_dna) > 0 and "dna_counts" in _oligo_df.columns:
                _oligo_df = _oligo_df[_oligo_df["dna_counts"] >= int(act_min_dna)].copy()
            results_df, act_summary = call_activity(
                _oligo_df,
                fdr_threshold=float(act_fdr_threshold),
            )
            if float(act_log2fc_min) > 0 and "log2_ratio" in results_df.columns:
                results_df.loc[results_df["log2_ratio"] < float(act_log2fc_min), "active"] = False
            results_df.to_csv(UPLOAD_DIR / "activity_results.tsv", sep="\t", index=False)
            step_status["Activity Analysis"] = ("✅", f"{time.time()-_t3:.1f}s")
            _render_status()

            st.success("All steps complete — go to Chat to run QC or QC & Plots to see results.")
            c1, c2, c3, c4, c5 = st.columns(5)
            if assoc_stats:
                c1.metric("Reads (assoc)", f"{assoc_stats['n_reads_total']:,}")
                c2.metric("Aligned", f"{assoc_stats['pct_aligned']:.1f}%")
                c3.metric("Barcodes (filtered)", f"{assoc_stats['n_barcodes_passing_filter']:,}")
            else:
                c1.metric("Reads (assoc)", "—")
                c2.metric("Aligned", "—")
                c3.metric("Barcodes (filtered)", "—")
            c4.metric("RNA replicates", len(rna_stats["replicates"]))
            c5.metric("Active CREs", f"{act_summary['n_active']:,}")

            if assoc_stats and assoc_stats.get("warnings"):
                for w in assoc_stats["warnings"]:
                    st.warning(w)

        except Exception as exc:
            for s in STEPS:
                if step_status[s][0].startswith("🔄"):
                    step_status[s] = ("❌", "failed")
            _render_status()
            st.error(f"Pipeline failed: {exc}")

    # ── File status ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Generated files")
    cols = st.columns(5)
    for col, (label, fname) in zip(cols, [
        ("Mapping table", "mapping_table.tsv"),
        ("Plasmid counts", "plasmid_counts.tsv"),
        ("Design manifest", "design_manifest.tsv"),
        ("RNA counts", "rna_counts.tsv"),
        ("Activity results", "activity_results.tsv"),
    ]):
        p = UPLOAD_DIR / fname
        if p.exists():
            col.success(f"**{label}**")
        else:
            col.warning(f"**{label}**")



# ══════════════════════════════════════════════════════════════════════════════
# CHAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _load_paper_excerpt(max_chars: int = 4000) -> str:
    try:
        return PAPER_PATH.read_text()[:max_chars]
    except FileNotFoundError:
        return ""


def _build_system_prompt() -> str:
    paper = _load_paper_excerpt()
    return f"""You are a bioinformatics assistant for lentiMPRA/CRE-seq analysis.
Default output directory: {OUTPUT_DIR}

## Agarwal et al. 2025 (Nature) — Context
{paper}

## Common workflows
- "run library QC"       → tool_library_summary_report
- "call active elements" → tool_call_active_elements_full
- "annotate motifs"      → tool_motif_enrichment_summary
- "show volcano plot"    → tool_plot_creseq with plot_type="volcano"
- "rank CREs"            → tool_rank_cre_candidates

When no paths are given, assume files are in {OUTPUT_DIR}/. Explain results in plain language."""


def _extract_charts(tool_name: str, result_text: str, charts: list, images: list) -> None:
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return

    rows = data.get("rows") if isinstance(data, dict) else (data if isinstance(data, list) else None)
    if rows and isinstance(rows, list) and rows and "mean_activity" in rows[0] and "pvalue" in rows[0]:
        x = [r.get("mean_activity", 0) for r in rows]
        y = [-math.log10(max(r.get("pvalue", 1), 1e-300)) for r in rows]
        colors = ["#e74c3c" if r.get("active") else "#95a5a6" for r in rows]
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=colors, size=5, opacity=0.7),
            text=[r.get("element_id", "") for r in rows],
            hovertemplate="%{text}<br>log₂=%{x:.2f}<br>-log₁₀p=%{y:.2f}<extra></extra>",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_hline(y=-math.log10(0.05), line_dash="dash", line_color="gray", opacity=0.6)
        fig.update_layout(title="Volcano Plot", xaxis_title="log₂ Activity",
                          yaxis_title="-log₁₀(p-value)", height=480)
        charts.append(fig)

    elif tool_name == "tool_motif_enrichment_summary" and isinstance(data, dict):
        motifs = data.get("top_motifs", data.get("motifs", []))[:15]
        if motifs:
            names = [m.get("motif", m.get("tf_name", "")) for m in motifs]
            ratios = [m.get("enrichment_ratio", m.get("odds_ratio", 0)) for m in motifs]
            colors = ["#e74c3c" if r >= 2 else "#3498db" for r in ratios]
            fig = go.Figure(go.Bar(x=ratios, y=names, orientation="h", marker_color=colors))
            fig.update_layout(title="Top Enriched TF Motifs", xaxis_title="Enrichment Ratio",
                              height=max(300, len(names) * 25 + 100),
                              yaxis=dict(autorange="reversed"))
            charts.append(fig)

    elif tool_name == "tool_plot_creseq" and isinstance(data, dict) and "plot_path" in data:
        if Path(data["plot_path"]).exists():
            images.append(data["plot_path"])


async def _agent_turn(user_text: str, history: list) -> dict:
    server_params = StdioServerParameters(
        command="python", args=["-m", "creseq_mcp.server"], cwd=PROJECT_ROOT
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in tools_result.tools
            ]
            client = anthropic.AsyncAnthropic()
            messages = list(history) + [{"role": "user", "content": user_text}]
            charts: list = []
            images: list = []
            tool_log: list[dict] = []

            while True:
                response = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=_build_system_prompt(),
                    tools=tools,
                    messages=messages,
                )
                if response.stop_reason == "end_turn":
                    text = next((b.text for b in response.content if hasattr(b, "text")), "")
                    return {"text": text, "charts": charts, "images": images, "tool_log": tool_log}

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        result = await session.call_tool(block.name, block.input)
                        result_text = result.content[0].text if result.content else "{}"
                        _extract_charts(block.name, result_text, charts, images)
                        snippet = result_text[:200] + ("…" if len(result_text) > 200 else "")
                        tool_log.append({"name": block.name, "inputs": block.input, "result": snippet})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break

    return {"text": "", "charts": charts, "images": images, "tool_log": []}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.header("CRE-seq Assistant")
    st.caption("Ask questions or trigger pipeline tools via Claude + MCP")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["content"]:
                st.markdown(msg["content"])
            for fig in msg.get("charts", []):
                st.plotly_chart(fig, use_container_width=True)
            for img_path in msg.get("images", []):
                st.image(img_path)
            if msg["role"] == "assistant" and msg.get("tool_log"):
                with st.expander(f"🔧 Tools used ({len(msg['tool_log'])})"):
                    for entry in msg["tool_log"]:
                        st.markdown(f"**`{entry['name']}`**")
                        if entry["inputs"]:
                            st.json(entry["inputs"], expanded=False)
                        st.caption(entry["result"])

    # ── Suggested prompt chips ────────────────────────────────────────────────
    SUGGESTED_PROMPTS = [
        "Run library QC",
        "Show volcano plot",
        "Annotate motifs",
        "Rank top CREs",
        "Call active elements",
        "Summarize results",
    ]
    _pending = st.session_state.pop("_pending_chat", None)

    chip_cols = st.columns(len(SUGGESTED_PROMPTS))
    for _col, _chip in zip(chip_cols, SUGGESTED_PROMPTS):
        if _col.button(_chip, key=f"chip_{_chip}", use_container_width=True):
            st.session_state["_pending_chat"] = _chip
            st.rerun()

    _typed = st.chat_input("Run library QC, annotate motifs, show volcano plot…")
    prompt = _pending or _typed

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt, "charts": [], "images": []}
        )

        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = _run_async(_agent_turn(prompt, api_history))
            if result["text"]:
                st.markdown(result["text"])
            for fig in result["charts"]:
                st.plotly_chart(fig, use_container_width=True)
            for img_path in result["images"]:
                st.image(img_path)
            if result.get("tool_log"):
                with st.expander(f"🔧 Tools used ({len(result['tool_log'])})"):
                    for entry in result["tool_log"]:
                        st.markdown(f"**`{entry['name']}`**")
                        if entry["inputs"]:
                            st.json(entry["inputs"], expanded=False)
                        st.caption(entry["result"])

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["text"],
            "charts": result["charts"],
            "images": result["images"],
            "tool_log": result.get("tool_log", []),
        })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QC & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 QC & Plots":
    st.header("QC & Analysis Plots")
    df = st.session_state.data

    tab_qc, tab_activity, tab_motif, tab_variant = st.tabs(
        ["🔬 Library QC", "📈 Activity Plots", "🔡 Motif Analysis", "🧪 Variant Effects"]
    )

    # ── Library QC ──────────────────────────────────────────────────────────
    with tab_qc:
        _mt = UPLOAD_DIR / "mapping_table.tsv"
        _pc = UPLOAD_DIR / "plasmid_counts.tsv"
        _dm = UPLOAD_DIR / "design_manifest.tsv"

        if not _mt.exists():
            st.info("No mapping_table.tsv found. Complete the Upload pipeline first.", icon="ℹ️")
        else:
            if "qc_report" not in st.session_state:
                st.session_state.qc_report = None

            with st.expander("⚙️ QC thresholds"):
                qc_thr_col1, qc_thr_col2, qc_thr_col3 = st.columns(3)
                qc_min_barcodes = qc_thr_col1.number_input(
                    "Min median barcodes/oligo",
                    min_value=1, max_value=50, value=5,
                    help="barcode_complexity check: minimum acceptable median barcodes per oligo.",
                )
                qc_min_recovery = qc_thr_col2.number_input(
                    "Min oligo recovery fraction",
                    min_value=0.1, max_value=1.0, value=0.8, step=0.05, format="%.2f",
                    help="oligo_recovery check: fraction of oligos that must reach ≥10 barcodes.",
                )
                qc_max_collision = qc_thr_col3.number_input(
                    "Max collision rate",
                    min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f",
                    help="barcode_collision_analysis check: max fraction of barcodes mapping to >1 oligo.",
                )
                _qc_thresholds = {
                    "barcode_complexity": {"min_median_barcodes": int(qc_min_barcodes)},
                    "oligo_recovery": {"min_recovery_fraction": float(qc_min_recovery)},
                    "barcode_collision_analysis": {"max_collision_rate": float(qc_max_collision)},
                }

            col_run, col_clear = st.columns([1, 4])
            if col_run.button("▶ Run Library QC"):
                with st.spinner("Running all QC checks…"):
                    try:
                        from creseq_mcp.qc.library import library_summary_report
                        qc_results = library_summary_report(
                            str(_mt), str(_pc),
                            str(_dm) if _dm.exists() else None,
                            thresholds_config=_qc_thresholds,
                        )
                        _, report_summary = qc_results["_report"]
                        st.session_state.qc_report = (qc_results, report_summary)
                    except Exception as e:
                        st.error(f"QC failed: {e}")

            if st.session_state.qc_report is not None:
                qc_results, qc_summary = st.session_state.qc_report
                overall = qc_summary.get("overall_pass", False)
                failed = qc_summary.get("failed_checks", [])
                warnings = qc_summary.get("warnings", [])

                if overall:
                    st.success("✅ Library QC: **PASS** — all checks passed", icon="✅")
                else:
                    st.error(f"❌ Library QC: **FAIL** — failed checks: {', '.join(failed)}", icon="❌")
                if warnings:
                    for w in warnings:
                        st.warning(w)

                tool_results = {k: v[1] for k, v in qc_results.items() if k != "_report"}

                # Per-tool summary cards
                tool_labels = {
                    "barcode_complexity": "Barcode Complexity",
                    "oligo_recovery": "Oligo Recovery",
                    "synthesis_error_profile": "Synthesis Errors",
                    "barcode_collision_analysis": "Barcode Collisions",
                    "barcode_uniformity": "Barcode Uniformity",
                    "plasmid_depth_summary": "Plasmid Depth",
                    "gc_content_bias": "GC Content Bias",
                    "variant_family_coverage": "Variant Family Coverage",
                }
                cols = st.columns(3)
                for i, (tool_key, label) in enumerate(tool_labels.items()):
                    res = tool_results.get(tool_key, {})
                    passed = res.get("pass", None)
                    badge = "✅" if passed else ("❌" if passed is False else "⚪")
                    with cols[i % 3]:
                        st.markdown(f"**{badge} {label}**")

                # QC summary CSV export
                _qc_export_rows = []
                for tool_key, label in tool_labels.items():
                    res = tool_results.get(tool_key, {})
                    _qc_export_rows.append({
                        "check": label,
                        "tool_key": tool_key,
                        "pass": res.get("pass"),
                        **{k: v for k, v in res.items() if k not in ("pass",) and not isinstance(v, (dict, list))},
                    })
                _qc_export_df = pd.DataFrame(_qc_export_rows)
                _qc_csv = _qc_export_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download QC summary (CSV)",
                    _qc_csv,
                    file_name="library_qc_summary.csv",
                    mime="text/csv",
                )

                st.divider()

                # Charts using per-tool DataFrames
                for tool_key, label in tool_labels.items():
                    tool_df = qc_results.get(tool_key, (pd.DataFrame(), {}))[0]
                    res = tool_results.get(tool_key, {})

                    if tool_key == "barcode_complexity" and not tool_df.empty and "n_barcodes" in tool_df.columns:
                        with st.expander(f"📊 {label}"):
                            fig = px.histogram(tool_df, x="n_barcodes", nbins=40,
                                title="Barcodes per Oligo", labels={"n_barcodes": "Barcodes"},
                                color_discrete_sequence=["#4C78A8"])
                            st.plotly_chart(fig, use_container_width=True)
                            med = res.get("median_barcodes_per_oligo")
                            if med:
                                st.metric("Median barcodes/oligo", f"{med:.1f}")

                    elif tool_key == "oligo_recovery" and not tool_df.empty:
                        with st.expander(f"📊 {label}"):
                            if "category" in tool_df.columns and "recovery_at_10" in tool_df.columns:
                                fig = px.bar(tool_df, x="category", y="recovery_at_10",
                                    title="Recovery at ≥10 Barcodes by Category",
                                    labels={"recovery_at_10": "Recovery fraction", "category": "Category"},
                                    color_discrete_sequence=["#54A24B"])
                                fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% target")
                                st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "synthesis_error_profile" and not tool_df.empty:
                        with st.expander(f"📊 {label}"):
                            rate_cols = [c for c in ["mismatch_rate", "indel_rate", "soft_clip_rate"] if c in tool_df.columns]
                            if rate_cols:
                                means = tool_df[rate_cols].mean().reset_index()
                                means.columns = ["Error type", "Rate"]
                                fig = px.bar(means, x="Error type", y="Rate",
                                    title="Mean Synthesis Error Rates",
                                    color_discrete_sequence=["#E45756"])
                                st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "plasmid_depth_summary" and res:
                        with st.expander(f"📊 {label}"):
                            st.metric("Median DNA count", f"{res.get('median_dna_count', 'N/A')}")
                            st.metric("Zero-count barcodes", f"{res.get('frac_zero_barcodes', 0):.1%}")

                    elif tool_key == "gc_content_bias" and not tool_df.empty and "gc_bin" in tool_df.columns:
                        with st.expander(f"📊 {label}"):
                            fig = px.line(tool_df, x="gc_bin", y="recovery_rate",
                                title="Recovery Rate by GC Content Bin",
                                labels={"gc_bin": "GC content", "recovery_rate": "Recovery rate"},
                                markers=True)
                            st.plotly_chart(fig, use_container_width=True)

                    elif tool_key == "variant_family_coverage" and res:
                        with st.expander(f"📊 {label}"):
                            frac = res.get("frac_complete_families", None)
                            miss_ref = res.get("n_families_missing_reference", None)
                            if frac is not None:
                                st.metric("Complete families", f"{frac:.1%}")
                            if miss_ref is not None:
                                st.metric("Families missing reference", str(miss_ref))

    # ── Activity Plots ──────────────────────────────────────────────────────
    with tab_activity:
        _act_path = UPLOAD_DIR / "activity_results.tsv"
        if _act_path.exists():
            act_df = pd.read_csv(_act_path, sep="\t")
            if "oligo_id" in act_df.columns and "element_id" not in act_df.columns:
                act_df = act_df.rename(columns={"oligo_id": "element_id"})
            st.info("Showing real activity data from activity_results.tsv", icon="✅")
        else:
            act_df = df
            st.info(
                "No activity results yet — showing demo data. "
                "Complete all four Upload steps to see real results.",
                icon="ℹ️",
            )

        n_active = int(act_df["active"].sum()) if "active" in act_df.columns else 0
        n_inactive = len(act_df) - n_active
        col1, col2, col3 = st.columns(3)
        col1.metric("Active CREs", f"{n_active:,}")
        col2.metric("Inactive CREs", f"{n_inactive:,}")
        col3.metric("Activity rate", f"{n_active / max(len(act_df), 1) * 100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                act_df, x="log2_ratio",
                color="active",
                color_discrete_map={True: "#E45756", False: "#72B7B2"},
                nbins=60,
                barmode="overlay",
                opacity=0.7,
                title="log₂(RNA/DNA) Distribution",
                labels={"log2_ratio": "log₂ RNA/DNA", "active": "Active"},
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="threshold")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            pval_col = "fdr" if "fdr" in act_df.columns else "pval"
            if pval_col in act_df.columns and act_df[pval_col].notna().any():
                volcano_df = act_df.copy()
                volcano_df["neg_log10_p"] = -np.log10(volcano_df[pval_col].clip(lower=1e-10))
                hover_col = "element_id" if "element_id" in volcano_df.columns else None
                fig = px.scatter(
                    volcano_df, x="log2_ratio", y="neg_log10_p",
                    color="active",
                    color_discrete_map={True: "#E45756", False: "#72B7B2"},
                    opacity=0.6,
                    title=f"Volcano Plot (y = −log₁₀ {pval_col})",
                    labels={
                        "log2_ratio": "log₂ RNA/DNA",
                        "neg_log10_p": f"−log₁₀({pval_col})",
                        "active": "Active",
                    },
                    hover_data=[hover_col] if hover_col else None,
                )
                fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
                fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No p-values available for volcano plot.")

        if "designed_category" in act_df.columns:
            cat_counts = (
                act_df.groupby(["designed_category", "active"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                cat_counts, x="designed_category", y="count",
                color="active",
                color_discrete_map={True: "#E45756", False: "#72B7B2"},
                barmode="stack",
                title="Active vs. Inactive by Element Category",
                labels={"designed_category": "Category", "count": "Elements", "active": "Active"},
            )
            st.plotly_chart(fig, use_container_width=True)
        elif "chromatin_state" in act_df.columns:
            state_counts = (
                act_df.groupby(["chromatin_state", "active"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                state_counts, x="chromatin_state", y="count",
                color="active",
                color_discrete_map={True: "#E45756", False: "#72B7B2"},
                barmode="stack",
                title="Active vs. Inactive CREs by Chromatin State",
                labels={"chromatin_state": "Chromatin State", "count": "Elements", "active": "Active"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Motif Analysis ──────────────────────────────────────────────────────
    with tab_motif:
        st.subheader("TF Motif Enrichment")
        _act_motif_path = UPLOAD_DIR / "activity_results.tsv"

        if _act_motif_path.exists():
            _motif_act_df = pd.read_csv(_act_motif_path, sep="\t")
        else:
            _motif_act_df = pd.DataFrame()

        if "top_motif" in _motif_act_df.columns and "active" in _motif_act_df.columns:
            try:
                from creseq_mcp.stats.library import motif_enrichment_summary
                motif_enr_df, motif_enr_summary = motif_enrichment_summary(str(_act_motif_path))
                st.info("Showing real motif enrichment from activity_results.tsv", icon="✅")

                fig = px.bar(
                    motif_enr_df.head(15), x="motif", y="enrichment_ratio",
                    color=(motif_enr_df.head(15)["enrichment_ratio"] > 2).map({True: "Enriched", False: "Background"}),
                    color_discrete_map={"Enriched": "#E45756", "Background": "#72B7B2"},
                    title="TF Motif Enrichment in Active CREs",
                    labels={"motif": "Motif", "enrichment_ratio": "Enrichment ratio (active/inactive)"},
                    text=motif_enr_df.head(15)["enrichment_ratio"].round(2),
                )
                fig.update_traces(texttemplate="%{text:.1f}×", textposition="outside")
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="no enrichment")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(motif_enr_df, use_container_width=True, hide_index=True)

                motif_freq = (
                    _motif_act_df.groupby("top_motif")["active"]
                    .agg(active_hits="sum", total="count")
                    .reset_index()
                )
                motif_freq["active_rate"] = (motif_freq["active_hits"] / motif_freq["total"] * 100).round(1)
                motif_freq = motif_freq.sort_values("active_rate", ascending=False)
                fig2 = px.bar(
                    motif_freq, x="top_motif", y="active_rate",
                    title="Active Rate by Top Motif (%)",
                    labels={"top_motif": "Motif", "active_rate": "Active (%)"},
                    color="active_rate", color_continuous_scale="RdBu",
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Motif enrichment failed: {e}")
        else:
            st.info(
                "No motif annotations found. Ask the agent to **annotate motifs** "
                "(Chat → 'annotate motifs for my data') to enable this tab.",
                icon="ℹ️",
            )

    # ── Variant Effects ─────────────────────────────────────────────────────
    with tab_variant:
        st.subheader("Variant Family Delta Scores")
        _delta_path = UPLOAD_DIR / "variant_delta_scores.tsv"

        if _delta_path.exists():
            delta_df = pd.read_csv(_delta_path, sep="\t")
            st.info(f"Loaded {len(delta_df):,} mutant–reference pairs from variant_delta_scores.tsv", icon="✅")

            n_sig = int(delta_df["significant"].sum()) if "significant" in delta_df.columns else 0
            n_fam = delta_df["variant_family"].nunique() if "variant_family" in delta_df.columns else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Mutants tested", f"{len(delta_df):,}")
            col2.metric("Variant families", f"{n_fam:,}")
            col3.metric("Significant (FDR < 5%)", f"{n_sig:,}")

            if "ref_log2" in delta_df.columns and "mutant_log2" in delta_df.columns:
                delta_df["Direction"] = delta_df["delta_log2"].apply(
                    lambda x: "Gain" if x > 0 else "Loss"
                )
                axis_max = max(delta_df[["ref_log2", "mutant_log2"]].abs().max()) * 1.1
                fig = px.scatter(
                    delta_df, x="ref_log2", y="mutant_log2",
                    color="significant" if "significant" in delta_df.columns else "Direction",
                    color_discrete_map={True: "#E45756", False: "#72B7B2"},
                    opacity=0.6,
                    hover_data=["oligo_id", "variant_family", "delta_log2"],
                    title="Mutant vs. Reference Activity",
                    labels={"ref_log2": "Reference log₂(RNA/DNA)", "mutant_log2": "Mutant log₂(RNA/DNA)"},
                )
                lims = [-axis_max, axis_max]
                fig.add_trace(go.Scatter(
                    x=lims, y=lims, mode="lines",
                    line=dict(dash="dash", color="gray"), showlegend=False,
                ))
                st.plotly_chart(fig, use_container_width=True)

            if "delta_log2" in delta_df.columns:
                top20 = delta_df.reindex(delta_df["delta_log2"].abs().nlargest(20).index)
                fig2 = px.bar(
                    top20.sort_values("delta_log2"), x="delta_log2", y="oligo_id",
                    orientation="h",
                    color="delta_log2",
                    color_continuous_scale="RdBu_r",
                    title="Top 20 Variants by |Δlog₂|",
                    labels={"delta_log2": "Δlog₂ (mutant − ref)", "oligo_id": "Oligo"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.download_button(
                "Download variant_delta_scores.tsv",
                delta_df.to_csv(sep="\t", index=False).encode(),
                file_name="variant_delta_scores.tsv",
                mime="text/tab-separated-values",
            )
        else:
            st.info(
                "No variant delta scores yet. Ask the agent to compute them "
                "(Chat → 'compute variant delta scores') or run tool_variant_delta_scores.",
                icon="ℹ️",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Results":
    st.header("Analysis Results")
    st.session_state.data = _load_results()
    df = st.session_state.data

    if df.empty:
        st.warning("No results yet — complete the Upload pipeline first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    n_active = int(df["active"].sum()) if "active" in df.columns else 0
    col1.metric("Total CREs", f"{len(df):,}")
    col2.metric("Active", f"{n_active:,}")
    col3.metric("Inactive", f"{len(df) - n_active:,}")
    col4.metric(
        "Median log₂ ratio (active)",
        f"{df.loc[df['active'], 'log2_ratio'].median():.2f}" if "active" in df.columns and n_active > 0 else "—",
    )

    st.subheader("Element Summary Table")

    display_cols = [c for c in ["element_id", "chrom", "start", "end", "dna_counts", "rna_counts", "log2_ratio", "fdr", "pval", "active", "designed_category", "chromatin_state", "top_motif"] if c in df.columns]

    filt_col1, filt_col2, filt_col3 = st.columns([2, 1, 1])
    search_text = filt_col1.text_input("🔍 Search element ID", placeholder="e.g. ENSR00001…")
    activity_filter = filt_col2.selectbox("Activity", ["All", "Active only", "Inactive only"], index=0)
    sort_by_fdr = filt_col3.checkbox("Sort by FDR ↑", value=False, help="Sort rows by FDR ascending (most significant first)")

    display_df = df.copy()
    if search_text:
        id_col = "element_id" if "element_id" in display_df.columns else display_df.columns[0]
        display_df = display_df[display_df[id_col].astype(str).str.contains(search_text, case=False, na=False)]
    if activity_filter == "Active only" and "active" in display_df.columns:
        display_df = display_df[display_df["active"]]
    elif activity_filter == "Inactive only" and "active" in display_df.columns:
        display_df = display_df[~display_df["active"]]
    if sort_by_fdr:
        fdr_col = "fdr" if "fdr" in display_df.columns else ("pval" if "pval" in display_df.columns else None)
        if fdr_col:
            display_df = display_df.sort_values(fdr_col, ascending=True)

    st.caption(f"Showing {len(display_df):,} of {len(df):,} elements")
    st.dataframe(
        display_df[[c for c in display_cols if c in display_df.columns]].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "active": st.column_config.CheckboxColumn("Active"),
            "log2_ratio": st.column_config.NumberColumn("log₂ ratio", format="%.3f"),
            "fdr": st.column_config.NumberColumn("FDR", format="%.4f"),
            "pval": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
    )

    st.divider()

    with st.expander("📌 Enrichment Summary"):
        _has_cat = "designed_category" in df.columns or "chromatin_state" in df.columns
        _cat_col = "designed_category" if "designed_category" in df.columns else ("chromatin_state" if "chromatin_state" in df.columns else None)
        if "active" in df.columns and _cat_col:
            _cat_summary = (
                df.groupby(_cat_col)["active"]
                .agg(active="sum", total="count")
                .assign(active_rate=lambda x: x["active"] / x["total"])
                .sort_values("active_rate", ascending=False)
                .reset_index()
            )
            st.markdown(f"**Active rate by {_cat_col}** ({len(df):,} elements total):")
            for _, row in _cat_summary.iterrows():
                bar = "█" * int(row["active_rate"] * 20) + "░" * (20 - int(row["active_rate"] * 20))
                st.markdown(
                    f"- **{row[_cat_col]}**: {row['active']:,}/{row['total']:,} active "
                    f"({row['active_rate']:.1%})  `{bar}`"
                )
        else:
            st.info("No category or chromatin state column found — run the full pipeline for enrichment stats.")

        if "log2_ratio" in df.columns and "active" in df.columns and n_active > 0:
            _top5 = df[df["active"]].nlargest(5, "log2_ratio")[["element_id" if "element_id" in df.columns else df.columns[0], "log2_ratio"]]
            st.markdown("**Top 5 most active CREs:**")
            for _, row in _top5.iterrows():
                st.markdown(f"- `{row.iloc[0]}` — log₂ = {row['log2_ratio']:.2f}")

        if "top_motif" in df.columns and "active" in df.columns:
            _motif_freq = (
                df[df["active"]]["top_motif"].value_counts().head(5)
            )
            if not _motif_freq.empty:
                st.markdown("**Most common motifs in active CREs:**")
                for motif, cnt in _motif_freq.items():
                    st.markdown(f"- {motif}: {cnt:,} elements")

    with st.expander("🧮 Statistical Model"):
        st.markdown(
            """
Activity was called using a negative binomial GLM (DESeq2-style):

- Size factors computed per sample using median-of-ratios
- Dispersion estimated via empirical Bayes shrinkage
- Hypothesis test: active vs. scrambled negative controls
- Multiple testing correction: Benjamini–Hochberg (FDR < 5%)
"""
        )

    st.divider()
    csv_buffer = io.StringIO()
    display_df[display_cols].to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇ Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="cre_seq_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
