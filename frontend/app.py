"""CRE-seq Analysis Tool — Streamlit frontend mockup."""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent_stub import query_agent as stub_query_agent
from mock_data import get_demo_data

from pathlib import Path
UPLOAD_DIR = Path.home() / ".creseq" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

try:
    from agent import ClaudeQCAgent, is_available as _gemini_available
    _GEMINI_READY = True
except ImportError:
    _GEMINI_READY = False
    _gemini_available = lambda: False

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRE-seq Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state defaults ────────────────────────────────────────────────────
if "data" not in st.session_state:
    st.session_state.data: pd.DataFrame = get_demo_data()
if "data_source" not in st.session_state:
    st.session_state.data_source: str = "demo"
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run: bool = True
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the CRE-seq MCP agent. I can run **QC**, **normalization**, "
                "**activity calling**, **motif enrichment**, **variant effect prediction**, "
                "and more.\n\nType **help** to see all available tools, or just describe what you want."
            ),
            "tools": [],
        }
    ]
if "gemini_agent" not in st.session_state:
    st.session_state.gemini_agent = None
if "file_paths" not in st.session_state:
    st.session_state.file_paths: dict[str, str] = {
        "mapping_table_path": "",
        "plasmid_count_path": "",
        "design_manifest_path": "",
    }

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧬 CRE-seq Analyzer")
    st.caption("BioEng 134 · Final Project")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📤 Upload", "💬 Chat", "📊 QC & Plots", "📋 Results"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Data source**")
    source_label = "Demo data" if st.session_state.data_source == "demo" else "Uploaded file"
    st.info(f"**{source_label}** · {len(st.session_state.data):,} elements")
    if st.session_state.data_source != "demo":
        if st.button("Reset to demo data", use_container_width=True):
            st.session_state.data = get_demo_data()
            st.session_state.data_source = "demo"
            st.session_state.analysis_run = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload":
    from creseq_mcp.processing import mpraflow as _mpraflow
    _use_mpraflow = False  # TEMP: force built-in pipeline for testing

    st.header("Upload CRE-seq Data")
    st.caption("Enter local file paths — no upload needed since the app runs on your machine.")

    # ── File path inputs ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        assoc_path_str = st.text_input(
            "Association FASTQ (barcode read)",
            placeholder="~/Downloads/Association/ENCFF853TAM.fastq.gz",
            help="R2 from your association sequencing run — contains the random barcodes.",
        )
        dna_path_str = st.text_input(
            "DNA Counting FASTQ",
            placeholder="~/Downloads/DNA Reps/ENCFF062NLF.fastq.gz",
            help="Sequencing run from your plasmid pool (separate from association).",
        )
    with col2:
        ref_path_str = st.text_input(
            "Barcode Reference TSV",
            placeholder="~/Downloads/reference_library.tsv",
            help="TSV with columns: oligo_id, barcode, sequence, designed_category",
        )
        rna_paths_str = st.text_input(
            "RNA FASTQs (comma-separated paths)",
            placeholder="~/Downloads/RNA_Counting/ENCFF388KGK.fastq.gz, ~/Downloads/RNA_Counting/ENCFF184MJL.fastq.gz.1",
            help="One path per replicate, separated by commas.",
        )

    with st.expander("Barcode options"):
        bc_col1, bc_col2, bc_col3 = st.columns(3)
        bc_len = bc_col1.number_input("Barcode length (bp)", min_value=6, max_value=30, value=15)
        bc_end = bc_col2.radio("Barcode position", ["3prime", "5prime"], horizontal=True)
        bc_mm  = bc_col3.number_input("Max mismatches", min_value=0, max_value=2, value=0)

    # ── Resolve and validate paths ───────────────────────────────────────────
    def _resolve(s: str) -> Path | None:
        s = s.strip()
        if not s:
            return None
        p = Path(s).expanduser()
        return p if p.exists() else None

    assoc_path = _resolve(assoc_path_str)
    ref_path   = _resolve(ref_path_str)
    dna_path   = _resolve(dna_path_str)
    rna_paths  = [p for s in rna_paths_str.split(",") if (p := _resolve(s)) is not None] if rna_paths_str.strip() else []

    # Show validation feedback
    for label, val, raw in [
        ("Association FASTQ", assoc_path, assoc_path_str),
        ("Reference TSV", ref_path, ref_path_str),
        ("DNA FASTQ", dna_path, dna_path_str),
    ]:
        if raw.strip() and not val:
            st.error(f"{label}: file not found — {raw.strip()}")

    if rna_paths_str.strip():
        for s in rna_paths_str.split(","):
            p = _resolve(s)
            if not p:
                st.error(f"RNA FASTQ not found: {s.strip()}")

    # ── Process button ───────────────────────────────────────────────────────
    ready = assoc_path and ref_path and dna_path and len(rna_paths) > 0
    if not ready:
        missing = [n for n, v in [
            ("Association FASTQ", assoc_path), ("Reference TSV", ref_path),
            ("DNA FASTQ", dna_path), ("RNA FASTQs", rna_paths or None),
        ] if not v]
        if missing:
            st.info(f"Still needed: {', '.join(missing)}")

    if st.button("▶ Process all files", type="primary", use_container_width=True, disabled=not ready):
        from creseq_mcp.processing.pipeline import process_and_save
        from creseq_mcp.processing.counting import process_dna_counting, process_rna_counting
        from creseq_mcp.qc.activity import activity_report

        progress = st.progress(0, text="Step 1/4 — Association…")
        try:
            assoc_stats = process_and_save(
                assoc_path, ref_path, UPLOAD_DIR,
                barcode_len=int(bc_len), barcode_end=bc_end, max_mismatch=int(bc_mm),
            )
            progress.progress(25, text="Step 2/4 — DNA counting…")

            dna_stats = process_dna_counting(
                dna_path, UPLOAD_DIR / "mapping_table.tsv", UPLOAD_DIR,
                barcode_len=int(bc_len), barcode_end=bc_end, max_mismatch=int(bc_mm),
            )
            progress.progress(50, text="Step 3/4 — RNA counting…")

            rna_stats = process_rna_counting(
                rna_paths, UPLOAD_DIR / "mapping_table.tsv", UPLOAD_DIR,
                barcode_len=int(bc_len), barcode_end=bc_end, max_mismatch=int(bc_mm),
            )
            progress.progress(75, text="Step 4/4 — Activity analysis…")

            manifest_path = UPLOAD_DIR / "design_manifest.tsv"
            _, act_summary = activity_report(
                UPLOAD_DIR / "plasmid_counts.tsv",
                UPLOAD_DIR / "rna_counts.tsv",
                manifest_path if manifest_path.exists() else None,
                upload_dir=UPLOAD_DIR,
            )
            progress.progress(100, text="Done!")

            st.success("All steps complete — go to Chat to run QC or QC & Plots to see results.")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Reads (assoc)", f"{assoc_stats['total_reads']:,}")
            c2.metric("Unique barcodes", f"{assoc_stats['unique_barcodes']:,}")
            c3.metric("DNA match rate", f"{dna_stats['match_rate']:.1%}")
            c4.metric("RNA replicates", len(rna_stats["replicates"]))
            c5.metric("Active CREs", f"{act_summary['n_active']:,}")

        except Exception as exc:
            progress.empty()
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
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat":
    st.header("MCP Agent Chat")
    st.caption("The agent dispatches MCP tools based on your request.")

    # render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for tool in msg.get("tools", []):
                st.info(f"🔧 Tool called: `{tool}`")

    # input
    if prompt := st.chat_input("Ask the agent… (e.g. 'run QC', 'find enriched motifs')"):
        st.session_state.messages.append({"role": "user", "content": prompt, "tools": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking…"):
                if _GEMINI_READY and _gemini_available():
                    if st.session_state.gemini_agent is None:
                        st.session_state.gemini_agent = ClaudeQCAgent(
                            os.environ["ANTHROPIC_API_KEY"]
                        )
                    response = st.session_state.gemini_agent.send_message(prompt)
                else:
                    time.sleep(0.4)
                    response = stub_query_agent(prompt, has_data=True)
            st.markdown(response.text)
            for tool in response.tools_called:
                st.info(f"🔧 Tool called: `{tool}`")

        st.session_state.messages.append(
            {"role": "assistant", "content": response.text, "tools": response.tools_called}
        )

    with st.sidebar:
        st.divider()
        if _GEMINI_READY and _gemini_available():
            st.success("Claude connected", icon="✅")
        else:
            st.warning("Set ANTHROPIC_API_KEY to enable real QC tools", icon="⚠️")
        st.divider()
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. How can I help?",
                    "tools": [],
                }
            ]
            if st.session_state.gemini_agent is not None:
                st.session_state.gemini_agent.reset()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QC & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 QC & Plots":
    st.header("QC & Analysis Plots")
    df = st.session_state.data

    tab_qc, tab_activity, tab_motif, tab_variant = st.tabs(
        ["🔬 QC Metrics", "📈 Activity Plots", "🔡 Motif Analysis", "🧪 Variant Effects"]
    )

    # ── QC Metrics ─────────────────────────────────────────────────────────
    with tab_qc:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total elements", f"{len(df):,}")
        col2.metric("Median DNA counts", f"{df['dna_counts'].median():.1f}")
        col3.metric("Median RNA counts", f"{df['rna_counts'].median():.1f}")
        col4.metric("Low-coverage (<5 DNA)", f"{(df['dna_counts'] < 5).sum():,}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="dna_counts", nbins=50,
                title="DNA Count Distribution",
                labels={"dna_counts": "DNA counts"},
                color_discrete_sequence=["#4C78A8"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                df, x="rna_counts", nbins=50,
                title="RNA Count Distribution",
                labels={"rna_counts": "RNA counts"},
                color_discrete_sequence=["#F58518"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df, x="dna_counts", y="rna_counts",
            color="active",
            color_discrete_map={True: "#E45756", False: "#72B7B2"},
            opacity=0.6,
            title="RNA vs. DNA Counts per Element",
            labels={"dna_counts": "DNA counts", "rna_counts": "RNA counts", "active": "Active"},
            log_x=True, log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

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
        st.caption("Enrichment of TF binding motifs in active vs. inactive CREs (HOMER-style).")

        motif_data = pd.DataFrame(
            {
                "Motif": ["SP1", "AP1 (FOSL2)", "NRF1", "CTCF", "YY1", "ETS1"],
                "Fold Enrichment": [2.8, 2.1, 1.9, 1.4, 1.2, 1.1],
                "p-value": [0.0012, 0.0041, 0.0089, 0.031, 0.078, 0.12],
                "Significant": [True, True, True, False, False, False],
            }
        )
        motif_data["-log10(p)"] = -np.log10(motif_data["p-value"])

        fig = px.bar(
            motif_data, x="Motif", y="Fold Enrichment",
            color="Significant",
            color_discrete_map={True: "#E45756", False: "#72B7B2"},
            title="TF Motif Enrichment in Active CREs",
            text="Fold Enrichment",
        )
        fig.update_traces(texttemplate="%{text:.1f}×", textposition="outside")
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="no enrichment")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(motif_data.drop(columns=["-log10(p)"]), use_container_width=True, hide_index=True)

        if "top_motif" in df.columns:
            st.subheader("Motif Hits per Element")
            motif_freq = df.groupby("top_motif")["active"].agg(["sum", "count"]).reset_index()
            motif_freq.columns = ["Motif", "Active hits", "Total hits"]
            motif_freq["Active rate"] = (motif_freq["Active hits"] / motif_freq["Total hits"] * 100).round(1)
            motif_freq = motif_freq.sort_values("Active rate", ascending=False)
            fig = px.bar(
                motif_freq[motif_freq["Motif"] != "None"],
                x="Motif", y="Active rate",
                title="Active Rate by Top Motif (%)",
                color="Active rate",
                color_continuous_scale="RdBu",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Variant Effects ─────────────────────────────────────────────────────
    with tab_variant:
        st.subheader("Allele-Specific Activity")
        st.caption("Regulatory variants with significant allele-specific activity (|Δlog₂| > 0.5, FDR < 10%).")

        variant_data = pd.DataFrame(
            {
                "element_id": [f"CRE_{i:04d}" for i in [42, 107, 183, 201, 256, 289, 299]],
                "Ref log₂": [2.41, 1.88, 2.05, 1.72, 2.31, 1.61, 1.95],
                "Alt log₂": [0.63, 0.91, 1.23, 2.51, 1.02, 0.88, 3.12],
                "Δlog₂": [1.78, 0.97, 0.82, -0.79, 1.29, 0.73, -1.17],
                "FDR": [0.002, 0.018, 0.041, 0.033, 0.007, 0.068, 0.044],
                "eQTL overlap": [True, False, True, True, False, False, True],
            }
        )
        variant_data["Direction"] = variant_data["Δlog₂"].apply(
            lambda x: "Ref > Alt" if x > 0 else "Alt > Ref"
        )

        fig = px.scatter(
            variant_data, x="Ref log₂", y="Alt log₂",
            color="Direction",
            size=variant_data["Δlog₂"].abs(),
            hover_data=["element_id", "Δlog₂", "FDR", "eQTL overlap"],
            title="Allele-Specific Activity: Ref vs. Alt",
            labels={"Ref log₂": "Reference allele log₂(RNA/DNA)", "Alt log₂": "Alternate allele log₂(RNA/DNA)"},
        )
        # diagonal reference line
        lims = [0, 3.5]
        fig.add_trace(
            go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(variant_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Results":
    st.header("Analysis Results")
    df = st.session_state.data

    if not st.session_state.analysis_run:
        st.warning("Analysis has not been run yet. Go to **Upload** and click **Run Analysis**.")

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

    display_cols = [c for c in ["element_id", "chrom", "start", "end", "dna_counts", "rna_counts", "log2_ratio", "pval", "active", "chromatin_state", "top_motif"] if c in df.columns]

    filter_active = st.checkbox("Show active elements only", value=False)
    display_df = df[df["active"]] if filter_active and "active" in df.columns else df

    st.dataframe(
        display_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "active": st.column_config.CheckboxColumn("Active"),
            "log2_ratio": st.column_config.NumberColumn("log₂ ratio", format="%.3f"),
            "pval": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
    )

    st.divider()

    with st.expander("📌 Enrichment Summary"):
        st.markdown(
            """
**Chromatin state enrichment** (active vs. inactive CREs):
- Active Enhancer: OR = 3.2, p < 0.001 ✅
- Promoter-flanking: OR = 1.8, p = 0.021 ✅
- Heterochromatin: OR = 0.2, p < 0.001 (depleted)

**Top enriched TF motifs:**
- SP1 (2.8×), AP1/FOSL2 (2.1×), NRF1 (1.9×)

**Regulatory variants:** 7 elements with allele-specific activity; 3 overlap eQTLs.
"""
        )

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
