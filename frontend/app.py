"""CRE-seq Analysis Tool — Streamlit frontend mockup."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent_stub import query_agent
from mock_data import get_demo_data

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
    st.header("Upload CRE-seq Data")
    st.markdown(
        "Upload your CRE-seq count table. Accepted formats: **CSV**, **TSV**, or **TXT**. "
        "Expected columns: `element_id`, `dna_counts`, `rna_counts` (plus optional metadata)."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "tsv", "txt"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
        try:
            df_uploaded = pd.read_csv(uploaded, sep=sep)
            # auto-compute log2_ratio if missing
            if "log2_ratio" not in df_uploaded.columns:
                if "rna_counts" in df_uploaded.columns and "dna_counts" in df_uploaded.columns:
                    df_uploaded["log2_ratio"] = np.log2(
                        (df_uploaded["rna_counts"] + 1) / (df_uploaded["dna_counts"] + 1)
                    )
            if "active" not in df_uploaded.columns and "log2_ratio" in df_uploaded.columns:
                df_uploaded["active"] = df_uploaded["log2_ratio"] > 1.0

            st.success(f"Loaded **{uploaded.name}** — {len(df_uploaded):,} rows × {len(df_uploaded.columns)} columns")
            st.dataframe(df_uploaded.head(20), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Elements", f"{len(df_uploaded):,}")
            col2.metric("Columns", len(df_uploaded.columns))
            col3.metric(
                "Active (est.)",
                f"{df_uploaded['active'].sum():,}" if "active" in df_uploaded.columns else "—",
            )

            if st.button("✅ Use this file for analysis", type="primary"):
                st.session_state.data = df_uploaded
                st.session_state.data_source = uploaded.name
                st.session_state.analysis_run = False
                st.success("Data loaded. Click **Run Analysis** below or navigate to Chat / QC & Plots.")

        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
    else:
        st.info("No file uploaded — using synthetic demo data (300 elements).")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

    st.divider()
    st.subheader("Run Analysis Pipeline")
    st.markdown(
        "Runs the full MCP pipeline: normalization → activity calling → annotation → motif enrichment."
    )

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        steps = [
            ("Normalizing counts (log₂ RNA/DNA)…", 0.6),
            ("Calling active elements vs. negative controls…", 0.8),
            ("Annotating with chromatin states…", 0.5),
            ("Running motif enrichment…", 1.0),
            ("Computing variant effects…", 0.7),
        ]
        progress = st.progress(0, text="Starting pipeline…")
        for i, (msg, delay) in enumerate(steps):
            progress.progress((i + 1) / len(steps), text=msg)
            time.sleep(delay)
        progress.empty()
        st.session_state.analysis_run = True
        st.success("Pipeline complete! Navigate to **QC & Plots** or **Results**.")


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
                time.sleep(0.4)  # simulate latency
                response = query_agent(prompt, has_data=True)
            st.markdown(response.text)
            for tool in response.tools_called:
                st.info(f"🔧 Tool called: `{tool}`")

        st.session_state.messages.append(
            {"role": "assistant", "content": response.text, "tools": response.tools_called}
        )

    with st.sidebar:
        st.divider()
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. How can I help?",
                    "tools": [],
                }
            ]
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
        n_active = df["active"].sum()
        n_inactive = len(df) - n_active
        col1, col2, col3 = st.columns(3)
        col1.metric("Active CREs", f"{n_active:,}")
        col2.metric("Inactive CREs", f"{n_inactive:,}")
        col3.metric("Activity rate", f"{n_active / len(df) * 100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="log2_ratio",
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
            if "pval" in df.columns:
                volcano_df = df.copy()
                volcano_df["neg_log10_p"] = -np.log10(volcano_df["pval"].clip(lower=1e-10))
                fig = px.scatter(
                    volcano_df, x="log2_ratio", y="neg_log10_p",
                    color="active",
                    color_discrete_map={True: "#E45756", False: "#72B7B2"},
                    opacity=0.6,
                    title="Volcano Plot",
                    labels={
                        "log2_ratio": "log₂ RNA/DNA",
                        "neg_log10_p": "−log₁₀(p-value)",
                        "active": "Active",
                    },
                    hover_data=["element_id"] if "element_id" in df.columns else None,
                )
                fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
                fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload data with a `pval` column to see the volcano plot.")

        if "chromatin_state" in df.columns:
            state_counts = (
                df.groupby(["chromatin_state", "active"])
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
