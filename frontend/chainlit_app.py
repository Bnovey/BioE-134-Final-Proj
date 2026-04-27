"""
Chainlit frontend for creseq-mcp.

Connects to the FastMCP server via stdio, exposes all 25 bioinformatics tools
to Claude, renders tool calls as visible steps, and displays Plotly charts inline.

Start:  chainlit run frontend/chainlit_app.py
"""

import json
import math
import os
from pathlib import Path

import chainlit as cl
import plotly.graph_objects as go
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = str(Path(__file__).parent.parent)
PAPER_PATH = (
    Path(PROJECT_ROOT) / "creseq_mcp" / "data" / "papers" / "agarwal2025_lentimpra.txt"
)
DEFAULT_OUTPUTS = str(Path.home() / "Desktop" / "creseq_outputs")

anthropic_client = AsyncAnthropic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_paper_excerpt(max_chars: int = 4000) -> str:
    try:
        return PAPER_PATH.read_text()[:max_chars]
    except FileNotFoundError:
        return "(Agarwal 2025 paper file not found)"


def _build_system_prompt(paper_excerpt: str) -> str:
    return f"""You are a bioinformatics assistant specialized in lentiMPRA/CRE-seq analysis.
You help researchers analyze ENCODE genomic data (HepG2 pilot, ENCSR463IRX) using the
creseq-mcp pipeline.

## Scientific Context — Agarwal et al. 2025, Nature

{paper_excerpt}

---

## Pipeline Overview

Default output directory: {DEFAULT_OUTPUTS}/

**Common workflows:**
- "run library QC"      → tool_library_summary_report (uses defaults if no paths given)
- "call active elements"→ tool_call_active_elements_full or tool_call_active_elements
- "annotate motifs"     → tool_motif_enrichment_summary, then optionally tool_literature_search_for_motifs
- "show volcano plot"   → tool_plot_creseq with plot_type="volcano"
- "rank top CREs"       → tool_rank_cre_candidates

When no file paths are provided, assume files live in {DEFAULT_OUTPUTS}/.
After each tool call, explain results in plain language. Chain tools when the task
requires it (e.g., motif enrichment → literature search → summary).
"""


async def _maybe_render_chart(tool_name: str, result_text: str) -> None:
    """Parse tool result JSON and render an appropriate inline chart."""
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return

    fig = None

    # --- Library QC summary ---
    if tool_name == "tool_library_summary_report":
        overall = data.get("overall_pass")
        if overall is not None:
            status = "PASS ✓" if overall else "FAIL ✗"
            failed = data.get("failed_checks", [])
            warnings = data.get("warnings", [])
            lines = [f"**QC Status: {status}**"]
            if failed:
                lines.append("\n**Failed checks:**")
                lines.extend(f"  - {c}" for c in failed)
            if warnings:
                lines.append("\n**Warnings:**")
                lines.extend(f"  - {w}" for w in warnings)
            if not failed and not warnings:
                lines.append("All checks passed.")
            await cl.Message(content="\n".join(lines)).send()
        return

    # --- Activity table → volcano plot ---
    rows = None
    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data

    if rows and len(rows) > 0 and isinstance(rows[0], dict):
        sample = rows[0]
        if "mean_activity" in sample and "pvalue" in sample:
            x = [r.get("mean_activity", 0) for r in rows]
            pvals = [max(r.get("pvalue", 1), 1e-300) for r in rows]
            y = [-math.log10(p) for p in pvals]
            colors = [
                "#e74c3c" if r.get("active") else "#95a5a6"
                for r in rows
            ]
            hover = [r.get("element_id", "") for r in rows]
            fig = go.Figure(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=colors, size=5, opacity=0.7),
                    text=hover,
                    hovertemplate="%{text}<br>log₂=%{x:.2f}<br>-log₁₀p=%{y:.2f}<extra></extra>",
                )
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.6)
            fig.add_hline(
                y=-math.log10(0.05), line_dash="dash", line_color="gray", opacity=0.6
            )
            fig.update_layout(
                title="Volcano Plot — CRE Activity",
                xaxis_title="log₂ Activity",
                yaxis_title="-log₁₀(p-value)",
                height=520,
                margin=dict(t=50, b=50),
            )

        elif "mean_activity" in sample:
            # Ranked activity bar (no p-values)
            top = sorted(rows, key=lambda r: r.get("mean_activity", 0), reverse=True)[:25]
            ids = [r.get("element_id", str(i)) for i, r in enumerate(top)]
            acts = [r.get("mean_activity", 0) for r in top]
            fig = go.Figure(
                go.Bar(
                    x=acts,
                    y=ids,
                    orientation="h",
                    marker_color="#3498db",
                )
            )
            fig.update_layout(
                title="Top CRE Candidates by Activity",
                xaxis_title="log₂ Activity",
                height=max(300, len(ids) * 22 + 100),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=160),
            )

    # --- Motif enrichment summary ---
    if tool_name == "tool_motif_enrichment_summary" and isinstance(data, dict):
        motifs = data.get("top_motifs", data.get("motifs", []))[:15]
        if motifs:
            names = [
                m.get("motif", m.get("tf_name", m.get("motif_id", f"motif_{i}")))
                for i, m in enumerate(motifs)
            ]
            ratios = [
                m.get("enrichment_ratio", m.get("odds_ratio", 0)) for m in motifs
            ]
            colors = ["#e74c3c" if r >= 2 else "#3498db" for r in ratios]
            fig = go.Figure(
                go.Bar(
                    x=ratios,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig.update_layout(
                title="Top Enriched TF Motifs",
                xaxis_title="Enrichment Ratio",
                height=max(300, len(names) * 25 + 100),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140),
            )

    # --- Ranked CRE candidates ---
    if tool_name == "tool_rank_cre_candidates" and isinstance(data, list) and fig is None:
        top = data[:20]
        if top and "mean_activity" in top[0]:
            ids = [r.get("element_id", str(i)) for i, r in enumerate(top)]
            acts = [r.get("mean_activity", 0) for r in top]
            fig = go.Figure(
                go.Bar(x=acts, y=ids, orientation="h", marker_color="#2ecc71")
            )
            fig.update_layout(
                title="Top CRE Candidates",
                xaxis_title="log₂ Activity",
                height=max(300, len(ids) * 22 + 100),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=160),
            )

    # --- PNG from tool_plot_creseq ---
    if tool_name == "tool_plot_creseq" and isinstance(data, dict) and "plot_path" in data:
        plot_path = data["plot_path"]
        if os.path.exists(plot_path):
            description = data.get("description", "Plot generated.")
            await cl.Message(
                content=description,
                elements=[cl.Image(path=plot_path, name="creseq_plot", display="inline")],
            ).send()
        return

    if fig is not None:
        await cl.Message(
            content="",
            elements=[cl.Plotly(figure=fig, display="inline", size="large")],
        ).send()


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "creseq_mcp.server"],
        cwd=PROJECT_ROOT,
    )

    # Keep context managers open across the session lifetime
    read_write_cm = stdio_client(server_params)
    read, write = await read_write_cm.__aenter__()

    session_cm = ClientSession(read, write)
    mcp_session = await session_cm.__aenter__()
    await mcp_session.initialize()

    tools_result = await mcp_session.list_tools()
    anthropic_tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in tools_result.tools
    ]

    paper_excerpt = _load_paper_excerpt()
    system_prompt = _build_system_prompt(paper_excerpt)

    cl.user_session.set("read_write_cm", read_write_cm)
    cl.user_session.set("session_cm", session_cm)
    cl.user_session.set("mcp_session", mcp_session)
    cl.user_session.set("anthropic_tools", anthropic_tools)
    cl.user_session.set("system_prompt", system_prompt)
    cl.user_session.set("messages", [])

    n = len(anthropic_tools)
    await cl.Message(
        content=(
            f"**CRE-seq Analysis Assistant** ready. "
            f"MCP server connected with **{n} tools**.\n\n"
            "Try:\n"
            '- `"run library QC"`\n'
            '- `"call active elements"`\n'
            '- `"annotate motifs"`\n'
            '- `"show volcano plot"`\n'
            '- `"rank top CREs"`\n'
            '- `"what did Agarwal 2025 find about HepG2 enhancers?"`\n'
        )
    ).send()


@cl.on_chat_end
async def end() -> None:
    for key in ("session_cm", "read_write_cm"):
        cm = cl.user_session.get(key)
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


@cl.on_message
async def on_message(message: cl.Message) -> None:
    mcp_session: ClientSession = cl.user_session.get("mcp_session")
    anthropic_tools: list = cl.user_session.get("anthropic_tools", [])
    system_prompt: str = cl.user_session.get("system_prompt", "")
    messages: list = cl.user_session.get("messages", [])

    messages.append({"role": "user", "content": message.content})

    while True:
        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            if text:
                await cl.Message(content=text).send()
            messages.append(
                {"role": "assistant", "content": [b.model_dump() for b in response.content]}
            )
            break

        if response.stop_reason == "tool_use":
            messages.append(
                {"role": "assistant", "content": [b.model_dump() for b in response.content]}
            )
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                async with cl.Step(name=block.name, type="tool") as step:
                    step.input = block.input
                    try:
                        result = await mcp_session.call_tool(block.name, block.input)
                        result_text = (
                            result.content[0].text if result.content else "{}"
                        )
                    except Exception as exc:
                        result_text = json.dumps({"error": str(exc)})
                    # Truncate display only — pass full text to model
                    step.output = result_text[:3000]

                await _maybe_render_chart(block.name, result_text)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason (e.g., max_tokens)
            break

    cl.user_session.set("messages", messages)
