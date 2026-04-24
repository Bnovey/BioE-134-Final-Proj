"""Claude agent with real CRE-seq library-QC tool calling via the Anthropic SDK."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from creseq_mcp.server import UPLOAD_DIR, _summary
from creseq_mcp.qc.library import (
    barcode_collision_analysis,
    barcode_complexity,
    barcode_uniformity,
    gc_content_bias,
    library_summary_report,
    oligo_length_qc,
    oligo_recovery,
    plasmid_depth_summary,
    synthesis_error_profile,
    variant_family_coverage,
)

_SYSTEM_PROMPT = (
    "You are a CRE-seq library QC assistant backed by real analysis tools. "
    "File path arguments are optional — omit them and the tools will automatically use "
    "whatever data the user has uploaded via the UI. "
    "Summarise results clearly: state PASS/FAIL and highlight any warnings. "
    "These tools cover library-side QC only (before RNA analysis). "
    "For downstream steps (normalisation, activity calling, motif enrichment) tell the user "
    "those tools are not yet implemented."
)

def _p(args: dict, key: str, filename: str) -> str:
    return args.get(key) or str(UPLOAD_DIR / filename)

_TOOLS: list[dict] = [
    {
        "name": "tool_barcode_complexity",
        "description": (
            "Per-oligo barcode count statistics. Returns distinct barcodes per oligo, "
            "fraction error-free, and median read depth. PASS when median barcodes/oligo >= 10."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "min_reads_per_barcode": {"type": "integer", "description": "Minimum reads per barcode (default 1)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_oligo_recovery",
        "description": (
            "Recovery rate of designed oligos by category. "
            "PASS when test_element recovery@10 >= 80% AND positive_control recovery@10 >= 95%."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_synthesis_error_profile",
        "description": (
            "Per-oligo synthesis error characterisation from CIGAR/MD tags. "
            "Reports mismatches, indels, soft-clip rates. PASS when median perfect_fraction >= 0.50."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_barcode_collision_analysis",
        "description": "Barcodes that map to more than one oligo. PASS when collision rate < 3%.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "min_read_support": {"type": "integer", "description": "Minimum read support to count a mapping (default 2)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_barcode_uniformity",
        "description": (
            "Per-oligo barcode abundance evenness in the plasmid pool (Gini coefficient). "
            "PASS when median Gini < 0.30."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
                "min_barcodes_per_oligo": {"type": "integer", "description": "Minimum barcodes per oligo (default 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_gc_content_bias",
        "description": (
            "Synthesis recovery stratified by oligo GC content. "
            "PASS when no GC bins show dropout below 50% of the median bin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
                "gc_bins": {"type": "integer", "description": "Number of GC bins (default 10)"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_oligo_length_qc",
        "description": (
            "Synthesis-truncation check comparing observed alignment length to designed length. "
            "PASS when median fraction_full_length >= 0.80."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_plasmid_depth_summary",
        "description": (
            "Barcode-level read-count statistics in the plasmid DNA library. "
            "PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_variant_family_coverage",
        "description": (
            "Coverage of CRE-seq variant families (reference + knockouts/mutants). "
            "PASS when >= 80% of families fully recovered AND zero families missing their reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file"},
            },
            "required": [],
        },
    },
    {
        "name": "tool_library_summary_report",
        "description": (
            "Comprehensive one-shot CRE-seq library QC report. "
            "Runs all applicable tools. Returns overall_pass, failed_checks, warnings, and per-tool results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping_table_path": {"type": "string", "description": "Path to the mapping table TSV file"},
                "plasmid_count_path": {"type": "string", "description": "Path to the plasmid count TSV file"},
                "design_manifest_path": {"type": "string", "description": "Path to the design manifest TSV file (optional)"},
            },
            "required": [],
        },
    },
]

_DISPATCH: dict[str, Any] = {
    "tool_barcode_complexity": lambda a: barcode_complexity(
        _p(a, "mapping_table_path", "mapping_table.tsv"), a.get("min_reads_per_barcode", 1)
    ),
    "tool_oligo_recovery": lambda a: oligo_recovery(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        a.get("thresholds"),
    ),
    "tool_synthesis_error_profile": lambda a: synthesis_error_profile(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv") if a.get("design_manifest_path") else None,
    ),
    "tool_barcode_collision_analysis": lambda a: barcode_collision_analysis(
        _p(a, "mapping_table_path", "mapping_table.tsv"), a.get("min_read_support", 2)
    ),
    "tool_barcode_uniformity": lambda a: barcode_uniformity(
        _p(a, "plasmid_count_path", "plasmid_counts.tsv"), a.get("min_barcodes_per_oligo", 5)
    ),
    "tool_gc_content_bias": lambda a: gc_content_bias(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
        a.get("gc_bins", 10),
    ),
    "tool_oligo_length_qc": lambda a: oligo_length_qc(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
    ),
    "tool_plasmid_depth_summary": lambda a: plasmid_depth_summary(
        _p(a, "plasmid_count_path", "plasmid_counts.tsv")
    ),
    "tool_variant_family_coverage": lambda a: variant_family_coverage(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv"),
    ),
    "tool_library_summary_report": lambda a: library_summary_report(
        _p(a, "mapping_table_path", "mapping_table.tsv"),
        _p(a, "plasmid_count_path", "plasmid_counts.tsv"),
        _p(a, "design_manifest_path", "design_manifest.tsv") if a.get("design_manifest_path") else None,
    ),
}


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str] = field(default_factory=list)


class ClaudeQCAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._messages: list[dict] = []

    def send_message(self, prompt: str) -> AgentResponse:
        self._messages.append({"role": "user", "content": prompt})
        tools_called: list[str] = []

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=self._messages,
            )
            self._messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tools_called.append(block.name)
                try:
                    content = json.dumps(_summary(_DISPATCH[block.name](block.input)))
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

            self._messages.append({"role": "user", "content": tool_results})

        text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )
        return AgentResponse(text=text, tools_called=tools_called)

    def reset(self) -> None:
        self._messages = []


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
