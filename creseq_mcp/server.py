"""
creseq_mcp/server.py
====================
MCP server entry point for the CRE-seq analysis toolkit.

Run with::

    python -m creseq_mcp.server
    # or
    mcp run creseq_mcp/server.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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

from creseq_mcp.stats.library import (
    aggregate_fastq_counts_to_elements,
    call_active_elements,
    count_barcodes_from_fastq,
    interpret_literature_evidence,
    literature_search_for_motifs,
    motif_enrichment_summary,
    normalize_activity,
    prepare_rag_context,
    rank_cre_candidates,
    search_encode_tf,
    search_jaspar_motif,
    search_pubmed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path.home() / ".creseq" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
    "creseq-mcp",
    instructions=(
        "CRE-seq library QC toolkit. "
        "File path arguments are optional — omit them to use data uploaded via the UI. "
        "Do NOT use for lentiMPRA or STARR-seq without adjusting thresholds."
    ),
)


def _path(arg: str | None, filename: str) -> str:
    return arg or str(UPLOAD_DIR / filename)


def _summary(result: tuple | dict) -> dict:
    """Extract the summary dict, dropping the DataFrame and coercing numpy scalars."""
    s = result[1] if isinstance(result, tuple) else {
        k: v[1] if isinstance(v, tuple) else v for k, v in result.items()
    }
    return json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def _serialise(result: tuple | dict) -> dict:
    """Convert a (DataFrame, summary) result to JSON-safe rows + summary."""
    import pandas as pd

    if isinstance(result, tuple):
        df, summary = result
        out = {
            "rows": df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else [],
            "summary": summary,
        }
    else:
        out = result

    return json.loads(
        json.dumps(
            out,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------


@mcp.tool()
def tool_barcode_complexity(
    mapping_table_path: str | None = None,
    min_reads_per_barcode: int = 1,
) -> dict:
    """
    Per-oligo barcode count statistics.

    Returns how many distinct barcodes support each designed oligo, what
    fraction are error-free (perfect CIGAR/MD), and the median read depth
    per barcode.  PASS when median barcodes/oligo >= 10.
    """
    return _summary(barcode_complexity(
        _path(mapping_table_path, "mapping_table.tsv"), min_reads_per_barcode
    ))


@mcp.tool()
def tool_oligo_recovery(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
    thresholds: list[int] | None = None,
) -> dict:
    """
    Recovery rate of designed oligos, broken out by designed_category.

    PASS when test_element recovery@10 >= 80% AND positive_control recovery@10 >= 95%.
    """
    return _summary(oligo_recovery(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
        thresholds,
    ))


@mcp.tool()
def tool_synthesis_error_profile(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Per-oligo synthesis error characterisation from CIGAR/MD tags.

    Reports mismatches, indels, soft-clip rates, and Spearman correlation
    between GC content and synthesis fidelity.  PASS when median perfect_fraction >= 0.50.
    """
    return _summary(synthesis_error_profile(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv") if design_manifest_path else None,
    ))


@mcp.tool()
def tool_barcode_collision_analysis(
    mapping_table_path: str | None = None,
    min_read_support: int = 2,
) -> dict:
    """
    Barcodes that map to more than one designed oligo.

    PASS when collision rate < 3%.
    """
    return _summary(barcode_collision_analysis(
        _path(mapping_table_path, "mapping_table.tsv"), min_read_support
    ))


@mcp.tool()
def tool_barcode_uniformity(
    plasmid_count_path: str | None = None,
    min_barcodes_per_oligo: int = 5,
) -> dict:
    """
    Per-oligo barcode abundance evenness in the plasmid pool (Gini coefficient).

    PASS when median Gini < 0.30.
    """
    return _summary(barcode_uniformity(
        _path(plasmid_count_path, "plasmid_counts.tsv"), min_barcodes_per_oligo
    ))


@mcp.tool()
def tool_gc_content_bias(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
    gc_bins: int = 10,
) -> dict:
    """
    Synthesis recovery stratified by oligo GC content.

    Flags GC bins with recovery < 50% of the median bin.  PASS when no dropout bins found.
    """
    return _summary(gc_content_bias(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
        gc_bins,
    ))


@mcp.tool()
def tool_oligo_length_qc(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Synthesis-truncation check comparing observed alignment length to designed length.

    PASS when median fraction_full_length >= 0.80.
    """
    return _summary(oligo_length_qc(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
    ))


@mcp.tool()
def tool_plasmid_depth_summary(plasmid_count_path: str | None = None) -> dict:
    """
    Barcode-level read-count statistics in the plasmid DNA library.

    PASS when median dna_count >= 10 AND fewer than 10% of barcodes have zero counts.
    """
    return _summary(plasmid_depth_summary(_path(plasmid_count_path, "plasmid_counts.tsv")))


@mcp.tool()
def tool_variant_family_coverage(
    mapping_table_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Coverage of CRE-seq variant families (reference + motif knockouts / point mutants).

    PASS when >= 80% of families fully recovered AND zero families missing their reference.
    """
    return _summary(variant_family_coverage(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(design_manifest_path, "design_manifest.tsv"),
    ))


@mcp.tool()
def tool_library_summary_report(
    mapping_table_path: str | None = None,
    plasmid_count_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Comprehensive one-shot CRE-seq library QC report.

    Runs all applicable tools and returns overall_pass, failed_checks, warnings,
    and per-tool summaries.
    """
    return _summary(library_summary_report(
        _path(mapping_table_path, "mapping_table.tsv"),
        _path(plasmid_count_path, "plasmid_counts.tsv"),
        _path(design_manifest_path, "design_manifest.tsv") if design_manifest_path else None,
    ))


@mcp.tool()
def tool_process_library(
    fastq_path: str,
    reference_path: str,
    barcode_len: int = 10,
    barcode_end: str = "3prime",
    max_mismatch: int = 1,
) -> dict:
    """
    Process a raw CRE-seq plasmid-DNA FASTQ against a barcode reference library.

    Writes mapping_table.tsv, plasmid_counts.tsv, and design_manifest.tsv to the
    upload directory so all QC tools can run without additional arguments.

    barcode_end: "3prime" (default) or "5prime".
    """
    from creseq_mcp.processing.pipeline import process_and_save

    return process_and_save(
        fastq_path, reference_path, UPLOAD_DIR,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )

# ---------------------------------------------------------------------------
# Stats tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_normalize_activity(
    count_table_path: str,
    pseudocount: float = 1.0,
    dna_col: str = "dna_counts",
    rna_col: str = "rna_counts",
    element_col: str = "element_id",
) -> dict:
    """
    Compute log2 RNA/DNA activity scores for each CRE.

    Use after QC has produced element-level DNA and RNA count tables.
    """
    return _serialise(
        normalize_activity(
            count_table_path=count_table_path,
            pseudocount=pseudocount,
            dna_col=dna_col,
            rna_col=rna_col,
            element_col=element_col,
        )
    )


@mcp.tool()
def tool_call_active_elements(
    activity_table_path: str,
    activity_col: str = "log2_activity",
    category_col: str = "designed_category",
    negative_control_label: str = "negative_control",
    activity_threshold: float = 1.0,
    fdr_threshold: float = 0.05,
) -> dict:
    """
    Classify CREs as active/inactive using an empirical negative-control background.
    """
    return _serialise(
        call_active_elements(
            activity_table_path=activity_table_path,
            activity_col=activity_col,
            category_col=category_col,
            negative_control_label=negative_control_label,
            activity_threshold=activity_threshold,
            fdr_threshold=fdr_threshold,
        )
    )


@mcp.tool()
def tool_rank_cre_candidates(
    activity_table_path: str,
    top_n: int = 20,
    activity_col: str = "log2_activity",
    q_col: str = "q_value",
) -> dict:
    """
    Rank CRE candidates by activity strength and statistical confidence.
    """
    return _serialise(
        rank_cre_candidates(
            activity_table_path=activity_table_path,
            top_n=top_n,
            activity_col=activity_col,
            q_col=q_col,
        )
    )


@mcp.tool()
def tool_motif_enrichment_summary(
    activity_table_path: str,
    motif_col: str = "top_motif",
    active_col: str = "active",
) -> dict:
    """
    Summarize TF motifs enriched among active CREs.
    """
    return _serialise(
        motif_enrichment_summary(
            activity_table_path=activity_table_path,
            motif_col=motif_col,
            active_col=active_col,
        )
    )


@mcp.tool()
def tool_prepare_rag_context(
    ranked_table_path: str,
    top_n: int = 10,
    motif_col: str = "top_motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
) -> dict:
    """
    Prepare top CREs and TF motif search terms for literature/API interpretation.
    """
    return _serialise(
        prepare_rag_context(
            ranked_table_path=ranked_table_path,
            top_n=top_n,
            motif_col=motif_col,
            target_cell_type=target_cell_type,
            off_target_cell_type=off_target_cell_type,
        )
    )


@mcp.tool()
def tool_count_barcodes_from_fastq(
    fastq_path: str,
    barcode_start: int = 0,
    barcode_length: int = 10,
    max_reads: int | None = None,
) -> dict:
    """
    Count fixed-position barcodes directly from raw FASTQ reads.
    """
    return _serialise(
        count_barcodes_from_fastq(
            fastq_path=fastq_path,
            barcode_start=barcode_start,
            barcode_length=barcode_length,
            max_reads=max_reads,
        )
    )


@mcp.tool()
def tool_aggregate_fastq_counts_to_elements(
    dna_barcode_counts_path: str,
    rna_barcode_counts_path: str,
    barcode_map_path: str,
) -> dict:
    """
    Aggregate DNA/RNA barcode counts to element-level CRE activity values.
    """
    return _serialise(
        aggregate_fastq_counts_to_elements(
            dna_barcode_counts_path=dna_barcode_counts_path,
            rna_barcode_counts_path=rna_barcode_counts_path,
            barcode_map_path=barcode_map_path,
        )
    )


@mcp.tool()
def tool_search_pubmed(
    query: str,
    max_results: int = 5,
    email: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Search PubMed for literature evidence using NCBI E-utilities.
    """
    return _serialise(
        search_pubmed(
            query=query,
            max_results=max_results,
            email=email,
            api_key=api_key,
        )
    )


@mcp.tool()
def tool_search_jaspar_motif(
    tf_name: str,
    species: int = 9606,
    collection: str = "CORE",
    max_results: int = 5,
) -> dict:
    """
    Search JASPAR for TF motif matrix profiles.
    """
    return _serialise(
        search_jaspar_motif(
            tf_name=tf_name,
            species=species,
            collection=collection,
            max_results=max_results,
        )
    )


@mcp.tool()
def tool_search_encode_tf(
    tf_name: str,
    cell_type: str | None = None,
    max_results: int = 5,
) -> dict:
    """
    Search ENCODE for TF/cell-type functional genomics records.
    """
    return _serialise(
        search_encode_tf(
            tf_name=tf_name,
            cell_type=cell_type,
            max_results=max_results,
        )
    )


@mcp.tool()
def tool_literature_search_for_motifs(
    motif_table_path: str,
    motif_col: str = "motif",
    target_cell_type: str | None = None,
    off_target_cell_type: str | None = None,
    top_n_motifs: int = 5,
    max_pubmed_results_per_motif: int = 3,
    max_database_results_per_motif: int = 3,
    email: str | None = None,
    ncbi_api_key: str | None = None,
) -> dict:
    """
    Run PubMed, JASPAR, and ENCODE API searches for top enriched motifs.
    """
    return _serialise(
        literature_search_for_motifs(
            motif_table_path=motif_table_path,
            motif_col=motif_col,
            target_cell_type=target_cell_type,
            off_target_cell_type=off_target_cell_type,
            top_n_motifs=top_n_motifs,
            max_pubmed_results_per_motif=max_pubmed_results_per_motif,
            max_database_results_per_motif=max_database_results_per_motif,
            email=email,
            ncbi_api_key=ncbi_api_key,
        )
    )


@mcp.tool()
def tool_interpret_literature_evidence(
    evidence_table_path: str,
) -> dict:
    """
    Summarize API-retrieved literature/database evidence for display.
    """
    return _serialise(
        interpret_literature_evidence(
            evidence_table_path=evidence_table_path,
        )
    )


if __name__ == "__main__":
    mcp.run()
