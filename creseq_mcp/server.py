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

UPLOAD_DIR = Path.home() / "Desktop" / "creseq_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
    "creseq-mcp",
    instructions=(
        "CRE-seq library QC toolkit. "
        "File path arguments are optional — omit them to use data uploaded via the UI. "
        "Do NOT use for lentiMPRA or STARR-seq without adjusting thresholds."
    ),
)

_PAPERS_DIR = Path(__file__).parent / "data" / "papers"


@mcp.resource("paper://agarwal2025-lentimpra")
def paper_agarwal2025() -> str:
    """
    Agarwal et al. 2025, Nature — 'Massively parallel characterization of
    transcriptional regulatory elements'. DOI: 10.1038/s41586-024-08430-9

    Large-scale lentiMPRA of >680,000 cCREs across HepG2, K562 and WTC11 cells.
    This is the primary reference for the ENCODE HepG2 lentiMPRA dataset
    (ENCSR463IRX) used by this pipeline.
    """
    return (_PAPERS_DIR / "agarwal2025_lentimpra.txt").read_text()


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
        _path(design_manifest_path, "design_manifest.tsv"),
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


@mcp.tool(name="call_active_elements")
def tool_call_active_elements_full(
    activity_table_path: str,
    negative_controls: list[str],
    fdr_threshold: float = 0.05,
    method: str = "empirical",
    count_table_path: str | None = None,
) -> dict:
    """
    Classify CRE-seq elements as active vs. inactive against a negative-control
    null distribution.  Outputs per-element pvalue, BH-corrected FDR, z-score,
    and fold-over-controls; writes <activity_table>_classified.tsv to disk.

    method='empirical' (default) uses median/MAD on log2 RNA/DNA activities.
    method='glm' will use a negative-binomial GLM on raw counts (not yet
    implemented; pass count_table_path when enabled).
    """
    from creseq_mcp.activity_calling import call_active_elements as _call

    return _serialise(
        _call(
            activity_table_path=activity_table_path,
            negative_controls=negative_controls,
            fdr_threshold=fdr_threshold,
            method=method,
            count_table_path=count_table_path,
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


@mcp.tool()
def tool_process_dna_counting(
    fastq_path: str,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """
    Count DNA barcodes from a plasmid-pool FASTQ → overwrites plasmid_counts.tsv.

    Requires mapping_table.tsv from the association step.
    barcode_end: "3prime" (default) or "5prime".
    """
    from creseq_mcp.processing.counting import process_dna_counting

    return process_dna_counting(
        fastq_path,
        str(UPLOAD_DIR / "mapping_table.tsv"),
        UPLOAD_DIR,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )


@mcp.tool()
def tool_process_rna_counting(
    fastq_paths: list[str],
    rep_names: list[str] | None = None,
    barcode_len: int = 20,
    barcode_end: str = "3prime",
    max_mismatch: int = 0,
) -> dict:
    """
    Count RNA barcodes across one or more replicate FASTQs → writes rna_counts.tsv.

    Requires mapping_table.tsv from the association step.
    fastq_paths: list of FASTQ paths, one per replicate.
    rep_names: optional list of replicate labels (default: rep1, rep2, …).
    """
    from creseq_mcp.processing.counting import process_rna_counting

    return process_rna_counting(
        fastq_paths,
        str(UPLOAD_DIR / "mapping_table.tsv"),
        UPLOAD_DIR,
        rep_names=rep_names,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )


@mcp.tool()
def tool_activity_report(
    dna_counts_path: str | None = None,
    rna_counts_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Normalize DNA/RNA counts → compute log2(RNA/DNA) per oligo → call active CREs.

    Saves activity_results.tsv to the upload directory.
    Uses z-test vs. negative controls when available; falls back to log2 > 1 threshold.
    """
    from creseq_mcp.qc.activity import activity_report

    _, summary = activity_report(
        _path(dna_counts_path, "plasmid_counts.tsv"),
        _path(rna_counts_path, "rna_counts.tsv"),
        _path(design_manifest_path, "design_manifest.tsv") if design_manifest_path else None,
        upload_dir=UPLOAD_DIR,
    )
    return summary


@mcp.tool(name="extract_sequences")
def tool_extract_sequences(
    classified_table: str,
    sequence_source: str,
    active_output: str = "active.fa",
    background_output: str = "background.fa",
) -> dict:
    """
    Bridge ``call_active_elements`` → ``motif_enrichment``.

    Reads a classified-elements TSV (with ``element_id``, ``active``,
    ``pvalue`` columns) and a sequence-source TSV (with ``element_id`` +
    ``sequence``) and writes two FASTAs: actives, and inactive test elements
    as background.  Negative controls (NaN pvalue) are excluded from both.
    Returns paths plus per-set counts.
    """
    from creseq_mcp.motif import extract_sequences_to_fasta

    return extract_sequences_to_fasta(
        classified_table=classified_table,
        sequence_source=sequence_source,
        active_output=active_output,
        background_output=background_output,
    )


@mcp.tool()
def tool_motif_enrichment(
    active_fasta: str,
    background_fasta: str,
    motif_database: str = "JASPAR2024",
    collection: str = "CORE",
    tax_group: str = "Vertebrates",
    score_threshold: float = 0.8,
    output_path: str | None = None,
) -> dict:
    """
    Test for TF binding motif enrichment in active CRE-seq elements.

    Scans active and background FASTA sequences against JASPAR motif PWMs on
    both strands and tests each motif for enrichment with one-sided Fisher's
    exact + BH-FDR.  Returns the enrichment table path and a summary of the
    top significant motifs.
    """
    from creseq_mcp.motif import motif_enrichment

    return motif_enrichment(
        active_fasta=active_fasta,
        background_fasta=background_fasta,
        motif_database=motif_database,
        collection=collection,
        tax_group=tax_group,
        score_threshold=score_threshold,
        output_path=output_path,
    )


@mcp.tool()
def tool_plot_creseq(
    data_file: str,
    plot_type: str,
    output_path: str = "plot.png",
    highlight_ids: list[str] | None = None,
    neg_control_ids: list[str] | None = None,
    annotation_file: str | None = None,
) -> dict:
    """
    Generate a publication-quality CRE-seq plot.

    plot_type ∈ {volcano, ranked_activity, replicate_correlation,
    annotation_boxplot, motif_dotplot}.  Returns the path to the saved
    figure plus a natural-language description of what it shows.
    """
    from creseq_mcp.plotting import plot_creseq

    return plot_creseq(
        data_file=data_file,
        plot_type=plot_type,
        output_path=output_path,
        highlight_ids=highlight_ids,
        neg_control_ids=neg_control_ids,
        annotation_file=annotation_file,
    )


@mcp.tool()
def tool_variant_delta_scores(
    activity_results_path: str | None = None,
    fdr_threshold: float = 0.05,
) -> dict:
    """
    Compute per-variant delta log₂ activity scores (mutant − reference).

    For each variant family in the activity results, subtracts the reference
    element's log₂(RNA/DNA) from each mutant to get Δlog₂. Applies
    Benjamini–Hochberg FDR correction and flags significant hits.

    Requires activity_results.tsv to have a variant_family column
    (or an element_id/oligo_id following the {family}_{allele} naming
    convention) and an is_reference column (or elements ending in '_ref').

    Returns a summary dict and writes variant_delta_scores.tsv to the
    output directory.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats as _scipy_stats
    from creseq_mcp.qc.activity import _bh_fdr

    act_path = Path(_path(activity_results_path, "activity_results.tsv"))
    if not act_path.exists():
        return {"error": f"activity_results.tsv not found at {act_path}"}

    df = pd.read_csv(act_path, sep="\t")
    id_col = "element_id" if "element_id" in df.columns else "oligo_id"

    if "variant_family" not in df.columns:
        df["variant_family"] = df[id_col].astype(str).str.rsplit("_", n=1).str[0]
    if "is_reference" not in df.columns:
        df["is_reference"] = df[id_col].astype(str).str.lower().str.endswith("_ref")

    ref_df = df[df["is_reference"]][["variant_family", "log2_ratio"]].rename(
        columns={"log2_ratio": "ref_log2"}
    )
    mut_df = df[~df["is_reference"]].copy()
    delta_df = mut_df.merge(ref_df, on="variant_family", how="inner")
    delta_df["delta_log2"] = delta_df["log2_ratio"] - delta_df["ref_log2"]
    delta_df.rename(columns={"log2_ratio": "mutant_log2"}, inplace=True)

    if len(delta_df) == 0:
        return {"error": "No mutant–reference pairs found. Check variant_family and is_reference columns."}

    mean_d = float(delta_df["delta_log2"].mean())
    std_d = max(float(delta_df["delta_log2"].std()), 1e-9)
    z_scores = [(d - mean_d) / std_d for d in delta_df["delta_log2"]]
    pvals = [float(2 * _scipy_stats.norm.sf(abs(z))) for z in z_scores]
    delta_df["pval"] = pvals
    delta_df["fdr"] = _bh_fdr(pvals)
    delta_df["significant"] = delta_df["fdr"] < fdr_threshold

    out_path = UPLOAD_DIR / "variant_delta_scores.tsv"
    delta_df.to_csv(out_path, sep="\t", index=False)

    n_sig = int(delta_df["significant"].sum())
    top5 = delta_df.nlargest(5, "delta_log2")[[id_col, "variant_family", "delta_log2", "fdr"]].to_dict("records")
    return {
        "n_pairs": len(delta_df),
        "n_significant": n_sig,
        "n_families": int(delta_df["variant_family"].nunique()),
        "median_delta_log2": float(delta_df["delta_log2"].median()),
        "top_gain_variants": top5,
        "output_path": str(out_path),
    }


@mcp.tool()
def tool_export_qc_html(
    mapping_table_path: str | None = None,
    plasmid_count_path: str | None = None,
    design_manifest_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run all library QC checks and export a self-contained HTML report.

    Runs library_summary_report(), then generates an HTML file with:
    - Overall pass/fail banner
    - Per-check summary table (check name, pass, key metrics)
    - One section per check with its scalar result values

    The HTML is self-contained and can be opened in any browser without
    additional dependencies.

    Returns the path to the saved HTML report and the overall pass/fail status.
    """
    import html as _html

    mt = _path(mapping_table_path, "mapping_table.tsv")
    pc = _path(plasmid_count_path, "plasmid_counts.tsv")
    dm = _path(design_manifest_path, "design_manifest.tsv") if (
        design_manifest_path or (UPLOAD_DIR / "design_manifest.tsv").exists()
    ) else None

    if not Path(mt).exists():
        return {"error": f"mapping_table.tsv not found at {mt}"}
    if not Path(pc).exists():
        return {"error": f"plasmid_counts.tsv not found at {pc}"}

    qc_results = library_summary_report(mt, pc, dm)
    _, report_summary = qc_results["_report"]
    overall_pass = report_summary.get("overall_pass", False)
    failed = report_summary.get("failed_checks", [])

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

    banner_color = "#27ae60" if overall_pass else "#e74c3c"
    banner_text = "✅ PASS — all checks passed" if overall_pass else f"❌ FAIL — failed: {', '.join(failed)}"

    rows_html = ""
    detail_sections = ""
    for tool_key, label in tool_labels.items():
        res = qc_results.get(tool_key, (None, {}))[1]
        passed = res.get("pass", None)
        badge = "✅" if passed else ("❌" if passed is False else "⚪")
        rows_html += f"<tr><td>{_html.escape(label)}</td><td>{badge}</td></tr>\n"

        scalar_items = {k: v for k, v in res.items() if k not in ("pass",) and not isinstance(v, (dict, list))}
        if scalar_items:
            items_html = "".join(f"<li><b>{_html.escape(str(k))}:</b> {_html.escape(str(v))}</li>" for k, v in scalar_items.items())
            detail_sections += f"<h3>{badge} {_html.escape(label)}</h3><ul>{items_html}</ul>\n"

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CRE-seq Library QC Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
.banner{{background:{banner_color};color:white;padding:16px;border-radius:6px;font-size:1.2em;margin-bottom:24px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f4f4f4}}h2{{margin-top:32px}}</style></head>
<body>
<h1>CRE-seq Library QC Report</h1>
<div class="banner">{banner_text}</div>
<h2>Summary</h2>
<table><tr><th>Check</th><th>Result</th></tr>{rows_html}</table>
<h2>Details</h2>{detail_sections}
</body></html>"""

    out = Path(output_path) if output_path else UPLOAD_DIR / "qc_report.html"
    out.write_text(html_content)
    return {
        "overall_pass": overall_pass,
        "failed_checks": failed,
        "report_path": str(out),
        "n_checks_run": len(tool_labels),
    }


if __name__ == "__main__":
    mcp.run()
