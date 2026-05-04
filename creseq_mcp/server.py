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
import os
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
    variant_family_coverage,
)

from creseq_mcp.literature.search import (
    interpret_literature_evidence,
    literature_search_for_motifs,
    motif_enrichment_summary,
    prepare_literature_rag_context,
    prepare_rag_context,
    rank_cre_candidates,
    search_encode_tf,
    search_jaspar_motif,
    search_pubmed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.environ.get("CRESEQ_UPLOAD_DIR", Path.home() / ".creseq" / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
    "creseq-mcp",
    instructions=(
        "Full CRE-seq analysis pipeline: library QC, barcode-to-oligo association, "
        "DNA/RNA counting, activity scoring, motif enrichment, variant delta scores, "
        "plotting, and RAG-based literature search. "
        "File path arguments are optional — omit them to use files already in the session. "
        "QC thresholds are calibrated for CRE-seq; adjust before applying to lentiMPRA or STARR-seq."
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
def tool_run_association(
    fastq_r1: str,
    design_fasta: str,
    fastq_r2: str | None = None,
    fastq_bc: str | None = None,
    labels_path: str | None = None,
    min_cov: int = 3,
    min_frac: float = 0.5,
    mapq_threshold: int = 20,
) -> dict:
    """
    Run the full association step: mappy alignment + STARCODE clustering.

    Maps R1 reads to the design FASTA, links barcodes, clusters with STARCODE,
    and writes mapping_table.tsv, plasmid_counts.tsv, design_manifest.tsv to
    CRESEQ_ASSOC_DIR (default ~/creseq_outputs).

    fastq_r1: R1 oligo reads FASTQ (required).
    design_fasta: FASTA of designed oligo sequences (required).
    fastq_r2: optional R2 paired-end reads.
    fastq_bc: optional barcode index FASTQ (ENCODE i5 format).
    labels_path: optional TSV with oligo_id + designed_category columns.
    """
    from creseq_mcp.association.association import run_association
    import shutil

    assoc_dir = Path(os.environ.get("CRESEQ_ASSOC_DIR", Path.home() / "creseq_outputs"))
    assoc_dir.mkdir(parents=True, exist_ok=True)

    result = run_association(
        fastq_r1=fastq_r1,
        design_fasta=design_fasta,
        outdir=assoc_dir,
        fastq_r2=fastq_r2,
        fastq_bc=fastq_bc,
        labels_path=labels_path,
        min_cov=min_cov,
        min_frac=min_frac,
        mapq_threshold=mapq_threshold,
    )

    for fname in ("mapping_table.tsv", "plasmid_counts.tsv", "design_manifest.tsv"):
        src = assoc_dir / fname
        if src.exists():
            shutil.copy(src, UPLOAD_DIR / fname)

    return result


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
    from creseq_mcp.activity.counting import process_dna_counting

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
    from creseq_mcp.activity.counting import process_rna_counting

    return process_rna_counting(
        fastq_paths,
        str(UPLOAD_DIR / "mapping_table.tsv"),
        UPLOAD_DIR,
        rep_names=rep_names,
        barcode_len=barcode_len,
        barcode_end=barcode_end,
        max_mismatch=max_mismatch,
    )


# ---------------------------------------------------------------------------
# Stats tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_rank_cre_candidates(
    activity_table_path: str | None = None,
    top_n: int = 20,
    activity_col: str = "log2_ratio",
    q_col: str = "fdr",
) -> dict:
    """
    Rank CRE candidates by activity strength and statistical confidence.
    """
    return _serialise(
        rank_cre_candidates(
            activity_table_path=_path(activity_table_path, "activity_results.tsv"),
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
    species: str = "human",
    top_n_motifs: int = 5,
    max_pubmed_results_per_motif: int = 3,
    max_database_results_per_motif: int = 3,
    email: str | None = None,
    ncbi_api_key: str | None = None,
    output_path: str | None = None,
    multi_intent_queries: bool = True,
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
            species=species,
            top_n_motifs=top_n_motifs,
            max_pubmed_results_per_motif=max_pubmed_results_per_motif,
            max_database_results_per_motif=max_database_results_per_motif,
            email=email,
            ncbi_api_key=ncbi_api_key,
            output_path=output_path or str(UPLOAD_DIR / "literature_evidence.tsv"),
            multi_intent_queries=multi_intent_queries,
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
def tool_prepare_literature_rag_context(
    evidence_table_path: str | None = None,
    max_records: int = 8,
    min_score: float = 4.0,
    max_context_chars: int = 700,
    output_path: str | None = None,
    max_per_tf: int = 5,
) -> dict:
    """
    Build citation-ready context chunks from literature_evidence.tsv for RAG.

    Filters low-scoring evidence, creates source IDs/citations, extracts compact
    title/abstract/database snippets, and writes literature_rag_context.tsv by
    default.
    """
    return _serialise(
        prepare_literature_rag_context(
            evidence_table_path=_path(evidence_table_path, "literature_evidence.tsv"),
            max_records=max_records,
            min_score=min_score,
            max_context_chars=max_context_chars,
            output_path=output_path or str(UPLOAD_DIR / "literature_rag_context.tsv"),
            max_per_tf=max_per_tf,
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
    from creseq_mcp.activity.counting import process_dna_counting

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
    from creseq_mcp.activity.counting import process_rna_counting

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
    from creseq_mcp.activity.normalize import activity_report

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
    gc_match: bool = False,
    gc_bin_size: float = 0.05,
    n_per_active: int = 1,
    random_state: int = 0,
) -> dict:
    """
    Split classified CRE-seq elements into active and background FASTA files
    suitable for motif enrichment.

    Bridges call_active_elements -> motif_enrichment by translating the
    classification table into the two-FASTA contrast that Fisher's exact
    enrichment requires. Crucially, *negative controls are excluded from
    both files* — they defined the null and would bias the contrast if
    placed in the background.

    Selection rules:
        active.fa     : rows where active == True
        background.fa : rows where active == False AND pvalue is not NaN
                        (i.e. inactive *test* elements; controls have NaN
                        p-values and are filtered out here)

    Args:
        classified_table: Path to a TSV produced by call_active_elements
            (must contain `element_id`, `active`, `pvalue` columns).
        sequence_source: Path to a TSV with `element_id` and `sequence`
            columns — typically the design manifest. The function auto-renames
            `oligo_id` -> `element_id` if needed.
        active_output: Path for the active sequences FASTA. Default "active.fa".
        background_output: Path for the background FASTA. Default "background.fa".
        gc_match: When True, sub-sample the background pool so its GC
            distribution matches the active set bin-for-bin. Defaults to
            False (every inactive test element used as background). Enable
            this to remove spurious enrichment of GC-rich motifs (SP1, EGR1,
            KLF family) that can ride along when active CREs are GC-richer
            than random DNA.
        gc_bin_size: GC-bin width for matching, in [0, 1]. Default 0.05 (20
            bins). Smaller bins = stricter matching but a smaller usable
            background pool.
        n_per_active: Number of background sequences drawn per active
            sequence within the matching bin. Default 1 (one-to-one).
        random_state: Seed for reproducible sampling. Default 0.

    Returns:
        dict with keys:
            active_fasta (str): path written for active sequences
            background_fasta (str): path written for background sequences
            n_active (int): number of records in active.fa
            n_background (int): number of records in background.fa
            gc_matched (bool): whether GC matching was applied

    Notes:
        - FASTA records use `element_id` as the header and the full sequence
          on a single line (no line wrapping).
        - Element IDs missing from the sequence source emit a UserWarning and
          are skipped — the function does not crash on partial coverage.
        - The active vs. background sets are disjoint by construction.
        - When gc_match=True, a UserWarning fires if a GC bin needed a
          nearest-bin fallback or sampling-with-replacement.
    """
    from creseq_mcp.motifs.enrichment import extract_sequences_to_fasta

    return extract_sequences_to_fasta(
        classified_table=classified_table,
        sequence_source=sequence_source,
        active_output=active_output,
        background_output=background_output,
        gc_match=gc_match,
        gc_bin_size=gc_bin_size,
        n_per_active=n_per_active,
        random_state=random_state,
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
    Identify transcription factor binding motifs enriched in active CRE-seq
    elements relative to inactive ones.

    Loads position weight matrices from JASPAR, scans both strands of every
    sequence with a log-odds PSSM, and tests each motif for differential
    occurrence between the active and background sets using one-sided
    Fisher's exact (alternative='greater'). P-values are corrected to FDRs
    via Benjamini-Hochberg. The result tells you which TFs likely drive the
    observed regulatory activity.

    Algorithm:
        1. Load motif PWMs from JASPAR (default: 879 vertebrate CORE motifs)
        2. Build log-odds PSSMs (Biopython); for each sequence, scan forward
           and reverse strands at the given relative-score threshold
        3. Per motif build a 2x2 contingency table:
                            hits   no-hits
              active     |   a   |   b
              background |   c   |   d
        4. Fisher's exact, alternative='greater'
        5. BH-FDR adjustment across all tested motifs
        6. Sort by [fdr ASC, odds_ratio DESC]; OR is capped at 999.0

    Args:
        active_fasta: Path to FASTA of active sequences (from extract_sequences).
        background_fasta: Path to FASTA of background sequences.
        motif_database: pyjaspar release tag. Default "JASPAR2024".
        collection: JASPAR collection name. Default "CORE" (curated, non-redundant).
        tax_group: Taxonomic group filter. Default "Vertebrates".
            Other valid values include "Plants", "Insects", "Fungi", "Nematodes".
        score_threshold: PSSM relative-score cutoff in [0, 1]. 1.0 = perfect
            consensus match; 0.7 is permissive; 0.8 (default) is balanced;
            0.85-0.9 is conservative. Lower thresholds increase sensitivity
            but inflate background hits.
        output_path: Optional TSV destination. Defaults to motif_enrichment.tsv
            in the upload directory.

    Returns:
        dict with keys:
            enrichment_table (str): path to the output TSV with columns
                motif_id, tf_name, n_active_hits, n_background_hits,
                n_active_total, n_background_total, odds_ratio, pvalue, fdr.
            summary (str): natural-language summary naming the top
                significant motifs (FDR < 0.05).

    Notes:
        - Both strands are scanned because TF binding sites are
          orientation-independent on duplex DNA.
        - Empty input FASTAs produce an empty enrichment table with the
          schema preserved (no crash).
        - Sequences shorter than a motif's PWM length are skipped silently
          for that motif.
    """
    from creseq_mcp.motifs.enrichment import motif_enrichment

    return motif_enrichment(
        active_fasta=active_fasta,
        background_fasta=background_fasta,
        motif_database=motif_database,
        collection=collection,
        tax_group=tax_group,
        score_threshold=score_threshold,
        output_path=output_path or str(UPLOAD_DIR / "motif_enrichment.tsv"),
    )


@mcp.tool()
def tool_plot_creseq(
    plot_type: str,
    data_file: str | None = None,
    output_path: str | None = None,
    highlight_ids: list[str] | None = None,
    neg_control_ids: list[str] | None = None,
    annotation_file: str | None = None,
) -> dict:
    """
    Generate publication-quality CRE-seq figures from analysis output tables.

    A single dispatcher routing to five plot types — pick whichever answers
    the question you have:
        - volcano             : effect-size vs. significance
        - ranked_activity     : per-element activity, sorted
        - replicate_correlation : reproducibility between replicates
        - annotation_boxplot  : activity stratified by user-supplied category
        - motif_dotplot       : top enriched TF motifs from motif_enrichment

    All plots use a consistent palette (active=red #E63946, inactive=grey
    #BBBBBB, controls=blue #457B9D, highlights=teal #2A9D8F) and are written
    at 200 DPI with 14 pt titles / 12 pt axis labels / 8 pt legends.

    Args:
        plot_type: One of {"volcano", "ranked_activity",
            "replicate_correlation", "annotation_boxplot", "motif_dotplot"}.
        data_file: TSV path. The required schema depends on plot_type:
            volcano              : `mean_activity` + (`pvalue` or `fdr`)
            ranked_activity      : `element_id` + `mean_activity` (+ `active`
                                   if you want bars colored)
            replicate_correlation: `repN_activity` columns for at least
                                   two replicates (N = 1, 2, ...)
            annotation_boxplot   : `element_id` + `mean_activity` (joins
                                   with annotation_file on element_id)
            motif_dotplot        : `tf_name`, `odds_ratio`, `fdr`,
                                   `n_active_hits` (the motif_enrichment output)
            Omit this argument to use activity_results.tsv from the upload directory.
        output_path: Destination PNG path. Defaults to <plot_type>.png in
            the upload directory.
        highlight_ids: Optional list of element_ids to highlight in teal
            (volcano, ranked_activity).
        neg_control_ids: Optional list of element_ids to draw in blue
            (volcano).
        annotation_file: Required when plot_type='annotation_boxplot' —
            TSV with columns `element_id` and `annotation`.

    Returns:
        dict with keys:
            plot_path (str): absolute path to the saved PNG.
            description (str): natural-language summary of the figure
                (e.g. "Volcano of 600 elements; 213 active at FDR<0.05",
                or "Pearson r=0.87 across 600 elements" for replicate plots).

    Raises:
        ValueError: unknown plot_type, missing required columns for the
            chosen plot, or annotation_file missing/invalid when required.

    Notes:
        - Uses matplotlib's headless Agg backend, so it is safe to call
          inside an MCP server without a display.
        - The motif_dotplot falls back to a "no significant motifs" note
          (still a valid PNG) if nothing reaches FDR < 0.05.
    """
    from creseq_mcp.plots.plots import plot_creseq

    resolved_output = output_path or str(UPLOAD_DIR / f"{plot_type}.png")
    return plot_creseq(
        data_file=_path(data_file, "activity_results.tsv"),
        plot_type=plot_type,
        output_path=resolved_output,
        highlight_ids=highlight_ids,
        neg_control_ids=neg_control_ids,
        annotation_file=annotation_file,
    )


@mcp.tool()
def tool_annotate_motifs(
    activity_results_path: str | None = None,
    design_manifest_path: str | None = None,
    tf_names: list[str] | None = None,
) -> dict:
    """
    Annotate each CRE with its best-matching TF motif (lightweight JASPAR REST scan).

    Updates activity_results.tsv in-place with a top_motif column.
    Enables the Motif Analysis tab in the frontend and motif_enrichment_summary.

    tf_names: optional list of TF names to scan (defaults to liver/HepG2 panel).
    """
    from creseq_mcp.motifs.annotate import annotate_top_motifs

    return annotate_top_motifs(
        activity_results_path=_path(activity_results_path, "activity_results.tsv"),
        design_manifest_path=_path(design_manifest_path, "design_manifest.tsv"),
        tf_names=tf_names,
        upload_dir=UPLOAD_DIR,
    )


@mcp.tool()
def tool_variant_delta_scores(
    activity_results_path: str | None = None,
    design_manifest_path: str | None = None,
) -> dict:
    """
    Compute variant effect scores: delta = mutant_log2_ratio - reference_log2_ratio.

    Requires oligo IDs with R:/A:/C: prefix convention (lentiMPRA standard),
    or a design manifest with variant_family and is_reference columns.
    Saves variant_delta_scores.tsv to the upload directory.
    """
    from creseq_mcp.variants.delta_scores import compute_variant_delta_scores

    _, summary = compute_variant_delta_scores(
        activity_results_path=_path(activity_results_path, "activity_results.tsv"),
        design_manifest_path=_path(design_manifest_path, "design_manifest.tsv"),
        upload_dir=UPLOAD_DIR,
    )
    return summary


if __name__ == "__main__":
    mcp.run()
