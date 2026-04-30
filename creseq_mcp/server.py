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
    normalize_activity,
    prepare_counts,
    prepare_rag_context,
    rank_cre_candidates,
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
def tool_prepare_counts(
    plasmid_counts_path: str | None = None,
    rna_counts_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Reshape raw barcode-level CRE-seq counts into the long-format table the
    activity-calling pipeline expects.

    Bridges the upstream counting step (which writes plasmid + RNA files in a
    barcode-keyed wide format with one column per replicate) and the downstream
    normalize_activity / call_active_elements tools (which expect one row per
    barcode-replicate observation with canonical column names). Use this as
    the first step in the post-counting pipeline.

    Algorithm:
        1. Inner-join plasmid and RNA files on (barcode, oligo_id)
        2. Melt rna_count_rep1, rna_count_rep2, ... columns into long form
        3. Rename to canonical schema; cast `replicate` to int

    Args:
        plasmid_counts_path: Path to plasmid_counts.tsv with columns
            `barcode`, `oligo_id`, `dna_count`. Defaults to
            ~/.creseq/uploads/plasmid_counts.tsv when omitted.
        rna_counts_path: Path to rna_counts.tsv with columns `barcode`,
            `oligo_id`, and one or more `rna_count_repN` columns (N = 1, 2, ...).
            Defaults to ~/.creseq/uploads/rna_counts.tsv.
        output_path: Where to write the long-format TSV. Defaults to
            ~/.creseq/uploads/counts_long.tsv.

    Returns:
        dict with keys:
            rows (list[dict]): every output row as a JSON-serialisable dict
                with fields `element_id` (str), `barcode_id` (str),
                `replicate` (int, 1-indexed), `dna_counts` (int), `rna_counts` (int).
            summary (dict): { n_rows: int, n_elements: int (unique element count),
                n_replicates: int, output_path: str }.

    Notes:
        - Output column renames: oligo_id -> element_id, barcode -> barcode_id,
          dna_count -> dna_counts, rna_count_rep{N} (wide) -> rna_counts (long)
          plus a new `replicate` column carrying N as an integer.
        - Inner join: barcodes present in only one file are dropped silently.
        - Output row count = n_input_rows * n_replicates.
    """
    out = output_path or str(UPLOAD_DIR / "counts_long.tsv")
    return _serialise(
        prepare_counts(
            plasmid_counts_path=_path(plasmid_counts_path, "plasmid_counts.tsv"),
            rna_counts_path=_path(rna_counts_path, "rna_counts.tsv"),
            output_path=out,
        )
    )


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


@mcp.tool(name="call_active_elements")
def tool_call_active_elements_full(
    activity_table_path: str,
    negative_controls: list[str],
    fdr_threshold: float = 0.05,
    method: str = "empirical",
    count_table_path: str | None = None,
) -> dict:
    """
    Classify CRE-seq elements as active vs. inactive against an empirical null
    derived from negative-control elements.

    Calls regulatory activity using a robust median/MAD null distribution
    rather than mean/std so a single outlier control cannot inflate the scale.
    A Shapiro-Wilk test on the controls reports normality (the empirical
    z-test assumes it); BH-FDR is computed only over the test set so negative
    controls do not dilute the multiple-testing correction.

    Algorithm (method='empirical'):
        1. null_center = median(neg_control activities)
        2. null_scale  = 1.4826 * MAD(neg_control activities)
        3. Shapiro-Wilk normality check on controls; warns if p < 0.01
        4. Per test element: zscore = (mean_activity - null_center) / null_scale
        5. One-sided p-value (upper tail of standard normal)
        6. BH-FDR adjustment over test elements only
        7. active = (pvalue <= alpha) AND (fdr <= fdr_threshold)
                    AND (mean_activity > null_center)

    Args:
        activity_table_path: Path to a per-element activity table (TSV) with
            required columns `element_id`, `mean_activity`, `std_activity`,
            `n_barcodes`.
        negative_controls: List of element_ids to use as the null. Need >=20
            for reliable z-test; a warning fires below that.
        fdr_threshold: BH q-value cutoff for the `active` flag. Default 0.05.
            Typical range 0.01-0.10.
        method: "empirical" (default) or "glm". The GLM negative-binomial
            method is a stub — calling it raises NotImplementedError.
        count_table_path: Reserved for the GLM method (ignored when
            method='empirical').

    Returns:
        dict with keys:
            rows (list[dict]): every input row plus 5 new columns:
                active (bool), pvalue (float|NaN), fdr (float|NaN),
                fold_over_controls (float|NaN), zscore (float|NaN).
                Negative controls carry NaN for p/fdr/z (excluded from testing
                against themselves).
            summary (dict): { n_total_elements, n_negative_controls,
                n_test_elements, n_active, n_inactive, fdr_threshold, method,
                null_distribution (dict with center, scale, estimator,
                n_controls, shapiro_pvalue), active_summary (dict),
                n_silencer_candidates, warnings (list[str]) }.

    Side effects:
        Writes <activity_table>_classified.tsv next to the input.

    Notes:
        - fold_over_controls = 2 ** (mean_activity - null_center) — linear-scale
          enrichment over the control median.
        - n_silencer_candidates counts elements significantly *below* the null
          (potential repressors); they are not flagged as `active`.
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
    from creseq_mcp.motif import extract_sequences_to_fasta

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
        output_path: Optional TSV destination. Defaults to a temp file.

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
        plot_type: One of {"volcano", "ranked_activity",
            "replicate_correlation", "annotation_boxplot", "motif_dotplot"}.
        output_path: Destination PNG path. Default "plot.png".
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
    from creseq_mcp.plotting import plot_creseq

    return plot_creseq(
        data_file=data_file,
        plot_type=plot_type,
        output_path=output_path,
        highlight_ids=highlight_ids,
        neg_control_ids=neg_control_ids,
        annotation_file=annotation_file,
    )


if __name__ == "__main__":
    mcp.run()
