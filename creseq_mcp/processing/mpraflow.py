"""
creseq_mcp/processing/mpraflow.py
==================================
Wrapper around the MPRAflow ASSOCIATION workflow (kircherlab/MPRAflow).

MPRAflow handles paired-end reads, STARCODE barcode clustering, and BWA
alignment — giving better barcode error correction and oligo recovery than
the built-in pipeline.py.

Requires: nextflow + conda (MPRAflow pulls its own bioinformatics deps).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from creseq_mcp.processing.pipeline import _make_cigar_md

logger = logging.getLogger(__name__)

# MPRAflow assigned_counts.tsv column names (varies slightly by version)
_BC_COLS = ("BC", "barcode", "barcode_sequence")
_ID_COLS = ("name", "oligo_id", "oligo", "element")
_COUNT_COLS = ("n", "count", "reads", "read_count")


def _nextflow_bin() -> str | None:
    """Return path to nextflow binary, checking ~/bin if not on PATH."""
    found = shutil.which("nextflow")
    if found:
        return found
    local = Path.home() / "bin" / "nextflow"
    return str(local) if local.exists() else None


def is_available() -> bool:
    return _nextflow_bin() is not None


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")


def run_mpraflow(
    fastq_bc: Path,
    fastq_oligo: Path,
    design_fasta: Path,
    outdir: Path,
    *,
    name: str = "library",
    profile: str = "conda",
) -> Path:
    """
    Run MPRAflow ASSOCIATION workflow.

    Parameters
    ----------
    fastq_bc    : FASTQ with barcode reads (typically R2)
    fastq_oligo : FASTQ with oligo reads (typically R1)
    design_fasta: FASTA of designed oligo sequences
    outdir      : Directory for MPRAflow output
    name        : Library name (used in output filenames)
    profile     : Nextflow profile — "conda" (default) or "docker"

    Returns
    -------
    Path to the assigned_counts TSV produced by MPRAflow.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        _nextflow_bin(), "run", "kircherlab/MPRAflow",
        "-entry", "ASSOCIATION",
        "--name", name,
        "--fastq_bc", str(fastq_bc),
        "--fastq_oligo", str(fastq_oligo),
        "--design", str(design_fasta),
        "--outdir", str(outdir),
        "-profile", profile,
    ]

    logger.info("Running MPRAflow: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)

    if result.returncode != 0:
        raise RuntimeError(
            f"MPRAflow exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )

    # MPRAflow writes to: {outdir}/statistic/assigned_counts/{name}.assigned_counts.tsv
    candidates = list(outdir.glob(f"**/{name}.assigned_counts.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"MPRAflow output not found under {outdir}. "
            "Check the MPRAflow log for errors."
        )

    return candidates[0]


def convert_to_qc_format(
    assigned_counts_path: Path,
    reference_path: Path,
    upload_dir: Path,
) -> dict:
    """
    Convert MPRAflow assigned_counts TSV → mapping_table, plasmid_counts,
    design_manifest TSVs in upload_dir.

    Note: CIGAR/MD in the mapping table are set to perfect-match placeholders
    because MPRAflow's barcode-level output does not include per-read alignment
    strings.  synthesis_error_profile and oligo_length_qc will therefore report
    trivially perfect synthesis quality.
    """
    counts = pd.read_csv(assigned_counts_path, sep="\t")

    bc_col = _find_col(counts, _BC_COLS)
    id_col = _find_col(counts, _ID_COLS)
    n_col = _find_col(counts, _COUNT_COLS)

    counts = counts.rename(columns={bc_col: "barcode", id_col: "oligo_id", n_col: "n_reads"})

    # Placeholder CIGAR/MD — barcode matches itself perfectly
    counts["cigar"] = counts["barcode"].apply(lambda bc: f"{len(bc)}M")
    counts["md"] = counts["barcode"].apply(lambda bc: str(len(bc)))

    mapping_table = counts[["barcode", "oligo_id", "cigar", "md", "n_reads"]].copy()

    plasmid_counts = mapping_table[["barcode", "oligo_id"]].copy()
    plasmid_counts["dna_count"] = mapping_table["n_reads"]

    ref_df = pd.read_csv(reference_path, sep="\t")
    manifest_cols = ["oligo_id", "sequence", "designed_category"]
    if "variant_family" in ref_df.columns:
        manifest_cols.append("variant_family")
    design_manifest = ref_df[manifest_cols].drop_duplicates("oligo_id")

    mapping_table.to_csv(upload_dir / "mapping_table.tsv", sep="\t", index=False)
    plasmid_counts.to_csv(upload_dir / "plasmid_counts.tsv", sep="\t", index=False)
    design_manifest.to_csv(upload_dir / "design_manifest.tsv", sep="\t", index=False)

    return {
        "total_reads": int(mapping_table["n_reads"].sum()),
        "unique_barcodes": len(plasmid_counts),
        "oligos_represented": mapping_table["oligo_id"].nunique(),
        "oligos_in_reference": len(design_manifest),
    }


def process_and_save(
    fastq_bc: Path,
    fastq_oligo: Path,
    design_fasta: Path,
    reference_path: Path,
    upload_dir: Path,
    *,
    name: str = "library",
    profile: str = "conda",
) -> dict:
    """Run MPRAflow then convert output to QC-ready TSVs."""
    mpraflow_out = upload_dir / "mpraflow_out"
    assigned_counts = run_mpraflow(
        fastq_bc, fastq_oligo, design_fasta, mpraflow_out,
        name=name, profile=profile,
    )
    return convert_to_qc_format(assigned_counts, reference_path, upload_dir)
