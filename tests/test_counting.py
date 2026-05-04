"""
tests/test_counting.py
======================
Pytest tests for creseq_mcp/activity/counting.py.

Coverage:
  - process_dna_counting: happy path, zero counts, mismatch tolerance
  - process_rna_counting: single rep, multiple reps, default rep names
  - Edge cases: empty FASTQ, no barcode matches
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from creseq_mcp.activity.counting import process_dna_counting, process_rna_counting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mapping_table(path: Path, barcodes: list[str], barcode_len: int = 10) -> None:
    rows = [{"barcode": bc, "oligo_id": f"oligo_{i:03d}"} for i, bc in enumerate(barcodes)]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_fastq(path: Path, seqs: list[str]) -> None:
    """Write sequences as a plain FASTQ. Sequence IS the barcode (3-prime end)."""
    with open(path, "w") as fh:
        for i, seq in enumerate(seqs):
            fh.write(f"@read{i:04d}\n{seq}\n+\n{'I' * len(seq)}\n")


def _write_fastq_gz(path: Path, seqs: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        for i, seq in enumerate(seqs):
            fh.write(f"@read{i:04d}\n{seq}\n+\n{'I' * len(seq)}\n")


# ---------------------------------------------------------------------------
# process_dna_counting
# ---------------------------------------------------------------------------

class TestProcessDnaCounting:
    def test_happy_path_counts_barcodes(self, tmp_path):
        barcodes = ["AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes, barcode_len=10)

        # 3 reads for barcode 0, 2 for barcode 1, 0 for barcode 2
        seqs = ["AAAAAAAAAA"] * 3 + ["CCCCCCCCCC"] * 2
        fq = tmp_path / "dna.fastq"
        _write_fastq(fq, seqs)

        result = process_dna_counting(fq, mapping, tmp_path, barcode_len=10)

        assert result["total_reads"] == 5
        assert result["matched_reads"] == 5
        assert result["match_rate"] == 1.0

        plasmid = pd.read_csv(tmp_path / "plasmid_counts.tsv", sep="\t")
        counts = dict(zip(plasmid["barcode"], plasmid["dna_count"]))
        assert counts["AAAAAAAAAA"] == 3
        assert counts["CCCCCCCCCC"] == 2
        assert counts["GGGGGGGGGG"] == 0

    def test_zero_counts_when_no_matches(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)

        fq = tmp_path / "dna.fastq"
        _write_fastq(fq, ["TTTTTTTTTT"] * 3)

        result = process_dna_counting(fq, mapping, tmp_path, barcode_len=10)

        assert result["matched_reads"] == 0
        assert result["match_rate"] == 0.0
        plasmid = pd.read_csv(tmp_path / "plasmid_counts.tsv", sep="\t")
        assert plasmid["dna_count"].sum() == 0

    def test_writes_plasmid_counts_tsv(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)
        fq = tmp_path / "dna.fastq"
        _write_fastq(fq, ["AAAAAAAAAA"] * 2)

        process_dna_counting(fq, mapping, tmp_path, barcode_len=10)

        assert (tmp_path / "plasmid_counts.tsv").exists()
        df = pd.read_csv(tmp_path / "plasmid_counts.tsv", sep="\t")
        assert set(df.columns) >= {"barcode", "oligo_id", "dna_count"}

    def test_mismatch_tolerance(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)

        # One read with 1 mismatch from the known barcode
        fq = tmp_path / "dna.fastq"
        _write_fastq(fq, ["AAAAAAAAAC"])  # 1 mismatch at position 9

        result = process_dna_counting(
            fq, mapping, tmp_path, barcode_len=10, max_mismatch=1
        )
        assert result["matched_reads"] == 1

    def test_empty_fastq(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)
        fq = tmp_path / "dna.fastq"
        fq.write_text("")

        result = process_dna_counting(fq, mapping, tmp_path, barcode_len=10)
        assert result["total_reads"] == 0
        assert result["matched_reads"] == 0


# ---------------------------------------------------------------------------
# process_rna_counting
# ---------------------------------------------------------------------------

class TestProcessRnaCounting:
    def test_single_replicate(self, tmp_path):
        barcodes = ["AAAAAAAAAA", "CCCCCCCCCC"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)

        fq = tmp_path / "rna_rep1.fastq"
        _write_fastq(fq, ["AAAAAAAAAA"] * 4 + ["CCCCCCCCCC"] * 2)

        result = process_rna_counting([fq], mapping, tmp_path, barcode_len=10)

        assert result["replicates"] == ["rep1"]
        assert result["total_barcodes"] == 2

        rna = pd.read_csv(tmp_path / "rna_counts.tsv", sep="\t")
        assert "rna_count_rep1" in rna.columns
        counts = dict(zip(rna["barcode"], rna["rna_count_rep1"]))
        assert counts["AAAAAAAAAA"] == 4
        assert counts["CCCCCCCCCC"] == 2

    def test_multiple_replicates(self, tmp_path):
        barcodes = ["AAAAAAAAAA", "CCCCCCCCCC"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)

        fq1 = tmp_path / "rna1.fastq"
        fq2 = tmp_path / "rna2.fastq"
        _write_fastq(fq1, ["AAAAAAAAAA"] * 3)
        _write_fastq(fq2, ["CCCCCCCCCC"] * 5)

        result = process_rna_counting([fq1, fq2], mapping, tmp_path, barcode_len=10)

        assert result["replicates"] == ["rep1", "rep2"]
        rna = pd.read_csv(tmp_path / "rna_counts.tsv", sep="\t")
        assert "rna_count_rep1" in rna.columns
        assert "rna_count_rep2" in rna.columns

    def test_custom_rep_names(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)
        fq = tmp_path / "rna.fastq"
        _write_fastq(fq, ["AAAAAAAAAA"] * 2)

        result = process_rna_counting(
            [fq], mapping, tmp_path, rep_names=["HepG2_rep1"], barcode_len=10
        )
        assert result["replicates"] == ["HepG2_rep1"]
        rna = pd.read_csv(tmp_path / "rna_counts.tsv", sep="\t")
        assert "rna_count_HepG2_rep1" in rna.columns

    def test_writes_rna_counts_tsv(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)
        fq = tmp_path / "rna.fastq"
        _write_fastq(fq, ["AAAAAAAAAA"])

        process_rna_counting([fq], mapping, tmp_path, barcode_len=10)
        assert (tmp_path / "rna_counts.tsv").exists()

    def test_no_matches_gives_zero_counts(self, tmp_path):
        barcodes = ["AAAAAAAAAA"]
        mapping = tmp_path / "mapping_table.tsv"
        _write_mapping_table(mapping, barcodes)
        fq = tmp_path / "rna.fastq"
        _write_fastq(fq, ["TTTTTTTTTT"] * 3)

        result = process_rna_counting([fq], mapping, tmp_path, barcode_len=10)
        rna = pd.read_csv(tmp_path / "rna_counts.tsv", sep="\t")
        assert rna["rna_count_rep1"].sum() == 0
        assert result["per_replicate"][0]["matched_reads"] == 0
