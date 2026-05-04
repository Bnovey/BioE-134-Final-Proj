"""
tests/test_association.py
=========================
Pytest tests for creseq_mcp/association/association.py.

Coverage:
  - _iter_fastq: parses read name, header-embedded barcode, and sequence
  - _load_bc_fastq: loads barcodes from a separate index FASTQ
  - _filter_assignments: coverage and fraction filters
  - _open_fasta: magic-byte gzip detection
  - _build_design_manifest: FASTA parsing, labels merge, missing labels fallback
  - _cluster_barcodes: identity fallback when starcode is absent
  - run_association: end-to-end with synthetic data (skipped if mappy absent)
"""

from __future__ import annotations

import gzip
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from creseq_mcp.association.association import (
    _build_design_manifest,
    _cluster_barcodes,
    _filter_assignments,
    _iter_fastq,
    _load_bc_fastq,
    _open_fasta,
)


# ---------------------------------------------------------------------------
# Helpers to write synthetic files
# ---------------------------------------------------------------------------

def _write_fastq(path: Path, reads: list[tuple[str, str, str]]) -> None:
    """Write (name, barcode_suffix, seq) as a plain FASTQ."""
    with open(path, "w") as fh:
        for name, bc_suffix, seq in reads:
            header = f"@{name} 1:N:0:{bc_suffix}+AAAA"
            fh.write(f"{header}\n{seq}\n+\n{'I' * len(seq)}\n")


def _write_bc_fastq(path: Path, reads: list[tuple[str, str]]) -> None:
    """Write (name, barcode_seq) as a plain FASTQ (index read format)."""
    with open(path, "w") as fh:
        for name, bc in reads:
            fh.write(f"@{name} 1:N:0:0\n{bc}\n+\n{'I' * len(bc)}\n")


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for oid, seq in records:
            fh.write(f">{oid}\n{seq}\n")


def _write_fasta_gz(path: Path, records: list[tuple[str, str]]) -> None:
    with gzip.open(path, "wt") as fh:
        for oid, seq in records:
            fh.write(f">{oid}\n{seq}\n")


# ---------------------------------------------------------------------------
# _iter_fastq
# ---------------------------------------------------------------------------

class TestIterFastq:
    def test_parses_name_barcode_seq(self, tmp_path):
        fq = tmp_path / "r1.fastq"
        _write_fastq(fq, [("read1", "ACGTACGTAC", "AAACCCGGGTTT")])
        reads = list(_iter_fastq(fq))
        assert len(reads) == 1
        name, bc, seq = reads[0]
        assert name == "read1"
        assert bc == "ACGTACGTAC"
        assert seq == "AAACCCGGGTTT"

    def test_multiple_reads(self, tmp_path):
        fq = tmp_path / "r1.fastq"
        _write_fastq(fq, [
            ("r1", "AAAAAAAAAA", "ACGT"),
            ("r2", "CCCCCCCCCC", "TGCA"),
            ("r3", "GGGGGGGGGG", "AAAA"),
        ])
        reads = list(_iter_fastq(fq))
        assert len(reads) == 3
        names = [r[0] for r in reads]
        assert names == ["r1", "r2", "r3"]

    def test_no_barcode_in_header_returns_none(self, tmp_path):
        fq = tmp_path / "r1.fastq"
        with open(fq, "w") as fh:
            fh.write("@read1\nACGT\n+\nIIII\n")
        reads = list(_iter_fastq(fq))
        assert len(reads) == 1
        _, bc, _ = reads[0]
        assert bc is None


# ---------------------------------------------------------------------------
# _load_bc_fastq
# ---------------------------------------------------------------------------

class TestLoadBcFastq:
    def test_returns_name_to_barcode_mapping(self, tmp_path):
        bc_fq = tmp_path / "bc.fastq"
        _write_bc_fastq(bc_fq, [("read1", "ACGTACGT"), ("read2", "TTTTCCCC")])
        bc_map = _load_bc_fastq(bc_fq)
        assert bc_map == {"read1": "ACGTACGT", "read2": "TTTTCCCC"}

    def test_empty_fastq_returns_empty_dict(self, tmp_path):
        bc_fq = tmp_path / "bc.fastq"
        bc_fq.write_text("")
        bc_map = _load_bc_fastq(bc_fq)
        assert bc_map == {}

    def test_gzipped_bc_fastq(self, tmp_path):
        bc_fq = tmp_path / "bc.fastq.gz"
        with gzip.open(bc_fq, "wt") as fh:
            fh.write("@read1 1:N:0:0\nACGTACGT\n+\nIIIIIIII\n")
        bc_map = _load_bc_fastq(bc_fq)
        assert bc_map == {"read1": "ACGTACGT"}


# ---------------------------------------------------------------------------
# _filter_assignments
# ---------------------------------------------------------------------------

class TestFilterAssignments:
    def test_passes_barcode_meeting_both_filters(self):
        pairs = [("BC01", "oligo_001")] * 5
        df = _filter_assignments(pairs, min_cov=3, min_frac=0.5)
        assert len(df) == 1
        assert df.iloc[0]["barcode"] == "BC01"
        assert df.iloc[0]["oligo_id"] == "oligo_001"

    def test_rejects_barcode_below_min_cov(self):
        pairs = [("BC01", "oligo_001")] * 2
        df = _filter_assignments(pairs, min_cov=3, min_frac=0.5)
        assert len(df) == 0

    def test_rejects_barcode_below_min_frac(self):
        # 3 reads to oligo_001, 3 reads to oligo_002 → 50% fraction, fails min_frac=0.6
        pairs = [("BC01", "oligo_001")] * 3 + [("BC01", "oligo_002")] * 3
        df = _filter_assignments(pairs, min_cov=3, min_frac=0.6)
        assert len(df) == 0

    def test_picks_top_oligo_when_ambiguous(self):
        # 4 reads to oligo_001, 1 read to oligo_002 → top oligo wins
        pairs = [("BC01", "oligo_001")] * 4 + [("BC01", "oligo_002")] * 1
        df = _filter_assignments(pairs, min_cov=3, min_frac=0.5)
        assert len(df) == 1
        assert df.iloc[0]["oligo_id"] == "oligo_001"

    def test_multiple_barcodes_filtered_independently(self):
        pairs = (
            [("BC01", "oligo_001")] * 5   # passes
            + [("BC02", "oligo_002")] * 1  # fails min_cov
        )
        df = _filter_assignments(pairs, min_cov=3, min_frac=0.5)
        assert len(df) == 1
        assert df.iloc[0]["barcode"] == "BC01"

    def test_empty_input_returns_empty_dataframe(self):
        df = _filter_assignments([], min_cov=3, min_frac=0.5)
        assert len(df) == 0
        assert set(df.columns) >= {"barcode", "oligo_id", "n_reads"}


# ---------------------------------------------------------------------------
# _open_fasta
# ---------------------------------------------------------------------------

class TestOpenFasta:
    def test_opens_plain_fasta(self, tmp_path):
        fa = tmp_path / "design.fa"
        _write_fasta(fa, [("oligo_001", "ACGTACGT")])
        with _open_fasta(fa) as fh:
            content = fh.read()
        assert ">oligo_001" in content

    def test_opens_gzipped_fasta_without_gz_extension(self, tmp_path):
        # Simulates reference.fa that is actually gzip-compressed
        fa = tmp_path / "reference.fa"
        _write_fasta_gz(fa, [("oligo_001", "ACGTACGT")])
        with _open_fasta(fa) as fh:
            content = fh.read()
        assert ">oligo_001" in content

    def test_opens_gzipped_fasta_with_gz_extension(self, tmp_path):
        fa = tmp_path / "reference.fa.gz"
        _write_fasta_gz(fa, [("oligo_002", "TTTTGGGG")])
        with _open_fasta(fa) as fh:
            content = fh.read()
        assert ">oligo_002" in content


# ---------------------------------------------------------------------------
# _build_design_manifest
# ---------------------------------------------------------------------------

class TestBuildDesignManifest:
    def test_basic_fasta_no_labels(self, tmp_path):
        fa = tmp_path / "design.fa"
        _write_fasta(fa, [("oligo_001", "ACGT"), ("oligo_002", "TTGG")])
        manifest = _build_design_manifest(fa, labels_path=None)
        assert len(manifest) == 2
        assert set(manifest.columns) >= {"oligo_id", "sequence", "designed_category"}
        assert (manifest["designed_category"] == "other").all()

    def test_labels_merged_correctly(self, tmp_path):
        fa = tmp_path / "design.fa"
        _write_fasta(fa, [("oligo_001", "ACGT"), ("oligo_002", "TTGG")])
        labels = pd.DataFrame({
            "oligo_id": ["oligo_001", "oligo_002"],
            "designed_category": ["test_element", "negative_control"],
        })
        labels_path = tmp_path / "labels.tsv"
        labels.to_csv(labels_path, sep="\t", index=False)
        manifest = _build_design_manifest(fa, labels_path=labels_path)
        cats = dict(zip(manifest["oligo_id"], manifest["designed_category"]))
        assert cats["oligo_001"] == "test_element"
        assert cats["oligo_002"] == "negative_control"

    def test_missing_labels_file_falls_back_to_other(self, tmp_path):
        fa = tmp_path / "design.fa"
        _write_fasta(fa, [("oligo_001", "ACGT")])
        manifest = _build_design_manifest(fa, labels_path=tmp_path / "nonexistent.tsv")
        assert (manifest["designed_category"] == "other").all()

    def test_gzipped_fasta_parsed(self, tmp_path):
        fa = tmp_path / "reference.fa"
        _write_fasta_gz(fa, [("oligo_001", "AAACCC"), ("oligo_002", "GGGTT")])
        manifest = _build_design_manifest(fa, labels_path=None)
        assert len(manifest) == 2
        assert set(manifest["oligo_id"]) == {"oligo_001", "oligo_002"}


# ---------------------------------------------------------------------------
# _cluster_barcodes
# ---------------------------------------------------------------------------

class TestClusterBarcodes:
    def test_identity_fallback_when_starcode_absent(self, monkeypatch):
        monkeypatch.setattr(
            "creseq_mcp.association.association._starcode_available",
            lambda: False,
        )
        barcodes = ["AAAA", "CCCC", "GGGG"]
        result = _cluster_barcodes(barcodes)
        assert result == {"AAAA": "AAAA", "CCCC": "CCCC", "GGGG": "GGGG"}

    def test_returns_dict_with_all_input_barcodes(self, monkeypatch):
        monkeypatch.setattr(
            "creseq_mcp.association.association._starcode_available",
            lambda: False,
        )
        barcodes = ["AAAA", "AAAA", "CCCC"]  # duplicates
        result = _cluster_barcodes(barcodes)
        assert "AAAA" in result
        assert "CCCC" in result


# ---------------------------------------------------------------------------
# run_association (end-to-end, skipped if mappy not installed)
# ---------------------------------------------------------------------------

try:
    import mappy as _mappy
    _HAS_MAPPY = True
except ImportError:
    _HAS_MAPPY = False


@pytest.mark.skipif(not _HAS_MAPPY, reason="mappy not installed")
class TestRunAssociation:
    @pytest.fixture
    def assoc_inputs(self, tmp_path):
        """Synthetic R1 FASTQ + design FASTA with matching sequences."""
        oligos = [
            ("oligo_001", "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
            ("oligo_002", "TTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAA"),
        ]
        fa = tmp_path / "design.fa"
        _write_fasta(fa, oligos)

        # 6 reads for oligo_001 under barcode BC01, 6 reads for oligo_002 under BC02
        reads = (
            [("r{:03d}".format(i), "ACACACACAC", oligos[0][1]) for i in range(6)]
            + [("r{:03d}".format(i + 6), "TGTGTGTGTG", oligos[1][1]) for i in range(6)]
        )
        fq = tmp_path / "r1.fastq"
        _write_fastq(fq, reads)
        return fq, fa, tmp_path

    def test_produces_output_files(self, assoc_inputs):
        from creseq_mcp.association.association import run_association
        fq, fa, outdir = assoc_inputs
        run_association(fq, fa, outdir, min_cov=1, min_frac=0.5)
        assert (outdir / "mapping_table.tsv").exists()
        assert (outdir / "plasmid_counts.tsv").exists()
        assert (outdir / "design_manifest.tsv").exists()

    def test_returns_summary_dict_with_pass(self, assoc_inputs):
        from creseq_mcp.association.association import run_association
        fq, fa, outdir = assoc_inputs
        result = run_association(fq, fa, outdir, min_cov=1, min_frac=0.5)
        assert "pass" in result
        assert "n_reads_total" in result
        assert "n_barcodes_passing_filter" in result
        assert result["n_reads_total"] == 12

    def test_separate_bc_fastq(self, tmp_path):
        from creseq_mcp.association.association import run_association
        oligos = [
            ("oligo_001", "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
        ]
        fa = tmp_path / "design.fa"
        _write_fasta(fa, oligos)

        # R1 has no barcode in header; barcodes come from separate index FASTQ
        with open(tmp_path / "r1.fastq", "w") as fh:
            for i in range(5):
                fh.write(f"@read{i:03d}\n{oligos[0][1]}\n+\n{'I'*len(oligos[0][1])}\n")
        _write_bc_fastq(tmp_path / "bc.fastq", [
            (f"read{i:03d}", "ACACACACAC") for i in range(5)
        ])

        result = run_association(
            tmp_path / "r1.fastq",
            fa,
            tmp_path / "out",
            fastq_bc=tmp_path / "bc.fastq",
            min_cov=1,
            min_frac=0.5,
        )
        assert result["n_reads_total"] == 5
