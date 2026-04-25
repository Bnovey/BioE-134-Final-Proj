"""
tests/conftest.py
=================
Shared pytest fixtures for CRE-seq QC tests.

Synthetic data parameters (mirroring a small but realistic CRE-seq library):
  - 500 designed oligos, 84 bp each
  - 10 bp barcodes, ~20 barcodes/oligo  →  ~10,000 rows in mapping table
  - designed_categories: test_element (380), scrambled_control (50),
    motif_knockout (30), positive_control (20), negative_control (20)
  - 30 motif_knockout oligos form 30 families, each with one test_element parent
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_OLIGOS = 500
N_TEST = 380
N_SCRAMBLED = 50
N_KNOCKOUT = 30
N_POS_CTRL = 20
N_NEG_CTRL = 20
OLIGO_LEN = 84
BC_LEN = 10
MEAN_BC_PER_OLIGO = 20
NUCLEOTIDES = ["A", "C", "G", "T"]

OLIGO_IDS = [f"oligo_{i:04d}" for i in range(N_OLIGOS)]


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_seq(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(NUCLEOTIDES, size=length))


def _gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _perfect_cigar_md(oligo_len: int):
    return f"{oligo_len}M", str(oligo_len)


def _mismatch_cigar_md(oligo_len: int, rng: np.random.Generator):
    pos = int(rng.integers(1, oligo_len - 1))
    return f"{oligo_len}M", f"{pos}A{oligo_len - pos - 1}"


# ---------------------------------------------------------------------------
# Design manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def design_manifest_df() -> pd.DataFrame:
    """
    500-oligo design manifest with categories and variant-family links.

    Variant families (one reference test_element + one motif_knockout sibling)
    are encoded with `variant_family` + `is_reference`.  The first N_KNOCKOUT
    test elements double as references; each is paired with one knockout in the
    motif_knockout block.
    """
    rng = _rng(44)
    test_ids = OLIGO_IDS[:N_TEST]
    scrambled_ids = OLIGO_IDS[N_TEST : N_TEST + N_SCRAMBLED]
    knockout_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED : N_TEST + N_SCRAMBLED + N_KNOCKOUT]
    pos_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED + N_KNOCKOUT : N_TEST + N_SCRAMBLED + N_KNOCKOUT + N_POS_CTRL]
    neg_ids = OLIGO_IDS[N_TEST + N_SCRAMBLED + N_KNOCKOUT + N_POS_CTRL :]

    # Each knockout shares a variant_family with one of the first N_KNOCKOUT
    # test elements (the family's reference).
    family_for_knockout = {ko: f"FAM{i:03d}" for i, ko in enumerate(knockout_ids)}
    family_for_ref = {test_ids[i]: f"FAM{i:03d}" for i in range(N_KNOCKOUT)}

    rows = []
    for oligo_id in OLIGO_IDS:
        seq = _rand_seq(OLIGO_LEN, rng)
        family = None
        is_ref = False
        if oligo_id in test_ids:
            cat = "test_element"
            if oligo_id in family_for_ref:
                family = family_for_ref[oligo_id]
                is_ref = True
        elif oligo_id in scrambled_ids:
            cat = "scrambled_control"
        elif oligo_id in knockout_ids:
            cat = "motif_knockout"
            family = family_for_knockout[oligo_id]
            is_ref = False
        elif oligo_id in pos_ids:
            cat = "positive_control"
        else:
            cat = "negative_control"

        rows.append(
            {
                "oligo_id": oligo_id,
                "sequence": seq,
                "length": OLIGO_LEN,
                "gc_content": _gc(seq),
                "designed_category": cat,
                "variant_family": family,
                "is_reference": is_ref,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def design_manifest_path(tmp_path_factory, design_manifest_df) -> str:
    p = tmp_path_factory.mktemp("data") / "design_manifest.tsv"
    design_manifest_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Mapping table (healthy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mapping_df() -> pd.DataFrame:
    """Healthy mapping table: ~20 barcodes/oligo, 85% perfect alignments."""
    rng = _rng(42)
    rows = []
    for oligo_id in OLIGO_IDS:
        n_bc = int(rng.integers(15, 26))  # 15–25 barcodes
        for _ in range(n_bc):
            bc = _rand_seq(BC_LEN, rng)
            n_reads = int(rng.integers(5, 101))
            if rng.random() < 0.85:
                cigar, md = _perfect_cigar_md(OLIGO_LEN)
            else:
                cigar, md = _mismatch_cigar_md(OLIGO_LEN, rng)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": n_reads,
                    "cigar": cigar,
                    "md": md,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def mapping_table_path(tmp_path_factory, mapping_df) -> str:
    p = tmp_path_factory.mktemp("data") / "mapping_table.tsv"
    mapping_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Plasmid count table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def plasmid_df(mapping_df) -> pd.DataFrame:
    """Plasmid count table derived from the mapping table."""
    rng = _rng(43)
    df = mapping_df[["barcode", "oligo_id"]].copy()
    # Use a narrow count range so per-oligo Gini stays < 0.30 (healthy library)
    df["dna_count"] = rng.integers(50, 151, size=len(df)).astype(int)
    return df


@pytest.fixture(scope="session")
def plasmid_count_path(tmp_path_factory, plasmid_df) -> str:
    p = tmp_path_factory.mktemp("data") / "plasmid_counts.tsv"
    plasmid_df.to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Edge-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def long_barcode_mapping_path(tmp_path_factory) -> str:
    """Mapping table with 15 bp barcodes (outside CRE-seq 8–12 bp window)."""
    rng = _rng(50)
    rows = []
    for oligo_id in OLIGO_IDS[:50]:
        for _ in range(20):
            bc = _rand_seq(15, rng)  # lentiMPRA-length barcodes
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "long_bc_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def long_oligo_manifest_path(tmp_path_factory) -> str:
    """Design manifest with 500 bp oligos (outside CRE-seq 84–200 bp window)."""
    rng = _rng(51)
    rows = [
        {
            "oligo_id": f"oligo_{i:04d}",
            "sequence": _rand_seq(500, rng),
            "length": 500,
            "gc_content": 0.5,
            "designed_category": "test_element",
            "parent_element_id": None,
        }
        for i in range(50)
    ]
    p = tmp_path_factory.mktemp("data") / "long_oligo_manifest.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def missing_ref_mapping_path(tmp_path_factory, design_manifest_df) -> str:
    """
    Mapping table where the reference oligos of several families are absent,
    but their knockouts are present.
    """
    rng = _rng(52)
    refs = design_manifest_df[
        (design_manifest_df["variant_family"].notna())
        & (design_manifest_df["is_reference"] == True)  # noqa: E712
    ]
    ref_ids = set(refs["oligo_id"].unique())

    rows = []
    for oligo_id in OLIGO_IDS:
        if oligo_id in ref_ids:
            continue  # drop all reference oligos → families lack their reference
        for _ in range(15):
            bc = _rand_seq(BC_LEN, rng)
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "missing_ref_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def low_recovery_mapping_path(tmp_path_factory, design_manifest_df) -> str:
    """Mapping table where only 50% of oligos are present (should fail oligo_recovery)."""
    rng = _rng(53)
    recovered = set(OLIGO_IDS[::2])  # every other oligo
    rows = []
    for oligo_id in OLIGO_IDS:
        if oligo_id not in recovered:
            continue
        for _ in range(12):
            bc = _rand_seq(BC_LEN, rng)
            cigar, md = _perfect_cigar_md(OLIGO_LEN)
            rows.append(
                {
                    "barcode": bc,
                    "oligo_id": oligo_id,
                    "n_reads": 20,
                    "cigar": cigar,
                    "md": md,
                }
            )
    p = tmp_path_factory.mktemp("data") / "low_recovery_mapping.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture(scope="session")
def empty_mapping_path(tmp_path_factory) -> str:
    """Empty mapping table (header only)."""
    p = tmp_path_factory.mktemp("data") / "empty_mapping.tsv"
    pd.DataFrame(
        columns=["barcode", "oligo_id", "n_reads", "cigar", "md"]
    ).to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Plotting fixtures (used by tests/test_plotting.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def classified_fixture() -> pd.DataFrame:
    """
    100-element classified table mimicking the output of
    ``call_active_elements_empirical``: 20 controls (NaN p/fdr/zscore),
    30 active (small p, large fold), 50 inactive.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for i in range(20):
        rows.append({
            "element_id": f"neg_ctrl_{i:03d}",
            "mean_activity": float(rng.normal(0, 0.3)),
            "std_activity": 0.3,
            "n_barcodes": 15,
            "active": False,
            "pvalue": np.nan,
            "fdr": np.nan,
            "fold_over_controls": np.nan,
            "zscore": np.nan,
        })
    for i in range(30):
        act = float(rng.normal(2.0, 0.5))
        rows.append({
            "element_id": f"active_{i:03d}",
            "mean_activity": act,
            "std_activity": 0.4,
            "n_barcodes": 15,
            "active": True,
            "pvalue": float(rng.uniform(1e-10, 1e-3)),
            "fdr": float(rng.uniform(1e-8, 0.04)),
            "fold_over_controls": 2 ** act,
            "zscore": act / 0.3,
        })
    for i in range(50):
        act = float(rng.normal(0.1, 0.3))
        rows.append({
            "element_id": f"inactive_{i:03d}",
            "mean_activity": act,
            "std_activity": 0.3,
            "n_barcodes": 15,
            "active": False,
            "pvalue": float(rng.uniform(0.1, 1.0)),
            "fdr": float(rng.uniform(0.2, 1.0)),
            "fold_over_controls": 2 ** act,
            "zscore": act / 0.3,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def activity_with_reps_fixture() -> pd.DataFrame:
    """100 elements with rep1/rep2/rep3_activity columns (correlated)."""
    rng = np.random.default_rng(42)
    n = 100
    base = rng.normal(0.5, 1.5, n)
    rows = []
    for i in range(n):
        rows.append({
            "element_id": f"elem_{i:03d}",
            "mean_activity": float(base[i]),
            "std_activity": 0.3,
            "n_barcodes": 15,
            "rep1_activity": float(base[i] + rng.normal(0, 0.2)),
            "rep2_activity": float(base[i] + rng.normal(0, 0.2)),
            "rep3_activity": float(base[i] + rng.normal(0, 0.25)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def annotation_fixture(tmp_path) -> Path:
    """TSV mapping element IDs to annotation categories."""
    categories = ["enhancer", "promoter", "CTCF", "heterochromatin"]
    lines: list[str] = []
    for i in range(30):
        lines.append(f"active_{i:03d}\tenhancer")
    for i in range(50):
        cat = categories[i % len(categories)]
        lines.append(f"inactive_{i:03d}\t{cat}")
    p = tmp_path / "annotations.tsv"
    p.write_text("element_id\tannotation\n" + "\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Motif-enrichment fixtures (used by tests/test_motif.py)
# ---------------------------------------------------------------------------


def _make_mock_motif(matrix_id: str, name: str, counts: dict) -> object:
    """Build a JASPAR-Motif-shaped mock with .matrix_id, .name, .counts."""
    return type(
        "MockMotif",
        (),
        {"matrix_id": matrix_id, "name": name, "counts": counts},
    )()


@pytest.fixture
def small_motif_list() -> list:
    """
    3 hand-built motifs for fast unit tests, matching the pyjaspar interface
    (.matrix_id, .name, .counts).  Avoids loading all ~880 JASPAR motifs.
    """
    gata = _make_mock_motif(
        "MA0035.1", "GATA1",
        {
            "A": [10, 0, 10, 0, 10, 10, 0, 0],
            "C": [0, 0, 0, 0, 0, 0, 0, 0],
            "G": [0, 10, 0, 0, 0, 0, 10, 10],
            "T": [0, 0, 0, 10, 0, 0, 0, 0],
        },
    )
    sp1 = _make_mock_motif(
        "MA0079.1", "SP1",
        {
            "A": [0, 0, 0, 0, 0, 0],
            "C": [0, 0, 0, 5, 5, 5],
            "G": [10, 10, 10, 5, 5, 5],
            "T": [0, 0, 0, 0, 0, 0],
        },
    )
    rand = _make_mock_motif(
        "RAND01", "RandomControl",
        {
            "A": [3, 3, 3, 3, 3, 3],
            "C": [3, 3, 3, 3, 3, 3],
            "G": [3, 3, 3, 3, 3, 3],
            "T": [3, 3, 3, 3, 3, 3],
        },
    )
    return [gata, sp1, rand]


@pytest.fixture
def active_fasta_with_motif(tmp_path):
    """30 × 170 bp sequences with GATA1 consensus 'AGATAAGG' planted at pos 50."""
    rng = np.random.default_rng(42)
    motif = "AGATAAGG"
    lines = []
    for i in range(30):
        left = "".join(rng.choice(list("ACGT"), 50))
        right = "".join(rng.choice(list("ACGT"), 170 - 50 - len(motif)))
        lines.append(f">active_{i:03d}\n{left + motif + right}")
    p = tmp_path / "active.fa"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def background_fasta_no_motif(tmp_path):
    """50 × 170 bp purely random ACGT sequences."""
    rng = np.random.default_rng(99)
    lines = [
        f">bg_{i:03d}\n{''.join(rng.choice(list('ACGT'), 170))}"
        for i in range(50)
    ]
    p = tmp_path / "background.fa"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def manifest_with_sequences_fixture(tmp_path, classified_fixture):
    """
    TSV manifest with element_id + sequence columns, covering every ID in
    classified_fixture.  170 bp random sequences — content is irrelevant for
    the bridge tests; only the lookup matters.
    """
    rng = np.random.default_rng(101)
    rows = []
    for eid in classified_fixture["element_id"]:
        seq = "".join(rng.choice(list("ACGT"), size=170))
        rows.append({"element_id": eid, "sequence": seq})
    p = tmp_path / "manifest.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture
def both_random_fasta(tmp_path):
    """Two random FASTAs (no enrichment expected) — returns (active_path, bg_path)."""
    rng = np.random.default_rng(77)
    active = [
        f">a_{i:03d}\n{''.join(rng.choice(list('ACGT'), 170))}"
        for i in range(30)
    ]
    bg = [
        f">b_{i:03d}\n{''.join(rng.choice(list('ACGT'), 170))}"
        for i in range(50)
    ]
    a = tmp_path / "active_rand.fa"
    b = tmp_path / "bg_rand.fa"
    a.write_text("\n".join(active) + "\n")
    b.write_text("\n".join(bg) + "\n")
    return str(a), str(b)


@pytest.fixture
def motif_fixture() -> pd.DataFrame:
    """Synthetic TF-motif enrichment table (8 motifs, mixed significance)."""
    return pd.DataFrame({
        "tf_name": ["GATA1", "AP-1", "SP1", "NF-E2", "CEBPB", "HNF4A", "FOXA1", "ETS1"],
        "odds_ratio": [4.2, 3.1, 2.5, 3.8, 1.3, 1.9, 2.1, 1.6],
        "fdr": [1e-8, 2e-5, 3e-3, 5e-7, 0.12, 0.03, 0.01, 0.08],
        "n_active_hits": [45, 38, 22, 41, 8, 15, 19, 11],
        "n_background_hits": [12, 15, 10, 13, 7, 9, 11, 8],
    })
