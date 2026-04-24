"""
Generate synthetic CRE-seq test data and write all 4 QC-ready TSVs
to ~/.creseq/uploads/.

Design
------
- 600 oligos: 350 test_elements, 100 positive_controls, 150 negative_controls
- 50 variant families (reference + 4 mutants each)
- 20 barcodes per oligo  →  barcode_complexity PASS
- ~3% intentional barcode collisions  →  collision analysis borderline
- ~70% perfect CIGAR/MD  →  synthesis_error_profile PASS
- Realistic DNA/RNA counts with active calling via neg-ctrl z-test
"""
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)

UPLOAD_DIR = Path.home() / ".creseq" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLIGO_LEN   = 230
BARCODE_LEN = 15
N_OLIGOS    = 600
N_BARCODES_PER_OLIGO = 20

# ── helpers ──────────────────────────────────────────────────────────────────

def rand_seq(n):
    return "".join(random.choices("ACGT", k=n))

def mutate(seq, n_mut=1):
    bases = list(seq)
    positions = random.sample(range(len(seq)), n_mut)
    for p in positions:
        bases[p] = random.choice([b for b in "ACGT" if b != bases[p]])
    return "".join(bases)

def make_cigar_md(obs, ref):
    n = min(len(obs), len(ref))
    clip = abs(len(obs) - len(ref))
    md_parts, run = [], 0
    for o, r in zip(obs[:n], ref[:n]):
        if o == r:
            run += 1
        else:
            if run:
                md_parts.append(str(run))
                run = 0
            md_parts.append(r)
    if run:
        md_parts.append(str(run))
    cigar = f"{n}M" + (f"{clip}S" if clip else "")
    md = "".join(md_parts) if md_parts else str(n)
    return cigar, md

# ── design manifest ──────────────────────────────────────────────────────────

print("Building design manifest…")
oligos = []

# 50 variant families: 1 reference + 4 mutants
for fam_i in range(50):
    ref_seq = rand_seq(OLIGO_LEN)
    fam_id  = f"FAM{fam_i:03d}"
    oligos.append({
        "oligo_id": f"{fam_id}_ref",
        "sequence": ref_seq,
        "designed_category": "test_element",
        "variant_family": fam_id,
        "is_reference": True,
    })
    for mut_i in range(4):
        oligos.append({
            "oligo_id": f"{fam_id}_mut{mut_i}",
            "sequence": mutate(ref_seq, n_mut=rng.integers(1, 5)),
            "designed_category": "test_element",
            "variant_family": fam_id,
            "is_reference": False,
        })

# remaining test elements (no family)
for i in range(350 - 250):
    oligos.append({
        "oligo_id": f"TEST{i:04d}",
        "sequence": rand_seq(OLIGO_LEN),
        "designed_category": "test_element",
        "variant_family": None,
        "is_reference": False,
    })

# positive controls
for i in range(100):
    oligos.append({
        "oligo_id": f"POSCTRL{i:03d}",
        "sequence": rand_seq(OLIGO_LEN),
        "designed_category": "positive_control",
        "variant_family": None,
        "is_reference": False,
    })

# negative controls (scrambled — low GC bias)
for i in range(150):
    oligos.append({
        "oligo_id": f"NEGCTRL{i:03d}",
        "sequence": rand_seq(OLIGO_LEN),
        "designed_category": "negative_control",
        "variant_family": None,
        "is_reference": False,
    })

manifest_df = pd.DataFrame(oligos)
manifest_df.to_csv(UPLOAD_DIR / "design_manifest.tsv", sep="\t", index=False)
print(f"  {len(manifest_df)} oligos written")

# ── barcode pool ─────────────────────────────────────────────────────────────

print("Generating barcodes…")
all_barcodes = set()
while len(all_barcodes) < N_OLIGOS * N_BARCODES_PER_OLIGO + 500:
    all_barcodes.add(rand_seq(BARCODE_LEN))
all_barcodes = list(all_barcodes)

oligo_ids = manifest_df["oligo_id"].tolist()
barcode_to_oligo = {}
oligo_to_barcodes = {}
bc_pool = iter(all_barcodes)

for oid in oligo_ids:
    bcs = [next(bc_pool) for _ in range(N_BARCODES_PER_OLIGO)]
    oligo_to_barcodes[oid] = bcs
    for bc in bcs:
        barcode_to_oligo[bc] = oid

# inject ~3% collisions (barcodes mapping to 2 oligos)
n_collisions = int(len(barcode_to_oligo) * 0.03)
collision_bcs = random.sample(list(barcode_to_oligo.keys()), n_collisions)
collision_targets = random.sample(oligo_ids, n_collisions)
for bc, oid in zip(collision_bcs, collision_targets):
    barcode_to_oligo[bc] = oid  # reassign to different oligo

# ── mapping table ─────────────────────────────────────────────────────────────

print("Building mapping table…")
seq_lookup = dict(zip(manifest_df["oligo_id"], manifest_df["sequence"]))
rows = []

for oid in oligo_ids:
    ref_seq = seq_lookup[oid]
    for bc in oligo_to_barcodes[oid]:
        n_reads = max(1, int(rng.poisson(50)))

        # ~70% perfect, ~20% 1 mismatch, ~10% soft-clipped
        r = rng.random()
        if r < 0.70:
            obs = ref_seq
        elif r < 0.90:
            obs = mutate(ref_seq, n_mut=rng.integers(1, 4))
        else:
            clip = rng.integers(5, 20)
            obs = ref_seq[:-clip]  # truncated → soft clip

        cigar, md = make_cigar_md(obs, ref_seq)
        rows.append({
            "barcode": bc,
            "oligo_id": barcode_to_oligo[bc],  # use (possibly collided) mapping
            "cigar": cigar,
            "md": md,
            "n_reads": n_reads,
        })

mapping_df = pd.DataFrame(rows)
mapping_df.to_csv(UPLOAD_DIR / "mapping_table.tsv", sep="\t", index=False)
print(f"  {len(mapping_df)} barcode rows written")

# ── plasmid counts ────────────────────────────────────────────────────────────

print("Building plasmid counts…")
plasmid_df = mapping_df[["barcode", "oligo_id"]].copy()
plasmid_df["dna_count"] = mapping_df["n_reads"].values
# zero out 5% of barcodes to simulate low-depth entries
zero_mask = rng.random(len(plasmid_df)) < 0.05
plasmid_df.loc[zero_mask, "dna_count"] = 0
plasmid_df.to_csv(UPLOAD_DIR / "plasmid_counts.tsv", sep="\t", index=False)
print(f"  {len(plasmid_df)} rows written")

# ── RNA counts ────────────────────────────────────────────────────────────────

print("Building RNA counts…")

def activity_level(oid):
    cat = manifest_df.loc[manifest_df["oligo_id"] == oid, "designed_category"].values[0]
    if cat == "positive_control":
        return 4.0   # very active
    elif cat == "negative_control":
        return -0.5  # inactive
    else:
        # test elements: ~40% active
        return rng.choice([2.5, -0.3], p=[0.4, 0.6])

rna_rows = plasmid_df[["barcode", "oligo_id"]].copy()
dna_sf = max(plasmid_df["dna_count"].sum() / 1e6, 1e-9)

for rep in ["rep1", "rep2"]:
    rna_counts = []
    for _, row in plasmid_df.iterrows():
        dna = row["dna_count"]
        log2_target = activity_level(row["oligo_id"])
        noise = rng.normal(0, 0.3)
        rna_sf = 1.0
        rna_raw = max(0, int((2 ** (log2_target + noise)) * (dna + 0.5) * rna_sf))
        rna_counts.append(rna_raw)
    rna_rows[f"rna_count_{rep}"] = rna_counts

rna_rows.to_csv(UPLOAD_DIR / "rna_counts.tsv", sep="\t", index=False)
print(f"  {len(rna_rows)} rows, 2 replicates written")

# ── run activity analysis ─────────────────────────────────────────────────────

print("Running activity analysis…")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from creseq_mcp.qc.activity import activity_report

_, summary = activity_report(
    UPLOAD_DIR / "plasmid_counts.tsv",
    UPLOAD_DIR / "rna_counts.tsv",
    UPLOAD_DIR / "design_manifest.tsv",
    upload_dir=UPLOAD_DIR,
)
print(f"  Active: {summary['n_active']} / {summary['n_oligos_after_filter']} ({summary['activity_rate']:.1%})")

print(f"\nAll files written to {UPLOAD_DIR}")
print("  mapping_table.tsv")
print("  plasmid_counts.tsv")
print("  design_manifest.tsv")
print("  rna_counts.tsv")
print("  activity_results.tsv")
print("\nGo to Chat and ask the agent to run QC.")
