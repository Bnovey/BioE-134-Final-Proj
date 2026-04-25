"""
End-to-end demo: generator → normalize → call_active → extract_sequences →
motif_enrichment → plots.

All intermediate and final outputs are written to ``demo_outputs/`` at the
project root.  The pipeline reads the generator's output (per-barcode counts +
per-oligo design manifest), bridges the schema joints between the per-barcode
tables and the per-element analysis tools, and stitches the modules together.

Schema joints handled here:
  - per-barcode plasmid + RNA counts → per-element counts (sum across barcodes,
    sum across replicates) for ``normalize_activity``;
  - ``log2_activity`` → ``mean_activity`` rename, plus per-element std + barcode
    counts pulled from the underlying per-barcode log2 values, so
    ``call_active_elements_empirical`` gets a complete per-element table;
  - manifest column ``oligo_id`` → ``element_id`` for ``extract_sequences``.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "demo_outputs"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path.home() / ".creseq" / "uploads"


def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ─── Step 1: Generator ───────────────────────────────────────────────────────

_hr("STEP 1 · Run generator")
import subprocess

gen_proc = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "generate_test_data.py")],
    capture_output=True, text=True, check=True,
)
print(gen_proc.stdout.strip().splitlines()[-1])
print("Generator outputs in:", UPLOAD_DIR)
for f in sorted(UPLOAD_DIR.glob("*.tsv")):
    print(f"  {f.name:25s}  {f.stat().st_size:>9,} bytes")


# ─── Step 2: Aggregate per-barcode → per-element + normalize_activity ────────

_hr("STEP 2 · normalize_activity")

plasmid = pd.read_csv(UPLOAD_DIR / "plasmid_counts.tsv", sep="\t")
rna = pd.read_csv(UPLOAD_DIR / "rna_counts.tsv", sep="\t")
manifest = pd.read_csv(UPLOAD_DIR / "design_manifest.tsv", sep="\t")

rep_cols = [c for c in rna.columns if c.startswith("rna_count_")]
print(f"Per-barcode rows: {len(plasmid):,} (DNA), {len(rna):,} (RNA), reps={rep_cols}")

# Per-barcode RNA total across replicates
rna["rna_total"] = rna[rep_cols].sum(axis=1)
bc = plasmid.merge(
    rna[["barcode", "oligo_id", "rna_total"]], on=["barcode", "oligo_id"], how="inner"
)

# Per-element aggregation: sum counts, count barcodes
agg = (
    bc.groupby("oligo_id")
    .agg(dna_counts=("dna_count", "sum"),
         rna_counts=("rna_total", "sum"),
         n_barcodes=("barcode", "count"))
    .reset_index()
    .rename(columns={"oligo_id": "element_id"})
)
elem_counts_path = DEMO_DIR / "element_counts.tsv"
agg.to_csv(elem_counts_path, sep="\t", index=False)
print(f"Aggregated to {len(agg):,} elements → {elem_counts_path.name}")

from creseq_mcp.stats.library import normalize_activity

norm_df, norm_summary = normalize_activity(str(elem_counts_path))
print("normalize_activity summary:", json.dumps(norm_summary, indent=2, default=str))

# Per-barcode log2 dispersion → per-element std for the empirical caller
bc["log2_bc"] = np.log2((bc["rna_total"] + 1.0) / (bc["dna_count"] + 1.0))
bc_std = (
    bc.groupby("oligo_id")["log2_bc"].std()
    .reset_index()
    .rename(columns={"oligo_id": "element_id", "log2_bc": "std_activity"})
)

activity_table = (
    norm_df[["element_id", "log2_activity"]]
    .rename(columns={"log2_activity": "mean_activity"})
    .merge(agg[["element_id", "n_barcodes"]], on="element_id")
    .merge(bc_std, on="element_id", how="left")
)
activity_path = DEMO_DIR / "activity_table.tsv"
activity_table.to_csv(activity_path, sep="\t", index=False)
print(
    f"Activity table → {activity_path.name}  "
    f"(median mean_activity={activity_table['mean_activity'].median():.3f})"
)


# ─── Step 3: call_active_elements ────────────────────────────────────────────

_hr("STEP 3 · call_active_elements")
from creseq_mcp.activity_calling import call_active_elements

# Negative control IDs come from the design manifest's designed_category column.
neg_ids = manifest.loc[
    manifest["designed_category"] == "negative_control", "oligo_id"
].astype(str).tolist()
print(f"Negative controls: {len(neg_ids)} IDs (e.g. {neg_ids[:3]}...)")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    call_result = call_active_elements(
        activity_table_path=str(activity_path),
        negative_controls=neg_ids,
        fdr_threshold=0.05,
        method="empirical",
        output_path=str(DEMO_DIR / "classified_elements.tsv"),
    )

call_summary = call_result["summary"]
print(f"Classified table → {Path(call_result['classified_elements']).name}")
print(json.dumps(
    {k: v for k, v in call_summary.items() if k not in ("warnings",)},
    indent=2, default=str,
))
if call_summary.get("warnings"):
    print("Warnings:")
    for w in call_summary["warnings"]:
        print(f"  - {w}")
for w in caught:
    print(f"  [pyWarning] {w.category.__name__}: {w.message}")


# ─── Step 4: extract_sequences_to_fasta ──────────────────────────────────────

_hr("STEP 4 · extract_sequences_to_fasta")

# The generator's manifest uses oligo_id; the bridge function requires
# element_id.  Write a renamed copy alongside the demo outputs.
manifest_for_extract = manifest.rename(columns={"oligo_id": "element_id"})
manifest_for_extract_path = DEMO_DIR / "manifest_with_sequences.tsv"
manifest_for_extract[["element_id", "sequence"]].to_csv(
    manifest_for_extract_path, sep="\t", index=False
)

from creseq_mcp.motif import extract_sequences_to_fasta

extract_result = extract_sequences_to_fasta(
    classified_table=call_result["classified_elements"],
    sequence_source=str(manifest_for_extract_path),
    active_output=str(DEMO_DIR / "active.fa"),
    background_output=str(DEMO_DIR / "background.fa"),
)
print(json.dumps(extract_result, indent=2))


# ─── Step 5: motif_enrichment (real JASPAR — ~30s) ───────────────────────────

_hr("STEP 5 · motif_enrichment (JASPAR2024 CORE Vertebrates)")
from creseq_mcp.motif import motif_enrichment

enrich_result = motif_enrichment(
    active_fasta=extract_result["active_fasta"],
    background_fasta=extract_result["background_fasta"],
    motif_database="JASPAR2024",
    score_threshold=0.85,
    output_path=str(DEMO_DIR / "motif_enrichment.tsv"),
)
print(f"Enrichment table → {Path(enrich_result['enrichment_table']).name}")
print(enrich_result["summary"])

# Quick inspection of the top of the table
enr_df = pd.read_csv(enrich_result["enrichment_table"], sep="\t")
print(f"\nMotifs evaluated: {len(enr_df)} | FDR<0.05: {(enr_df['fdr']<0.05).sum()}")
print(enr_df.head(8).to_string(index=False))


# ─── Step 6: plot_creseq ─────────────────────────────────────────────────────

_hr("STEP 6 · plot_creseq")
from creseq_mcp.plotting import plot_creseq

volcano = plot_creseq(
    data_file=call_result["classified_elements"],
    plot_type="volcano",
    output_path=str(DEMO_DIR / "volcano.png"),
    neg_control_ids=neg_ids,
)
print(f"Volcano → {Path(volcano['plot_path']).name}")
print("  ", volcano["description"])

dotplot = plot_creseq(
    data_file=enrich_result["enrichment_table"],
    plot_type="motif_dotplot",
    output_path=str(DEMO_DIR / "motif_dotplot.png"),
)
print(f"Motif dot plot → {Path(dotplot['plot_path']).name}")
print("  ", dotplot["description"])


# ─── Wrap up ─────────────────────────────────────────────────────────────────

_hr("DONE · contents of demo_outputs/")
for f in sorted(DEMO_DIR.iterdir()):
    print(f"  {f.name:35s}  {f.stat().st_size:>9,} bytes")
