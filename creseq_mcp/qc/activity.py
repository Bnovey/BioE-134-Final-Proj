"""
creseq_mcp/qc/activity.py
==========================
Normalization and activity calling for lentiMPRA / CRE-seq.

Steps
-----
1. RPM-style size-factor normalization (per sample)
2. log2(RNA/DNA) per barcode, averaged across replicates
3. Collapse to per-oligo median (requiring min_barcodes)
4. Activity calling: Sarrah's empirical null (median/MAD, BH FDR)
   Falls back to log2_ratio > 1 threshold when controls are absent.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PSEUDO = 0.5  # pseudocount added before log2


def normalize_and_compute_ratios(
    dna_counts_path: str | Path,
    rna_counts_path: str | Path,
    design_manifest_path: str | Path | None = None,
    *,
    min_barcodes: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize DNA and RNA counts → per-barcode log2(RNA/DNA) → collapse to per-oligo.
    """
    dna = pd.read_csv(dna_counts_path, sep="\t")
    rna = pd.read_csv(rna_counts_path, sep="\t")

    rep_cols = [c for c in rna.columns if c.startswith("rna_count_")]
    if not rep_cols:
        raise ValueError("rna_counts.tsv has no rna_count_* columns")

    merged = dna[["barcode", "oligo_id", "dna_count"]].merge(
        rna[["barcode"] + rep_cols], on="barcode", how="inner"
    )

    dna_sf = max(merged["dna_count"].sum() / 1e6, 1e-9)
    merged["norm_dna"] = (merged["dna_count"] + _PSEUDO) / dna_sf

    log2_cols = []
    for col in rep_cols:
        sf = max(merged[col].sum() / 1e6, 1e-9)
        norm_col = f"norm_{col}"
        merged[norm_col] = (merged[col] + _PSEUDO) / sf
        log2_col = f"log2_{col}"
        merged[log2_col] = np.log2(merged[norm_col] / merged["norm_dna"])
        log2_cols.append(log2_col)

    merged["log2_ratio"] = merged[log2_cols].mean(axis=1)

    oligo_df = (
        merged.groupby("oligo_id")
        .agg(**{
            "n_barcodes": ("barcode", "count"),
            "median_dna": ("dna_count", "median"),
            "log2_ratio": ("log2_ratio", "median"),
            **{col: (col, "median") for col in log2_cols},
        })
        .reset_index()
    )

    oligo_df = oligo_df[oligo_df["n_barcodes"] >= min_barcodes].copy()

    if design_manifest_path and Path(design_manifest_path).exists():
        manifest = pd.read_csv(design_manifest_path, sep="\t")
        oligo_df = oligo_df.merge(manifest, on="oligo_id", how="left")

    return oligo_df, {
        "n_barcodes_merged": len(merged),
        "n_oligos_after_filter": len(oligo_df),
        "min_barcodes_filter": min_barcodes,
        "replicates": rep_cols,
        "median_log2_ratio": float(oligo_df["log2_ratio"].median()),
    }


def _call_activity(
    oligo_df: pd.DataFrame,
    *,
    neg_ctrl_category: str = "negative_control",
    fdr_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Classify CREs using Sarrah's empirical null (median/MAD + BH FDR).
    Falls back to log2_ratio > 1 when <3 controls are present.
    """
    from creseq_mcp.activity_calling import call_active_elements_empirical

    df = oligo_df.copy()

    # Build the element_id / mean_activity columns Sarrah's function expects.
    activity_table = df.rename(columns={"oligo_id": "element_id", "log2_ratio": "mean_activity"})
    if "n_barcodes" not in activity_table.columns:
        activity_table["n_barcodes"] = pd.NA

    neg_ctrl_ids: list[str] = []
    if "designed_category" in df.columns:
        neg_ctrl_ids = (
            df.loc[df["designed_category"] == neg_ctrl_category, "oligo_id"]
            .dropna()
            .tolist()
        )

    if len(neg_ctrl_ids) >= 3:
        classified, summary = call_active_elements_empirical(
            activity_table, neg_ctrl_ids, fdr_threshold
        )
        # Map results back onto oligo_df using oligo_id.
        classified = classified.rename(columns={"element_id": "oligo_id", "mean_activity": "log2_ratio"})
        keep_cols = [c for c in ("active", "pvalue", "fdr", "zscore", "fold_over_controls") if c in classified.columns]
        df = df.merge(classified[["oligo_id"] + keep_cols], on="oligo_id", how="left")
        return df, {
            "method": "empirical_median_mad",
            "n_neg_controls": len(neg_ctrl_ids),
            "fdr_threshold": fdr_threshold,
            "n_active": int(df["active"].sum()),
            "n_inactive": int((~df["active"]).sum()),
            "activity_rate": round(float(df["active"].mean()), 4),
            "warnings": summary.get("warnings", []),
        }

    df["pvalue"] = np.nan
    df["fdr"] = np.nan
    df["active"] = df["log2_ratio"] > 1.0
    return df, {
        "method": "threshold_log2gt1",
        "n_active": int(df["active"].sum()),
        "n_inactive": int((~df["active"]).sum()),
        "activity_rate": round(float(df["active"].mean()), 4),
    }


_LOCUS_RE = re.compile(r"\[([^\]]+)\]")


def _add_variant_cols(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Derive variant_family and is_reference from oligo_id when those columns
    are absent.  Handles the lentiMPRA naming convention:
      R:<TF>_<coords>_[<locus>]  →  reference allele, family = <locus>
      A:<TF>_<coords>_[<locus>]  →  alternate allele, family = <locus>
      C:<TF>_<coords>_[<locus>]  →  control allele,   family = <locus>
      seq#####                   →  no variant family (NaN)
    Falls back to designed_category == "reference" when prefix is ambiguous.
    """
    manifest = manifest.copy()

    def _family(oid: str) -> str | None:
        m = _LOCUS_RE.search(oid)
        return m.group(1) if m else None

    def _is_ref(row) -> bool | None:
        oid = str(row.get("oligo_id", ""))
        if oid.startswith("R:"):
            return True
        if oid.startswith("A:") or oid.startswith("C:"):
            return False
        cat = str(row.get("designed_category", ""))
        if cat == "reference":
            return True
        if cat in ("alternate", "control"):
            return False
        return None

    manifest["variant_family"] = manifest["oligo_id"].apply(_family)
    manifest["is_reference"] = manifest.apply(_is_ref, axis=1)
    return manifest


def compute_variant_delta_scores(
    activity_results_path: str | Path,
    design_manifest_path: str | Path,
    upload_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    For each variant family, compute delta = mutant_log2_ratio − reference_log2_ratio.
    Tests significance via z-test across all deltas (BH FDR).
    Saves variant_delta_scores.tsv when upload_dir is provided.

    If the manifest lacks variant_family / is_reference columns they are
    automatically derived from the oligo_id using the R:/A:/C: prefix
    convention used by lentiMPRA design tables.
    """
    results = pd.read_csv(activity_results_path, sep="\t")
    manifest = pd.read_csv(design_manifest_path, sep="\t")

    if not {"variant_family", "is_reference"}.issubset(manifest.columns):
        manifest = _add_variant_cols(manifest)

    # activity_results.tsv may already carry variant_family / is_reference
    # from an earlier manifest merge; drop them before re-merging so we don't
    # end up with _x/_y suffixed columns.
    results = results.drop(
        columns=[c for c in ("variant_family", "is_reference") if c in results.columns]
    )
    df = results.merge(
        manifest[["oligo_id", "variant_family", "is_reference"]],
        on="oligo_id", how="left",
    )
    # Round-tripping a Python bool through CSV produces the strings "True"/
    # "False"; coerce to a real bool before comparing so siblings are filtered
    # correctly.
    if df["is_reference"].dtype != bool:
        df["is_reference"] = (
            df["is_reference"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )

    families = df["variant_family"].dropna().unique()
    rows = []
    skipped = 0

    for fam in families:
        fam_df = df[df["variant_family"] == fam]
        ref_rows = fam_df[fam_df["is_reference"]]
        if len(ref_rows) == 0:
            skipped += 1
            continue
        ref_log2 = float(ref_rows["log2_ratio"].iloc[0])
        mutants = fam_df[~fam_df["is_reference"]]
        for _, row in mutants.iterrows():
            rows.append({
                "oligo_id": row["oligo_id"],
                "variant_family": fam,
                "ref_log2": ref_log2,
                "mutant_log2": float(row["log2_ratio"]),
                "delta_log2": float(row["log2_ratio"]) - ref_log2,
            })

    if not rows:
        return pd.DataFrame(), {
            "n_families": int(len(families)),
            "n_families_skipped": skipped,
            "n_mutants": 0,
            "warnings": ["No variant families with recovered references found."],
            "pass": False,
        }

    delta_df = pd.DataFrame(rows)

    # z-test across all deltas
    all_deltas = delta_df["delta_log2"].values
    delta_mean = float(all_deltas.mean())
    delta_std = max(float(all_deltas.std(ddof=1)), 1e-9)
    z_scores = (all_deltas - delta_mean) / delta_std
    pvals = [_norm_sf(z) for z in z_scores]
    fdrs = _bh_fdr(pvals)

    delta_df["pval"] = pvals
    delta_df["fdr"] = fdrs
    delta_df["significant"] = delta_df["fdr"] < 0.05

    if upload_dir is not None:
        delta_df.to_csv(Path(upload_dir) / "variant_delta_scores.tsv", sep="\t", index=False)

    summary = {
        "n_families": int(len(families)),
        "n_families_skipped": skipped,
        "n_mutants": int(len(delta_df)),
        "n_significant": int(delta_df["significant"].sum()),
        "median_abs_delta": round(float(delta_df["delta_log2"].abs().median()), 4),
        "warnings": [],
        "pass": True,
    }

    return delta_df, summary


def activity_report(
    dna_counts_path: str | Path,
    rna_counts_path: str | Path,
    design_manifest_path: str | Path | None = None,
    upload_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: normalize → Sarrah's empirical classifier → save activity_results.tsv."""
    oligo_df, norm_summary = normalize_and_compute_ratios(
        dna_counts_path, rna_counts_path, design_manifest_path
    )
    results_df, call_summary = _call_activity(oligo_df)

    if upload_dir is not None:
        results_df.to_csv(upload_dir / "activity_results.tsv", sep="\t", index=False)

    return results_df, {**norm_summary, **call_summary}
