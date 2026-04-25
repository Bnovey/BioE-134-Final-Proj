"""
creseq_mcp/qc/activity.py
==========================
Normalization and activity calling for lentiMPRA / CRE-seq.

Steps
-----
1. RPM-style size-factor normalization (per sample)
2. log2(RNA/DNA) per barcode, averaged across replicates
3. Collapse to per-oligo median (requiring min_barcodes)
4. Activity calling: z-test vs. negative control distribution → BH FDR
   Falls back to log2_ratio > 1 threshold when controls are absent.
"""
from __future__ import annotations

import logging
import re
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PSEUDO = 0.5  # pseudocount added before log2


def _norm_sf(z: float) -> float:
    """One-sided p-value (upper tail) for a standard-normal z-score."""
    return 0.5 * erfc(z / sqrt(2))


def _bh_fdr(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction — no external dependencies."""
    n = len(pvals)
    arr = np.array(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        ranked[i] = min(ranked[i], ranked[i + 1])
    result = np.empty(n)
    result[order] = np.clip(ranked, 0, 1)
    return result


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


def call_activity(
    oligo_df: pd.DataFrame,
    *,
    neg_ctrl_category: str = "negative_control",
    fdr_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Call active CREs relative to negative control distribution.
    Falls back to log2_ratio > 1 threshold when controls are absent.
    """
    df = oligo_df.copy()

    neg_ratios = pd.Series([], dtype=float)
    if "designed_category" in df.columns:
        neg_mask = df["designed_category"] == neg_ctrl_category
        neg_ratios = df.loc[neg_mask, "log2_ratio"].dropna()

    if len(neg_ratios) >= 3:
        neg_mean = float(neg_ratios.mean())
        neg_std = max(float(neg_ratios.std()), 1e-9)

        pvals = [_norm_sf((r - neg_mean) / neg_std) for r in df["log2_ratio"]]
        fdrs = _bh_fdr(pvals)
        df["pval"] = pvals
        df["fdr"] = fdrs
        df["active"] = df["fdr"] < fdr_threshold

        return df, {
            "method": "z_test_vs_neg_ctrl",
            "n_neg_controls": int(neg_mask.sum()),
            "neg_ctrl_mean_log2": round(neg_mean, 4),
            "fdr_threshold": fdr_threshold,
            "n_active": int(df["active"].sum()),
            "n_inactive": int((~df["active"]).sum()),
            "activity_rate": round(float(df["active"].mean()), 4),
        }

    df["pval"] = np.nan
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

    df = results.merge(
        manifest[["oligo_id", "variant_family", "is_reference"]],
        on="oligo_id", how="left",
    )

    families = df["variant_family"].dropna().unique()
    rows = []
    skipped = 0

    for fam in families:
        fam_df = df[df["variant_family"] == fam]
        ref_rows = fam_df[fam_df["is_reference"] == True]
        if len(ref_rows) == 0:
            skipped += 1
            continue
        ref_log2 = float(ref_rows["log2_ratio"].iloc[0])
        mutants = fam_df[fam_df["is_reference"] != True]
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
    """Full pipeline: normalize → call activity → optionally save activity_results.tsv."""
    oligo_df, norm_summary = normalize_and_compute_ratios(
        dna_counts_path, rna_counts_path, design_manifest_path
    )
    results_df, call_summary = call_activity(oligo_df)

    if upload_dir is not None:
        results_df.to_csv(upload_dir / "activity_results.tsv", sep="\t", index=False)

    return results_df, {**norm_summary, **call_summary}
