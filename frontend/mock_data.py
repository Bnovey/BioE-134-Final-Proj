import numpy as np
import pandas as pd


def generate_cre_seq_data(n_elements: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    element_ids = [f"CRE_{i:04d}" for i in range(n_elements)]
    chromosomes = rng.choice(["chr1", "chr2", "chr3", "chr4", "chr5", "chrX"], n_elements)
    starts = rng.integers(1_000_000, 200_000_000, n_elements)
    ends = starts + rng.integers(150, 500, n_elements)

    dna_counts = rng.negative_binomial(20, 0.4, n_elements).astype(float) + 1
    # ~30% of elements are active (higher RNA/DNA)
    active_mask = rng.random(n_elements) < 0.30
    rna_counts = np.where(
        active_mask,
        dna_counts * rng.lognormal(1.5, 0.6, n_elements),
        dna_counts * rng.lognormal(-0.5, 0.4, n_elements),
    )
    rna_counts = np.maximum(rna_counts, 1).astype(float)

    log2_ratio = np.log2(rna_counts / dna_counts)
    # p-value proxy: active elements get low p-values
    pval = np.where(active_mask, rng.beta(0.5, 5, n_elements), rng.beta(5, 1, n_elements))
    active_called = (log2_ratio > 1.0) & (pval < 0.05)

    gc_content = rng.beta(5, 5, n_elements)
    chromatin_state = rng.choice(
        ["Active Enhancer", "Weak Enhancer", "Promoter", "Heterochromatin", "Quiescent"],
        n_elements,
        p=[0.15, 0.20, 0.10, 0.30, 0.25],
    )
    top_motif = rng.choice(
        ["SP1", "CTCF", "GATA1", "AP1", "NRF1", "ETS1", "YY1", "None"],
        n_elements,
        p=[0.12, 0.10, 0.10, 0.12, 0.08, 0.08, 0.10, 0.30],
    )

    df = pd.DataFrame(
        {
            "element_id": element_ids,
            "chrom": chromosomes,
            "start": starts,
            "end": ends,
            "dna_counts": dna_counts.round(1),
            "rna_counts": rna_counts.round(1),
            "log2_ratio": log2_ratio.round(3),
            "pval": pval.round(4),
            "active": active_called,
            "gc_content": gc_content.round(3),
            "chromatin_state": chromatin_state,
            "top_motif": top_motif,
        }
    )
    return df


def get_demo_data() -> pd.DataFrame:
    return generate_cre_seq_data()
