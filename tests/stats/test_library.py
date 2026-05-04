"""
tests/stats/test_library.py
============================
Tests for creseq_mcp/literature/search.py.

Coverage:
  - rank_cre_candidates: rank ordering, top_element, q_col fallback
  - motif_enrichment_summary: missing column raises ValueError
  - build_pubmed_query: synonym-aware query construction
  - score_evidence_records: evidence relevance flags and sorting
  - search_pubmed: mocked HTTP response returns DataFrame with title column
  - search_jaspar_motif: mocked HTTP response returns matrix_id
  - literature_search_for_motifs: mocked combined API search and TSV output
  - prepare_literature_rag_context: RAG chunk filtering, citations, snippets
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from creseq_mcp.literature.search import (
    build_pubmed_query,
    build_queries,
    classify_evidence,
    interpret_literature_evidence,
    literature_search_for_motifs,
    motif_enrichment_summary,
    prepare_literature_rag_context,
    rank_cre_candidates,
    score_evidence_records,
    search_encode_tf,
    search_jaspar_motif,
    search_pubmed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _activity_tsv(tmp_path: Path, n=60, seed=7) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i < 20:
            cat, base = "negative_control", -0.4
        elif i < 30:
            cat, base = "positive_control", 3.0
        else:
            cat, base = "test_element", float(rng.choice([-0.2, 2.5]))
        rows.append({
            "oligo_id": f"O{i:03d}",
            "log2_ratio": base + float(rng.normal(0, 0.15)),
            "designed_category": cat,
        })
    p = tmp_path / "activity_results.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


# ---------------------------------------------------------------------------
# rank_cre_candidates
# ---------------------------------------------------------------------------

class TestRankCrCandidates:
    def test_rank_ordering(self, tmp_path):
        p = _activity_tsv(tmp_path)
        df, summary = rank_cre_candidates(p, top_n=10, activity_col="log2_ratio", q_col="q_value")

        assert list(df["rank"]) == list(range(1, len(df) + 1))
        assert len(df) == 10
        assert summary["top_element"] is not None

    def test_missing_q_col_defaults_to_one(self, tmp_path):
        # Table has no q_value column → should not raise, q defaults to 1.0
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i)} for i in range(20)]
        p = tmp_path / "no_q.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = rank_cre_candidates(str(p), top_n=5, activity_col="log2_ratio", q_col="q_value")
        assert len(df) == 5
        assert (df["q_value"] == 1.0).all()

    def test_top_n_larger_than_table(self, tmp_path):
        rows = [{"oligo_id": f"O{i}", "log2_ratio": float(i)} for i in range(5)]
        p = tmp_path / "small.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = rank_cre_candidates(str(p), top_n=50, activity_col="log2_ratio")
        assert len(df) == 5


# ---------------------------------------------------------------------------
# motif_enrichment_summary
# ---------------------------------------------------------------------------

class TestMotifEnrichmentSummary:
    def test_missing_motif_col_raises(self, tmp_path):
        rows = [{"oligo_id": "O1", "log2_ratio": 1.0, "active": True}]
        p = tmp_path / "no_motif.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        with pytest.raises(ValueError, match="top_motif"):
            motif_enrichment_summary(str(p))

    def test_happy_path(self, tmp_path):
        rows = [
            {"oligo_id": f"O{i}", "active": i % 2 == 0, "top_motif": "SP1" if i % 3 == 0 else "GATA1"}
            for i in range(30)
        ]
        p = tmp_path / "motifs.tsv"
        pd.DataFrame(rows).to_csv(p, sep="\t", index=False)

        df, summary = motif_enrichment_summary(str(p))
        assert "enrichment_ratio" in df.columns
        assert summary["n_motifs_tested"] == 2


# ---------------------------------------------------------------------------
# PubMed query construction and evidence scoring
# ---------------------------------------------------------------------------

class TestQueryAndScoring:
    def test_build_queries_returns_multi_intent_boolean_queries(self):
        queries = build_queries("NRF2", cell_type="HepG2", species="human")

        assert set(queries) == {"mpra", "binding", "motif", "perturbation"}
        assert "NFE2L2[Title/Abstract]" in queries["mpra"]
        assert '"Homo sapiens"[Title/Abstract]' in queries["binding"]
        assert '"ChIP-seq"[Title/Abstract]' in queries["binding"]
        assert "CRISPR[Title/Abstract]" in queries["perturbation"]
        assert "NOT (yeast[Title/Abstract] OR drosophila[Title/Abstract])" in queries["motif"]

    def test_build_pubmed_query_expands_tf_synonyms_and_context(self):
        query = build_pubmed_query("NRF2", target_cell_type="HepG2", off_target_cell_type="K562")

        assert "NRF2[Title/Abstract]" in query
        assert "NFE2L2[Title/Abstract]" in query
        assert '"massively parallel reporter assay"[Title/Abstract]' in query
        assert "HepG2[Title/Abstract]" in query
        assert "NOT (K562[Title/Abstract])" in query

    def test_classify_evidence_prefers_direct_mpra(self):
        record = {
            "title": "NRF2 enhancer MPRA in HepG2",
            "abstract": "Massively parallel reporter assay identifies enhancer activation.",
        }

        assert classify_evidence(record) == "MPRA"

    def test_score_evidence_records_ranks_relevant_pubmed_first(self):
        evidence = pd.DataFrame([
            {
                "source": "JASPAR",
                "motif": "GATA1",
                "name": "GATA1",
                "pubdate": "",
            },
            {
                "source": "PubMed",
                "motif": "GATA1",
                "title": "GATA binding protein 1 controls HepG2 enhancer activity in MPRA",
                "journal": "Genome Research",
                "pubdate": "2024",
            },
        ])

        scored = score_evidence_records(evidence, target_cell_type="HepG2")

        assert scored.iloc[0]["source"] == "PubMed"
        assert bool(scored.iloc[0]["tf_match"]) is True
        assert bool(scored.iloc[0]["cell_type_match"]) is True
        assert bool(scored.iloc[0]["assay_keyword_match"]) is True
        assert scored.iloc[0]["evidence_class"] == "MPRA"
        assert scored.iloc[0]["confidence"] in {"medium", "high"}
        assert scored.iloc[0]["evidence_score"] > scored.iloc[1]["evidence_score"]

    def test_score_evidence_records_does_not_count_query_terms(self):
        evidence = pd.DataFrame([{
            "source": "PubMed",
            "motif": "GATA1",
            "query": "GATA1 HepG2 enhancer MPRA",
            "title": "Unrelated metabolism study",
            "abstract": "This paper does not discuss the requested cell or assay context.",
        }])

        scored = score_evidence_records(evidence, target_cell_type="HepG2")

        assert bool(scored.iloc[0]["cell_type_match"]) is False
        assert bool(scored.iloc[0]["regulatory_keyword_match"]) is False
        assert bool(scored.iloc[0]["assay_keyword_match"]) is False


# ---------------------------------------------------------------------------
# search_pubmed (mocked)
# ---------------------------------------------------------------------------

class TestSearchPubmed:
    def _mock_response(self, ids, summaries):
        esearch_data = {"esearchresult": {"idlist": ids}}
        esummary_data = {"result": summaries}
        abstract_xml = (
            "<PubmedArticleSet>"
            + "".join(
                f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
                f"<Article><Abstract><AbstractText>{summaries.get(pmid, {}).get('abstract', '')}</AbstractText>"
                f"</Abstract></Article></MedlineCitation></PubmedArticle>"
                for pmid in ids
            )
            + "</PubmedArticleSet>"
        )

        call_count = 0

        def fake_get(url, params=None, timeout=15, headers=None):
            nonlocal call_count
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            if call_count == 0:
                mock.json.return_value = esearch_data
            elif call_count == 1:
                mock.json.return_value = esummary_data
            else:
                mock.text = abstract_xml
            call_count += 1
            return mock

        return fake_get

    def test_returns_dataframe_with_title(self):
        ids = ["12345678"]
        summaries = {
            "12345678": {
                "title": "GATA1 enhancer MPRA study",
                "fulljournalname": "Nature Genetics",
                "pubdate": "2024",
                "abstract": "GATA1 regulates enhancer activity in K562 cells using reporter assays.",
                "authors": [{"name": "Smith A"}],
            }
        }
        fake_get = self._mock_response(ids, summaries)

        with patch("creseq_mcp.literature.search.requests.get", side_effect=fake_get), \
             patch("creseq_mcp.literature.search.time.sleep"):
            df, summary = search_pubmed("GATA1 enhancer MPRA", max_results=1)

        assert len(df) == 1
        assert df.iloc[0]["title"] == "GATA1 enhancer MPRA study"
        assert "K562" in df.iloc[0]["abstract"]
        assert summary["n_results"] == 1

    def test_no_results(self):
        fake_get = self._mock_response([], {})

        with patch("creseq_mcp.literature.search.requests.get", side_effect=fake_get), \
             patch("creseq_mcp.literature.search.time.sleep"):
            df, summary = search_pubmed("xyzzy gibberish query", max_results=5)

        assert len(df) == 0
        assert summary["n_results"] == 0


# ---------------------------------------------------------------------------
# search_jaspar_motif (mocked)
# ---------------------------------------------------------------------------

class TestSearchJasparMotif:
    def test_returns_matrix_id(self):
        mock_data = {
            "results": [
                {
                    "matrix_id": "MA0139.1",
                    "name": "CTCF",
                    "collection": "CORE",
                    "tax_group": "vertebrates",
                    "species": [{"name": "Homo sapiens"}],
                    "class": ["C2H2 zinc finger factors"],
                    "family": ["CTCF-related"],
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_data

        with patch("creseq_mcp.literature.search.requests.get", return_value=mock_resp):
            df, summary = search_jaspar_motif("CTCF")

        assert len(df) == 1
        assert df.iloc[0]["matrix_id"] == "MA0139.1"
        assert summary["n_results"] == 1

    def test_no_results_warning(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": []}

        with patch("creseq_mcp.literature.search.requests.get", return_value=mock_resp):
            df, summary = search_jaspar_motif("NOTAREALFACTOR999")

        assert len(df) == 0
        assert any("No JASPAR" in w for w in summary["warnings"])

    def test_filters_fuzzy_nonmatching_results(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"matrix_id": "MA0046.1", "name": "HNF1A"},
                {"matrix_id": "MA0114.4", "name": "HNF4A"},
            ]
        }

        with patch("creseq_mcp.literature.search.requests.get", return_value=mock_resp):
            df, summary = search_jaspar_motif("HNF4A", max_results=5)

        assert len(df) == 1
        assert df.iloc[0]["name"] == "HNF4A"
        assert summary["n_results"] == 1


# ---------------------------------------------------------------------------
# search_encode_tf (mocked)
# ---------------------------------------------------------------------------

class TestSearchEncodeTf:
    def test_filters_nonmatching_targets(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "@graph": [
                {
                    "accession": "ENCSR_BAD",
                    "assay_title": "TF ChIP-seq",
                    "target": {"label": "CTCF"},
                    "biosample_ontology": {"term_name": "HepG2"},
                    "status": "released",
                },
                {
                    "accession": "ENCSR_GOOD",
                    "assay_title": "TF ChIP-seq",
                    "target": {"label": "NFE2L2"},
                    "biosample_ontology": {"term_name": "HepG2"},
                    "status": "released",
                },
            ]
        }

        with patch("creseq_mcp.literature.search.requests.get", return_value=mock_resp):
            df, summary = search_encode_tf("NRF2", cell_type="HepG2", max_results=5)

        assert len(df) == 1
        assert df.iloc[0]["accession"] == "ENCSR_GOOD"
        assert summary["n_results"] == 1


# ---------------------------------------------------------------------------
# literature_search_for_motifs and interpretation
# ---------------------------------------------------------------------------

class TestLiteratureSearchForMotifs:
    def test_combined_search_accepts_tf_name_column_scores_and_writes_output(self, tmp_path):
        motif_table = tmp_path / "motif_enrichment.tsv"
        pd.DataFrame([
            {"tf_name": "NRF2", "odds_ratio": 5.0, "fdr": 0.001},
            {"tf_name": "GATA1", "odds_ratio": 2.0, "fdr": 0.02},
        ]).to_csv(motif_table, sep="\t", index=False)

        def fake_pubmed(query, max_results=3, email=None, api_key=None):
            return pd.DataFrame([{
                "source": "PubMed",
                "query": query,
                "pmid": "1",
                "title": "NFE2L2 NRF2 enhancer MPRA in HepG2",
                "journal": "Genome Biology",
                "pubdate": "2024",
                "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
            }]), {"warnings": [], "pass": True}

        def fake_jaspar(tf_name, species=9606, collection="CORE", max_results=3):
            return pd.DataFrame([{
                "source": "JASPAR",
                "tf_name": tf_name,
                "matrix_id": "MA0000.1",
                "name": tf_name,
            }]), {"warnings": [], "pass": True}

        def fake_encode(tf_name, cell_type=None, max_results=3):
            return pd.DataFrame([{
                "source": "ENCODE",
                "tf_name": tf_name,
                "cell_type_query": cell_type,
                "accession": "ENCSR000AAA",
                "assay_title": "TF ChIP-seq",
                "target_label": tf_name,
                "biosample_term_name": cell_type,
            }]), {"warnings": [], "pass": True}

        output_path = tmp_path / "literature_evidence.tsv"
        with patch("creseq_mcp.literature.search.search_pubmed", side_effect=fake_pubmed), \
             patch("creseq_mcp.literature.search.search_jaspar_motif", side_effect=fake_jaspar), \
             patch("creseq_mcp.literature.search.search_encode_tf", side_effect=fake_encode), \
             patch("creseq_mcp.literature.search.time.sleep"):
            df, summary = literature_search_for_motifs(
                str(motif_table),
                target_cell_type="HepG2",
                top_n_motifs=1,
                output_path=output_path,
            )

        assert summary["motifs_searched"] == ["NRF2"]
        assert summary["output_path"] == str(output_path)
        assert output_path.exists()
        assert {"evidence_score", "evidence_class", "confidence", "tf_match", "tf_synonyms"}.issubset(df.columns)
        assert df["evidence_score"].notna().all()
        assert "NFE2L2" in df.iloc[0]["tf_synonyms"]

    def test_interpret_literature_evidence_uses_scores(self, tmp_path):
        evidence_path = tmp_path / "evidence.tsv"
        pd.DataFrame([
            {"source": "PubMed", "motif": "SP1", "title": "SP1 enhancer reporter assay", "evidence_score": 6.0},
            {"source": "JASPAR", "motif": "GATA1", "name": "GATA1", "evidence_score": 3.0},
        ]).to_csv(evidence_path, sep="\t", index=False)

        df, summary = interpret_literature_evidence(str(evidence_path))

        assert len(df) == 3
        assert summary["motif_top_scores"]["SP1"] == 6.0
        assert "strongest retrieved evidence is for SP1" in summary["interpretation"]

    def test_pubmed_falls_back_when_target_cell_query_is_empty(self, tmp_path):
        motif_table = tmp_path / "motifs.tsv"
        pd.DataFrame([{"motif": "HNF4A", "enrichment_ratio": 3.0}]).to_csv(
            motif_table, sep="\t", index=False
        )

        calls = []

        def fake_pubmed(query, max_results=3, email=None, api_key=None):
            calls.append(query)
            if len(calls) == 1:
                return pd.DataFrame(), {"warnings": ["No PubMed results found."], "pass": True}
            return pd.DataFrame([{
                "source": "PubMed",
                "query": query,
                "pmid": "2",
                "title": "HNF4A promoter regulation study",
                "abstract": "HNF4A controls promoter activity in hepatic transcriptional regulation.",
                "pubdate": "2024",
            }]), {"warnings": [], "pass": True}

        with patch("creseq_mcp.literature.search.search_pubmed", side_effect=fake_pubmed), \
             patch("creseq_mcp.literature.search.search_jaspar_motif", return_value=(pd.DataFrame(), {"warnings": [], "pass": True})), \
             patch("creseq_mcp.literature.search.search_encode_tf", return_value=(pd.DataFrame(), {"warnings": [], "pass": True})), \
             patch("creseq_mcp.literature.search.time.sleep"):
            df, summary = literature_search_for_motifs(
                str(motif_table),
                target_cell_type="HepG2",
                top_n_motifs=1,
                multi_intent_queries=False,
            )

        assert len(calls) == 2
        assert "HepG2" in calls[0]
        assert "HepG2" not in calls[1]
        assert df.iloc[0]["query_scope"] == "tf_regulatory_fallback"
        assert any("broader TF/regulatory query" in w for w in summary["warnings"])


# ---------------------------------------------------------------------------
# prepare_literature_rag_context
# ---------------------------------------------------------------------------

class TestPrepareLiteratureRagContext:
    def test_filters_sorts_and_writes_citation_ready_context(self, tmp_path):
        evidence_path = tmp_path / "evidence.tsv"
        output_path = tmp_path / "rag_context.tsv"
        pd.DataFrame([
            {
                "source": "PubMed",
                "motif": "NRF2",
                "evidence_type": "literature",
                "pmid": "32293113",
                "title": "Nrf2 regulates enhancer-linked transcription in HepG2",
                "abstract": (
                    "NFE2L2/NRF2 controls transcriptional regulation in HepG2 cells. "
                    "Reporter assay results support enhancer activity at oxidative stress loci."
                ),
                "url": "https://pubmed.ncbi.nlm.nih.gov/32293113/",
                "tf_match": True,
                "cell_type_match": True,
                "regulatory_keyword_match": True,
                "assay_keyword_match": True,
                "evidence_score": 7.0,
            },
            {
                "source": "JASPAR",
                "motif": "GATA1",
                "evidence_type": "motif_database",
                "matrix_id": "MA0035.4",
                "name": "GATA1",
                "url": "https://jaspar.elixir.no/matrix/MA0035.4/",
                "tf_match": True,
                "cell_type_match": False,
                "regulatory_keyword_match": False,
                "assay_keyword_match": False,
                "evidence_score": 3.0,
            },
        ]).to_csv(evidence_path, sep="\t", index=False)

        df, summary = prepare_literature_rag_context(
            str(evidence_path),
            max_records=5,
            min_score=4.0,
            max_context_chars=120,
            output_path=output_path,
        )

        assert len(df) == 1
        assert df.iloc[0]["source_id"] == "PMID:32293113"
        assert df.iloc[0]["citation"] == "PMID 32293113"
        assert "HepG2" in df.iloc[0]["context"]
        assert "matches TF/synonym" in df.iloc[0]["why_relevant"]
        assert "matches target cell type" in df.iloc[0]["why_relevant"]
        assert df.iloc[0]["evidence_type"] == "Reporter_assay"
        assert df.iloc[0]["direction"] in {"activation", "unknown"}
        assert df.iloc[0]["confidence"] in {"medium", "high"}
        assert summary["n_context_records"] == 1
        assert summary["output_path"] == str(output_path)
        assert output_path.exists()

    def test_empty_when_no_records_meet_threshold(self, tmp_path):
        evidence_path = tmp_path / "evidence.tsv"
        output_path = tmp_path / "empty_rag.tsv"
        pd.DataFrame([{
            "source": "PubMed",
            "motif": "SP1",
            "pmid": "1",
            "title": "Weak evidence",
            "evidence_score": 1.0,
        }]).to_csv(evidence_path, sep="\t", index=False)

        df, summary = prepare_literature_rag_context(
            str(evidence_path),
            min_score=4.0,
            output_path=output_path,
        )

        assert df.empty
        assert summary["pass"] is False
        assert summary["output_path"] == str(output_path)
        assert output_path.exists()
        assert list(pd.read_csv(output_path, sep="\t").columns) == [
            "source_id",
            "motif",
            "tf",
            "motif_id",
            "source",
            "evidence_type",
            "source_evidence_type",
            "title",
            "url",
            "citation",
            "claim",
            "direction",
            "cell_type",
            "assay",
            "encode_support",
            "evidence_score",
            "confidence",
            "context",
            "why_relevant",
            "contradicts",
        ]
        assert any("score threshold" in w for w in summary["warnings"])

    def test_rescores_unscored_evidence_and_limits_records(self, tmp_path):
        evidence_path = tmp_path / "unscored.tsv"
        long_abstract = (
            "Background text. " * 30
            + "NRF2 controls transcriptional regulation in HepG2 cells through enhancer activity. "
            + "Reporter assay validation supports the regulatory element. "
            + "Trailing text. " * 30
        )
        pd.DataFrame([
            {
                "source": "PubMed",
                "motif": "NRF2",
                "evidence_type": "literature",
                "pmid": "123",
                "title": "A broad NRF2 study",
                "abstract": long_abstract,
                "pubdate": "2024",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            },
            {
                "source": "ENCODE",
                "motif": "NRF2",
                "evidence_type": "functional_genomics",
                "accession": "ENCSR488EES",
                "target_label": "NFE2L2",
                "biosample_term_name": "HepG2",
                "assay_title": "TF ChIP-seq",
                "url": "https://www.encodeproject.org/experiments/ENCSR488EES/",
            },
        ]).to_csv(evidence_path, sep="\t", index=False)

        df, summary = prepare_literature_rag_context(
            str(evidence_path),
            max_records=1,
            min_score=4.0,
            max_context_chars=160,
        )

        assert len(df) == 1
        assert summary["n_input_records"] == 2
        assert summary["n_context_records"] == 1
        assert df.iloc[0]["source_id"] == "PMID:123"
        assert df.iloc[0]["evidence_score"] >= 6.0
        assert "NRF2 controls transcriptional regulation" in df.iloc[0]["context"]
        assert len(df.iloc[0]["context"]) <= 170
        assert df.iloc[0]["context"].startswith("... ")
        assert df.iloc[0]["context"].endswith(" ...")

    def test_builds_jaspar_and_encode_context_records(self, tmp_path):
        evidence_path = tmp_path / "database_evidence.tsv"
        pd.DataFrame([
            {
                "source": "JASPAR",
                "motif": "GATA1",
                "evidence_type": "motif_database",
                "matrix_id": "MA0035.4",
                "name": "GATA1",
                "collection": "CORE",
                "family": "GATA-type zinc fingers",
                "url": "https://jaspar.elixir.no/matrix/MA0035.4/",
                "tf_match": True,
                "cell_type_match": False,
                "regulatory_keyword_match": False,
                "assay_keyword_match": False,
                "evidence_score": 4.0,
            },
            {
                "source": "ENCODE",
                "motif": "HNF4A",
                "evidence_type": "functional_genomics",
                "accession": "ENCSR469FBY",
                "target_label": "HNF4A",
                "biosample_term_name": "HepG2",
                "assay_title": "TF ChIP-seq",
                "status": "released",
                "url": "https://www.encodeproject.org/experiments/ENCSR469FBY/",
                "tf_match": True,
                "cell_type_match": True,
                "regulatory_keyword_match": False,
                "assay_keyword_match": False,
                "evidence_score": 4.5,
            },
        ]).to_csv(evidence_path, sep="\t", index=False)

        df, summary = prepare_literature_rag_context(
            str(evidence_path),
            max_records=5,
            min_score=4.0,
        )

        assert list(df["source_id"]) == ["ENCODE:ENCSR469FBY", "JASPAR:MA0035.4"]
        assert list(df["citation"]) == ["ENCODE ENCSR469FBY", "JASPAR MA0035.4"]
        assert "target HNF4A" in df.iloc[0]["context"]
        assert "biosample HepG2" in df.iloc[0]["context"]
        assert "JASPAR motif profile MA0035.4" in df.iloc[1]["context"]
        assert summary["sources"] == {"ENCODE": 1, "JASPAR": 1}
        assert summary["motifs"] == ["GATA1", "HNF4A"]
