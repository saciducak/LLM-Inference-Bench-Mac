"""Tests for the quality evaluation module."""

import pytest
from bench.quality import evaluate_quality, TURKISH_CHARS


class TestEvaluateQualityBasics:
    """Test fundamental quality scoring behavior."""

    def test_empty_output_returns_zero(self):
        result = evaluate_quality("", "Test prompt", "bilgi")
        assert result["composite"] == 0
        assert all(v == 0 for k, v in result.items())

    def test_whitespace_only_returns_zero(self):
        result = evaluate_quality("   \n\t  ", "Test prompt", "bilgi")
        assert result["composite"] == 0

    def test_none_output_returns_zero(self):
        result = evaluate_quality(None, "Test prompt", "bilgi")
        assert result["composite"] == 0

    def test_scores_within_bounds(self, turkish_text_good):
        result = evaluate_quality(
            turkish_text_good, "Fotosentez nedir?", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen"])
        for key, value in result.items():
            assert 0 <= value <= 100, f"{key} score {value} out of bounds"


class TestLengthScoring:
    """Test the length dimension of quality scoring."""

    def test_optimal_length_scores_high(self):
        text = " ".join(["kelime"] * 50)  # 50 words
        result = evaluate_quality(text, "prompt", "bilgi",
                                  min_tokens=30, max_tokens=300)
        assert result["length"] >= 80

    def test_very_short_text_penalized(self):
        result = evaluate_quality("Kısa.", "Uzun bir açıklama yap", "bilgi",
                                  min_tokens=50, max_tokens=300)
        assert result["length"] < 30

    def test_very_long_text_partially_penalized(self):
        text = " ".join(["kelime"] * 600)  # Way over max
        result = evaluate_quality(text, "prompt", "bilgi",
                                  min_tokens=30, max_tokens=100)
        assert result["length"] < 100
        assert result["length"] >= 50  # Not zero, just penalized


class TestKeywordScoring:
    """Test keyword matching behavior."""

    def test_all_keywords_match(self):
        text = "Güneş ışığı klorofil ile oksijen üretir enerji sağlar"
        result = evaluate_quality(
            text, "Fotosentez nedir?", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen", "enerji"])
        assert result["keywords"] == 100.0

    def test_no_keywords_match(self):
        text = "Bu tamamen alakasız bir cevaptır bilgisayar programlama"
        result = evaluate_quality(
            text, "Fotosentez nedir?", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen"])
        assert result["keywords"] == 0.0

    def test_partial_keywords(self):
        text = "Güneş ışığı önemlidir ama oksijen hakkında bilgi yok"
        result = evaluate_quality(
            text, "prompt", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen"])
        # 2 out of 3 = ~66.7%
        assert 60 <= result["keywords"] <= 70

    def test_fallback_without_expected_keywords(self):
        text = "Fotosentez bitkilerin enerji üretme sürecidir"
        result = evaluate_quality(text, "Fotosentez nedir?", "bilgi")
        assert "keywords" in result
        assert result["keywords"] >= 0


class TestCoherenceScoring:
    """Test sentence structure evaluation."""

    def test_well_formed_sentences(self):
        text = ("Bu ilk cümledir. İkinci cümle de gayet düzgün. "
                "Üçüncü cümle ile devam edelim.")
        result = evaluate_quality(text, "prompt", "bilgi")
        assert result["coherence"] >= 50

    def test_fragmented_text_penalized(self):
        text = "a b c"
        result = evaluate_quality(text, "prompt", "bilgi")
        assert result["coherence"] <= 50


class TestTurkishCharScoring:
    """Test Turkish character density scoring."""

    def test_rich_turkish_text_scores_high(self, turkish_text_good):
        result = evaluate_quality(turkish_text_good, "prompt", "bilgi")
        assert result["turkish"] >= 40

    def test_english_text_scores_low(self):
        text = "This is a completely English response about photosynthesis."
        result = evaluate_quality(text, "prompt", "bilgi")
        assert result["turkish"] <= 10

    def test_non_turkish_category_gets_default(self):
        text = "def binary_search(arr, target): return -1"
        result = evaluate_quality(text, "prompt", "kod")
        assert result["turkish"] == 50  # Default for non-turkish categories


class TestRelevanceScoring:
    """Test repetition detection."""

    def test_diverse_vocabulary_scores_high(self, turkish_text_good):
        result = evaluate_quality(turkish_text_good, "prompt", "bilgi")
        assert result["relevance"] >= 70

    def test_repetitive_text_penalized(self, turkish_text_poor):
        result = evaluate_quality(turkish_text_poor, "prompt", "bilgi")
        assert result["relevance"] < 80

    def test_very_short_text_gets_default(self):
        result = evaluate_quality("Evet.", "prompt", "bilgi")
        assert result["relevance"] == 50  # Default for < 5 words


class TestCompositeScoring:
    """Test the weighted composite calculation."""

    def test_composite_is_weighted_average(self):
        text = " ".join(["kelime"] * 50)
        result = evaluate_quality(text, "prompt", "bilgi")
        weights = {"length": 0.10, "keywords": 0.25, "coherence": 0.20,
                   "turkish": 0.15, "relevance": 0.15, "repetition": 0.15}
        expected = sum(result[k] * weights[k] for k in weights)
        assert abs(result["composite"] - min(100, expected)) < 0.1

    def test_good_text_beats_poor_text(self, turkish_text_good, turkish_text_poor):
        good = evaluate_quality(
            turkish_text_good, "Fotosentez nedir?", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen"])
        poor = evaluate_quality(
            turkish_text_poor, "Fotosentez nedir?", "bilgi",
            expected_keywords=["güneş", "klorofil", "oksijen"])
        assert good["composite"] > poor["composite"]


class TestHallucinationWeakness:
    """Document the known limitation: keyword matching can't detect wrong facts."""

    def test_hallucination_still_gets_keyword_score(self, turkish_text_hallucination):
        """This test documents the known weakness of rule-based evaluation.
        A factually wrong response that contains expected keywords will still
        score reasonably well on the keyword dimension. This is why we need
        LLM-as-a-Judge (quality_judge.py) for semantic evaluation."""
        result = evaluate_quality(
            turkish_text_hallucination, "Fotosentez nedir?", "bilgi",
            expected_keywords=["klorofil", "güneş", "oksijen"])
        # All keywords present despite factual errors
        assert result["keywords"] == 100.0
        # Document this as a known limitation, not a bug
