"""
Output quality evaluation for Turkish LLM outputs.
Provides composite scoring across multiple dimensions.
"""

import re
from typing import Dict, List


# Turkish-specific characters
TURKISH_CHARS = set("ğüşıöçĞÜŞİÖÇ")


def evaluate_quality(output: str, prompt: str, category: str,
                     expected_keywords: List[str] = None,
                     min_tokens: int = 20, max_tokens: int = 500) -> Dict[str, float]:
    """
    Evaluate the quality of LLM output.
    Returns a dict with individual scores and composite score (0-100).
    """
    if not output or not output.strip():
        return {"composite": 0, "length": 0, "keywords": 0,
                "coherence": 0, "turkish": 0, "relevance": 0}

    scores = {}

    # 1. Length Score (0-100): Is the output a reasonable length?
    word_count = len(output.split())
    if word_count < min_tokens // 2:
        scores["length"] = max(0, (word_count / (min_tokens // 2)) * 50)
    elif min_tokens <= word_count <= max_tokens:
        scores["length"] = 100
    elif word_count > max_tokens:
        overshoot = (word_count - max_tokens) / max_tokens
        scores["length"] = max(50, 100 - overshoot * 50)
    else:
        ratio = word_count / min_tokens
        scores["length"] = min(100, ratio * 100)

    # 2. Keyword Match (0-100): Does output contain expected keywords?
    if expected_keywords:
        output_lower = output.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
        scores["keywords"] = (matches / len(expected_keywords)) * 100
    else:
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        output_lower = output.lower()
        if prompt_words:
            matches = sum(1 for w in prompt_words if w in output_lower)
            scores["keywords"] = min(100, (matches / max(1, len(prompt_words))) * 100)
        else:
            scores["keywords"] = 50

    # 3. Coherence (0-100): Proper sentence structure
    sentences = re.split(r'[.!?।]+', output)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        proper = sum(1 for s in sentences
                     if len(s.split()) >= 3 and s[0].isupper() or s[0] in TURKISH_CHARS)
        scores["coherence"] = min(100, (proper / len(sentences)) * 100)
    else:
        scores["coherence"] = 10

    # 4. Turkish Character Ratio (0-100)
    if category in ["bilgi", "analiz", "yaratici"]:
        total_chars = len(output)
        if total_chars > 0:
            turkish_count = sum(1 for c in output if c in TURKISH_CHARS)
            ratio = turkish_count / total_chars
            scores["turkish"] = min(100, ratio * 1000)
        else:
            scores["turkish"] = 0
    else:
        scores["turkish"] = 50

    # 5. Relevance (0-100): No repetition/garbage (unigram)
    words = output.split()
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        scores["relevance"] = min(100, unique_ratio * 120)
    else:
        scores["relevance"] = 50

    # 6. N-gram Repetition Penalty (0-100): Detect degenerate loops
    scores["repetition"] = _ngram_repetition_score(output)

    # Composite: weighted average
    weights = {"length": 0.10, "keywords": 0.25, "coherence": 0.20,
               "turkish": 0.15, "relevance": 0.15, "repetition": 0.15}
    composite = sum(scores[k] * weights[k] for k in weights)
    scores["composite"] = min(100, composite)

    return scores


def _ngram_repetition_score(text: str, n: int = 3) -> float:
    """
    Score text based on n-gram diversity (0-100).
    High repetition of n-grams indicates degenerate output (looping).
    Returns 100 for diverse text, 0 for highly repetitive text.
    """
    words = text.lower().split()
    if len(words) < n + 1:
        return 50.0  # Too short to evaluate

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 50.0

    unique_ratio = len(set(ngrams)) / len(ngrams)
    # Scale: 1.0 unique_ratio = 100, 0.3 = ~0
    return min(100.0, max(0.0, (unique_ratio - 0.3) / 0.7 * 100))
