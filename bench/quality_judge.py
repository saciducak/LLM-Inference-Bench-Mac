"""
LLM-as-a-Judge quality evaluation module.
Uses a local LLM (via Ollama) as a semantic judge for output quality.
This supplements the rule-based evaluator in quality.py with actual
semantic understanding — detecting hallucinations, factual errors,
and coherence issues that regex cannot catch.

Usage:
    judge = QualityJudge()
    if judge.is_available():
        score = judge.evaluate(prompt, response, category)
"""

import json
import requests
import time
from typing import Dict, Optional


# Turkish rubric for the judge model
JUDGE_RUBRIC_TR = """Sen bir Türkçe dil kalitesi değerlendirmecisisin.
Aşağıdaki yanıtı verilen soruya göre değerlendir.

Puanlama kriterleri (her biri 0-100):
1. factual_accuracy: Yanıt olgusal olarak doğru mu? Halüsinasyon var mı?
2. relevance: Yanıt soruyla ne kadar alakalı?
3. coherence: Yanıt tutarlı ve mantıklı mı?
4. language_quality: Türkçe dil bilgisi ve akıcılık kalitesi

Sadece JSON formatında yanıt ver, başka bir şey yazma:
{"factual_accuracy": X, "relevance": X, "coherence": X, "language_quality": X, "overall": X, "reasoning": "..."}
"""


class QualityJudge:
    """
    Semantic quality evaluation using a local LLM as judge.
    Designed to work with Ollama but can be extended to other backends.
    """

    def __init__(self, model: str = "qwen2.5:1.5b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = None

    def is_available(self) -> bool:
        """Check if the judge model is accessible."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self._available = any(self.model in m for m in models)
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def evaluate(self, prompt: str, response: str,
                 category: str = "bilgi") -> Optional[Dict[str, float]]:
        """
        Evaluate response quality using the judge model.

        Returns dict with scores (0-100) for each dimension,
        or None if evaluation fails.
        """
        if not self.is_available():
            return None

        judge_prompt = f"""{JUDGE_RUBRIC_TR}

Soru: {prompt}
Kategori: {category}
Yanıt: {response}

JSON değerlendirme:"""

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": judge_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 256,
                        "temperature": 0.0,  # Deterministic judging
                    },
                },
                timeout=60,
            )

            if resp.status_code != 200:
                return None

            output = resp.json().get("response", "")

            # Try to extract JSON from response
            scores = self._parse_judge_output(output)
            return scores

        except Exception:
            return None

    def _parse_judge_output(self, output: str) -> Optional[Dict[str, float]]:
        """Parse JSON scores from judge output."""
        try:
            # Try direct JSON parse
            data = json.loads(output.strip())
            return self._validate_scores(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in output
        import re
        json_match = re.search(r'\{[^{}]+\}', output)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_scores(data)
            except json.JSONDecodeError:
                pass

        return None

    def _validate_scores(self, data: dict) -> Dict[str, float]:
        """Ensure all scores are valid numbers in 0-100 range."""
        dimensions = ["factual_accuracy", "relevance", "coherence",
                       "language_quality", "overall"]
        result = {}
        for dim in dimensions:
            val = data.get(dim, 50)
            try:
                val = float(val)
                result[dim] = max(0, min(100, val))
            except (TypeError, ValueError):
                result[dim] = 50.0

        result["reasoning"] = str(data.get("reasoning", ""))
        return result
