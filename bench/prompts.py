"""
Turkish prompt sets for benchmarking.
Each prompt includes category, text, expected keywords, and token bounds.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt with evaluation metadata."""
    category: str
    text: str
    expected_keywords: List[str] = field(default_factory=list)
    min_tokens: int = 30
    max_tokens: int = 300


# ─── Prompt Library ─────────────────────────────────────────────────────────

PROMPTS_BILGI = [
    BenchmarkPrompt(
        category="bilgi",
        text="Yapay zekanın tarihçesini kısaca özetle. Önemli dönüm noktalarını ve kişileri belirt.",
        expected_keywords=["turing", "yapay", "zeka", "öğrenme", "sinir", "ağ"],
        min_tokens=50, max_tokens=300,
    ),
    BenchmarkPrompt(
        category="bilgi",
        text="Kuantum bilgisayarların klasik bilgisayarlardan farkı nedir? Basit bir dille açıkla.",
        expected_keywords=["kubit", "süperpozisyon", "kuantum", "klasik", "hesaplama"],
        min_tokens=40, max_tokens=250,
    ),
    BenchmarkPrompt(
        category="bilgi",
        text="Fotosentez nedir ve bitkiler için neden önemlidir? Kısaca açıkla.",
        expected_keywords=["güneş", "ışık", "klorofil", "oksijen", "enerji"],
        min_tokens=30, max_tokens=200,
    ),
]

PROMPTS_ANALIZ = [
    BenchmarkPrompt(
        category="analiz",
        text="Apple Silicon çiplerinin x86 mimarisine kıyasla avantajlarını ve dezavantajlarını değerlendir.",
        expected_keywords=["arm", "enerji", "performans", "unified", "bellek", "uyumluluk"],
        min_tokens=50, max_tokens=350,
    ),
    BenchmarkPrompt(
        category="analiz",
        text="Uzaktan çalışma modelinin avantajlarını ve dezavantajlarını analiz et.",
        expected_keywords=["verimlilik", "iletişim", "esneklik", "izolasyon", "takım"],
        min_tokens=50, max_tokens=300,
    ),
]

PROMPTS_YARATICI = [
    BenchmarkPrompt(
        category="yaratici",
        text="İstanbul'un Boğaz'ı hakkında 4 kıtalık bir şiir yaz.",
        expected_keywords=["boğaz", "istanbul", "su", "köprü"],
        min_tokens=30, max_tokens=200,
    ),
    BenchmarkPrompt(
        category="yaratici",
        text="Bir yapay zeka asistanının günlüğünden bir sayfa yaz. Birinci tekil şahıs kullan.",
        expected_keywords=["ben", "bugün", "kullanıcı", "öğren"],
        min_tokens=50, max_tokens=300,
    ),
]

PROMPTS_KOD = [
    BenchmarkPrompt(
        category="kod",
        text="Python'da binary search algoritmasını yaz ve nasıl çalıştığını açıkla.",
        expected_keywords=["def", "mid", "return", "left", "right"],
        min_tokens=30, max_tokens=300,
    ),
    BenchmarkPrompt(
        category="kod",
        text="Python'da bir linked list sınıfı yaz. insert ve display metodlarını ekle.",
        expected_keywords=["class", "Node", "self", "next", "def"],
        min_tokens=30, max_tokens=350,
    ),
]

PROMPTS_CEVIRI = [
    BenchmarkPrompt(
        category="ceviri",
        text="Translate the following text to Turkish: 'Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.'",
        expected_keywords=["makine", "öğrenme", "yapay", "zeka", "veri"],
        min_tokens=20, max_tokens=150,
    ),
]

# ─── Prompt Sets ────────────────────────────────────────────────────────────

ALL_PROMPTS: List[BenchmarkPrompt] = (
    PROMPTS_BILGI + PROMPTS_ANALIZ + PROMPTS_YARATICI +
    PROMPTS_KOD + PROMPTS_CEVIRI
)

QUICK_PROMPTS: List[BenchmarkPrompt] = [
    PROMPTS_BILGI[0],     # 1 bilgi
]

STANDARD_PROMPTS: List[BenchmarkPrompt] = [
    PROMPTS_BILGI[0],
    PROMPTS_KOD[0],
    PROMPTS_YARATICI[0],
]


def get_prompts(mode: str = "standard") -> List[BenchmarkPrompt]:
    """Get prompt set based on benchmark mode."""
    if mode == "quick":
        return QUICK_PROMPTS
    elif mode == "full":
        return ALL_PROMPTS
    else:
        return STANDARD_PROMPTS
