"""Shared test fixtures for LLM Inference Bench."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.backends.base import InferenceResult
from bench.config import ModelConfig, BenchmarkConfig


@pytest.fixture
def sample_inference_result():
    """A realistic InferenceResult for testing."""
    return InferenceResult(
        output_text="Yapay zekanın tarihçesi 1950'lerde Turing ile başlar. "
                     "Sinir ağları ve derin öğrenme bu alanın temel taşlarıdır.",
        token_count=45,
        time_to_first_token=0.085,
        total_generation_time=1.2,
        model_load_time=2.5,
        prompt_eval_time=0.085,
        tokens_per_second=37.5,
        peak_ram_mb=1200.0,
        baseline_ram_mb=250.0,
        ram_delta_mb=950.0,
        quality_score=72.0,
        quality_details={"length": 80, "keywords": 66, "coherence": 72,
                         "turkish": 70, "relevance": 78, "composite": 72},
        backend_name="Ollama",
        model_name="test-model:latest",
        quantization="Q4_0",
        prompt_category="bilgi",
        timestamp="2026-05-19T12:00:00Z",
    )


@pytest.fixture
def sample_model_config():
    """A realistic ModelConfig for testing."""
    return ModelConfig(
        name="Test Model Q4",
        ollama_id="test-model:latest",
        quantization="Q4_0",
        runtime="ollama",
        size_label="1.5B",
    )


@pytest.fixture
def sample_mlx_config():
    return ModelConfig(
        name="Test MLX 4bit",
        mlx_id="test-community/model-4bit",
        quantization="4-bit",
        runtime="mlx",
        size_label="1.5B",
    )


@pytest.fixture
def sample_llamacpp_config():
    return ModelConfig(
        name="Test GGUF Q4",
        gguf_path="models/test-q4.gguf",
        quantization="Q4_K_M",
        runtime="llama.cpp",
        size_label="1.5B",
    )


@pytest.fixture
def empty_inference_result():
    """An empty/failed InferenceResult."""
    return InferenceResult(
        output_text="",
        token_count=0,
        backend_name="Ollama",
        model_name="test-model",
        error="Connection refused",
    )


@pytest.fixture
def benchmark_config():
    return BenchmarkConfig()


@pytest.fixture
def turkish_text_good():
    """High-quality Turkish text for quality evaluation."""
    return (
        "Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve suyu "
        "glikoza dönüştürdüğü biyokimyasal bir süreçtir. Bu süreçte klorofil "
        "pigmenti ışık enerjisini absorbe eder. Üretilen oksijen atmosfere "
        "salınır ve canlıların yaşamı için kritik öneme sahiptir."
    )


@pytest.fixture
def turkish_text_poor():
    """Low-quality output: repetitive, short, no Turkish chars."""
    return "the the the plant plant plant good good"


@pytest.fixture
def turkish_text_hallucination():
    """Factually wrong but keyword-matching text."""
    return (
        "Fotosentez klorofil ile yapılmaz. Güneş ışığı bitkilere zarar verir "
        "ve oksijen aslında zararlı bir gazdır."
    )
