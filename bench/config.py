"""
Benchmark configuration module.
Defines all configurable parameters for benchmark runs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class BenchmarkMode(Enum):
    """Benchmark execution modes."""
    QUICK = "quick"      # 1 prompt, 1 iteration — fast sanity check
    STANDARD = "standard"  # 3 prompts, 2 iterations — balanced
    FULL = "full"        # All prompts, 3 iterations — comprehensive


class Runtime(Enum):
    """Supported inference runtimes."""
    OLLAMA = "ollama"
    LLAMACPP = "llama.cpp"
    MLX = "mlx"


@dataclass
class ModelConfig:
    """Configuration for a single model to benchmark."""
    name: str                          # Human-readable name
    ollama_id: Optional[str] = None    # Ollama model identifier
    gguf_path: Optional[str] = None    # Path to GGUF file for llama.cpp
    mlx_id: Optional[str] = None       # HuggingFace MLX model ID
    quantization: str = "default"      # Quantization level label
    runtime: str = "ollama"            # Which runtime to use
    size_label: str = ""               # e.g., "3B", "7B", "1.5B"


@dataclass
class BenchmarkConfig:
    """Master benchmark configuration."""
    
    # Execution parameters
    mode: BenchmarkMode = BenchmarkMode.STANDARD
    max_tokens: int = 256
    temperature: float = 0.7
    timeout_seconds: int = 120
    warmup_runs: int = 1
    
    # Iteration counts per mode
    iterations: Dict[str, int] = field(default_factory=lambda: {
        "quick": 1,
        "standard": 2,
        "full": 3,
    })
    
    # Memory sampling interval (seconds)
    memory_sample_interval: float = 0.1
    
    # Output configuration
    results_dir: str = "results"
    verbose: bool = False
    
    @property
    def num_iterations(self) -> int:
        return self.iterations.get(self.mode.value, 2)


# ─── Default Model Configurations ───────────────────────────────────────────

DEFAULT_OLLAMA_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="Gemma 2B (Q4_0)",
        ollama_id="gemma2:2b",
        quantization="Q4_0",
        runtime="ollama",
        size_label="2B",
    ),
    ModelConfig(
        name="Phi-3 Mini (Q4_0)",
        ollama_id="phi3:mini",
        quantization="Q4_0",
        runtime="ollama",
        size_label="3.8B",
    ),
    ModelConfig(
        name="Qwen2.5 1.5B (Q4_0)",
        ollama_id="qwen2.5:1.5b",
        quantization="Q4_0",
        runtime="ollama",
        size_label="1.5B",
    ),
    ModelConfig(
        name="Llama 3.2 3B (Q4_0)",
        ollama_id="llama3.2:3b",
        quantization="Q4_0",
        runtime="ollama",
        size_label="3B",
    ),
    ModelConfig(
        name="Qwen2.5 1.5B (Q8_0)",
        ollama_id="qwen2.5:1.5b-instruct-q8_0",
        quantization="Q8_0",
        runtime="ollama",
        size_label="1.5B",
    ),
]

DEFAULT_LLAMACPP_MODELS: List[ModelConfig] = [
    # Users need to download GGUF files and set paths
    ModelConfig(
        name="Qwen2.5 1.5B (Q2_K) [GGUF]",
        gguf_path="models/qwen2.5-1.5b-instruct-q2_k.gguf",
        quantization="Q2_K",
        runtime="llama.cpp",
        size_label="1.5B",
    ),
    ModelConfig(
        name="Qwen2.5 1.5B (Q4_K_M) [GGUF]",
        gguf_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        quantization="Q4_K_M",
        runtime="llama.cpp",
        size_label="1.5B",
    ),
    ModelConfig(
        name="Qwen2.5 1.5B (Q8_0) [GGUF]",
        gguf_path="models/qwen2.5-1.5b-instruct-q8_0.gguf",
        quantization="Q8_0",
        runtime="llama.cpp",
        size_label="1.5B",
    ),
]

DEFAULT_MLX_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="Qwen2.5 1.5B (4-bit) [MLX]",
        mlx_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization="4-bit",
        runtime="mlx",
        size_label="1.5B",
    ),
    ModelConfig(
        name="Qwen2.5 1.5B (8-bit) [MLX]",
        mlx_id="mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        quantization="8-bit",
        runtime="mlx",
        size_label="1.5B",
    ),
    ModelConfig(
        name="Gemma 2B (4-bit) [MLX]",
        mlx_id="mlx-community/gemma-2-2b-it-4bit",
        quantization="4-bit",
        runtime="mlx",
        size_label="2B",
    ),
]


def get_all_models() -> List[ModelConfig]:
    """Return all default model configurations."""
    return DEFAULT_OLLAMA_MODELS + DEFAULT_LLAMACPP_MODELS + DEFAULT_MLX_MODELS


def get_models_by_runtime(runtime: str) -> List[ModelConfig]:
    """Return model configurations filtered by runtime."""
    runtime_map = {
        "ollama": DEFAULT_OLLAMA_MODELS,
        "llama.cpp": DEFAULT_LLAMACPP_MODELS,
        "mlx": DEFAULT_MLX_MODELS,
    }
    return runtime_map.get(runtime, [])
