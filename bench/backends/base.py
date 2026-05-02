"""
Abstract base class for inference backends.
All runtime backends (Ollama, llama.cpp, MLX) implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time


@dataclass
class InferenceResult:
    """
    Result of a single inference run.
    Contains all timing, throughput, and output metrics.
    """
    # Output
    output_text: str = ""
    token_count: int = 0
    
    # Timing (seconds)
    time_to_first_token: float = 0.0     # TTFT
    total_generation_time: float = 0.0    # Total wall-clock time
    model_load_time: float = 0.0         # Time to load model into memory
    prompt_eval_time: float = 0.0        # Prompt processing time
    
    # Throughput
    tokens_per_second: float = 0.0       # Decode throughput
    
    # Memory (MB)
    peak_ram_mb: float = 0.0             # Peak RSS during inference
    baseline_ram_mb: float = 0.0         # RAM before inference
    ram_delta_mb: float = 0.0            # Difference
    
    # Quality (set by quality evaluator)
    quality_score: float = 0.0           # Composite quality score (0-100)
    quality_details: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    backend_name: str = ""
    model_name: str = ""
    quantization: str = ""
    prompt_category: str = ""
    error: Optional[str] = None
    timestamp: str = ""
    
    # Raw backend-specific data
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "output_text": self.output_text,
            "token_count": self.token_count,
            "time_to_first_token_ms": round(self.time_to_first_token * 1000, 2),
            "total_generation_time_s": round(self.total_generation_time, 3),
            "model_load_time_s": round(self.model_load_time, 3),
            "prompt_eval_time_s": round(self.prompt_eval_time, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "baseline_ram_mb": round(self.baseline_ram_mb, 1),
            "ram_delta_mb": round(self.ram_delta_mb, 1),
            "quality_score": round(self.quality_score, 1),
            "quality_details": {k: round(v, 2) for k, v in self.quality_details.items()},
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "quantization": self.quantization,
            "prompt_category": self.prompt_category,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    Each backend must implement:
    - is_available(): Check if the runtime is installed/running
    - load_model(): Load a model into memory
    - generate(): Run inference and return InferenceResult
    - cleanup(): Release resources
    """
    
    def __init__(self, name: str):
        self.name = name
        self._model_loaded = False
        self._current_model = None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system."""
        pass
    
    @abstractmethod
    def load_model(self, model_config) -> float:
        """
        Load a model. Returns load time in seconds.
        
        Args:
            model_config: ModelConfig instance
            
        Returns:
            Load time in seconds
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, 
                 temperature: float = 0.7) -> InferenceResult:
        """
        Run inference on the loaded model.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            InferenceResult with all metrics
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release resources and unload model."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Return current backend status."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "model_loaded": self._model_loaded,
            "current_model": self._current_model,
        }
