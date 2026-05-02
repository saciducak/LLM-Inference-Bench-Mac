"""
Ollama inference backend.
Uses the Ollama REST API for model management and inference.
"""

import json
import time
import requests
from typing import Optional
from datetime import datetime, timezone

from .base import InferenceBackend, InferenceResult


class OllamaBackend(InferenceBackend):
    """
    Backend for Ollama inference server.
    Communicates via REST API at localhost:11434.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(name="Ollama")
        self.base_url = base_url
        self._session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def get_available_models(self) -> list:
        """List models currently available in Ollama."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def pull_model(self, model_id: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            resp = self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_id, "stream": False},
                timeout=600,  # 10 min timeout for large downloads
            )
            return resp.status_code == 200
        except Exception:
            return False
    
    def load_model(self, model_config) -> float:
        """
        Load model into Ollama. If not available locally, attempts to pull.
        Returns load time in seconds.
        """
        model_id = model_config.ollama_id
        if not model_id:
            raise ValueError(f"No Ollama model ID for {model_config.name}")
        
        start = time.perf_counter()
        
        # Check if model exists locally
        available = self.get_available_models()
        base_name = model_id.split(":")[0] if ":" in model_id else model_id
        
        model_found = any(
            model_id in m or base_name in m 
            for m in available
        )
        
        if not model_found:
            # Try to pull the model
            if not self.pull_model(model_id):
                raise RuntimeError(f"Failed to pull Ollama model: {model_id}")
        
        # Warm up: send a minimal request to load model into memory
        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=120,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}")
        
        load_time = time.perf_counter() - start
        self._model_loaded = True
        self._current_model = model_id
        
        return load_time
    
    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7) -> InferenceResult:
        """
        Run inference via Ollama API with streaming to capture TTFT.
        """
        if not self._current_model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        result = InferenceResult(
            backend_name=self.name,
            model_name=self._current_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        payload = {
            "model": self._current_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        
        try:
            wall_start = time.perf_counter()
            first_token_time = None
            output_tokens = []
            final_data = None
            
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not line:
                    continue
                
                chunk = json.loads(line)
                
                if chunk.get("response"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_tokens.append(chunk["response"])
                
                if chunk.get("done"):
                    final_data = chunk
            
            wall_end = time.perf_counter()
            
            # Populate result from Ollama's built-in metrics
            result.output_text = "".join(output_tokens)
            result.total_generation_time = wall_end - wall_start
            
            if first_token_time is not None:
                result.time_to_first_token = first_token_time - wall_start
            
            if final_data:
                eval_count = final_data.get("eval_count", 0)
                eval_duration = final_data.get("eval_duration", 0)  # nanoseconds
                prompt_eval_duration = final_data.get("prompt_eval_duration", 0)
                load_duration = final_data.get("load_duration", 0)
                
                result.token_count = eval_count
                result.prompt_eval_time = prompt_eval_duration / 1e9 if prompt_eval_duration else 0
                result.model_load_time = load_duration / 1e9 if load_duration else 0
                
                if eval_duration > 0 and eval_count > 0:
                    result.tokens_per_second = (eval_count / eval_duration) * 1e9
                
                # Store raw metrics
                result.raw_metrics = {
                    "eval_count": eval_count,
                    "eval_duration_ns": eval_duration,
                    "prompt_eval_count": final_data.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ns": prompt_eval_duration,
                    "load_duration_ns": load_duration,
                    "total_duration_ns": final_data.get("total_duration", 0),
                }
        
        except requests.exceptions.Timeout:
            result.error = "Request timed out"
        except requests.exceptions.ConnectionError:
            result.error = "Connection to Ollama server failed"
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def cleanup(self):
        """Close session."""
        self._session.close()
        self._model_loaded = False
        self._current_model = None
