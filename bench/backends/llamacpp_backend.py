"""
llama.cpp inference backend.
Uses llama-cpp-python bindings for GGUF model inference.
"""

import time
import os
from datetime import datetime, timezone
from .base import InferenceBackend, InferenceResult


class LlamaCppBackend(InferenceBackend):
    def __init__(self, n_gpu_layers=-1, n_ctx=2048):
        super().__init__(name="llama.cpp")
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None
        self._llama_module = None

    def is_available(self):
        try:
            import llama_cpp
            self._llama_module = llama_cpp
            return True
        except ImportError:
            return False

    def load_model(self, model_config):
        if not self._llama_module and not self.is_available():
            raise RuntimeError("llama-cpp-python is not installed")
        gguf_path = model_config.gguf_path
        if not gguf_path:
            raise ValueError(f"No GGUF path for {model_config.name}")
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF not found: {gguf_path}")
        if self._llm:
            del self._llm
        start = time.perf_counter()
        self._llm = self._llama_module.Llama(
            model_path=gguf_path, n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx, verbose=False)
        load_time = time.perf_counter() - start
        self._model_loaded = True
        self._current_model = model_config.name
        return load_time

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        if not self._llm:
            raise RuntimeError("No model loaded")
        result = InferenceResult(
            backend_name=self.name, model_name=self._current_model or "",
            timestamp=datetime.now(timezone.utc).isoformat())
        try:
            wall_start = time.perf_counter()
            first_token_time = None
            output_tokens = []
            token_count = 0
            for chunk in self._llm(prompt, max_tokens=max_tokens,
                                    temperature=temperature, stream=True, echo=False):
                t = chunk["choices"][0]["text"]
                if t:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_tokens.append(t)
                    token_count += 1
            wall_end = time.perf_counter()
            result.output_text = "".join(output_tokens)
            result.token_count = token_count
            result.total_generation_time = wall_end - wall_start
            if first_token_time:
                result.time_to_first_token = first_token_time - wall_start
            if first_token_time and token_count > 1:
                decode_time = wall_end - first_token_time
                if decode_time > 0:
                    result.tokens_per_second = (token_count - 1) / decode_time
            elif token_count > 0 and result.total_generation_time > 0:
                result.tokens_per_second = token_count / result.total_generation_time
            result.prompt_eval_time = result.time_to_first_token
        except Exception as e:
            result.error = str(e)
        return result

    def cleanup(self):
        if self._llm:
            del self._llm
            self._llm = None
        self._model_loaded = False
        self._current_model = None
