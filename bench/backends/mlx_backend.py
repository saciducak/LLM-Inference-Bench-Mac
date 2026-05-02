"""
MLX inference backend.
Uses mlx-lm for native Apple Silicon inference.
"""

import time
from datetime import datetime, timezone
from .base import InferenceBackend, InferenceResult


class MLXBackend(InferenceBackend):
    def __init__(self):
        super().__init__(name="MLX")
        self._model = None
        self._tokenizer = None
        self._mlx_lm = None

    def is_available(self):
        try:
            import mlx_lm
            self._mlx_lm = mlx_lm
            return True
        except ImportError:
            return False

    def load_model(self, model_config):
        if not self._mlx_lm and not self.is_available():
            raise RuntimeError("mlx-lm is not installed")
        mlx_id = model_config.mlx_id
        if not mlx_id:
            raise ValueError(f"No MLX model ID for {model_config.name}")
        if self._model:
            del self._model
            del self._tokenizer
        start = time.perf_counter()
        self._model, self._tokenizer = self._mlx_lm.load(mlx_id)
        load_time = time.perf_counter() - start
        self._model_loaded = True
        self._current_model = model_config.name
        return load_time

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        if not self._model:
            raise RuntimeError("No model loaded")
        result = InferenceResult(
            backend_name=self.name, model_name=self._current_model or "",
            timestamp=datetime.now(timezone.utc).isoformat())
        try:
            wall_start = time.perf_counter()
            first_token_time = None
            output_tokens = []
            token_count = 0

            # Use stream_generate for token-by-token timing
            if hasattr(self._mlx_lm, 'stream_generate'):
                for token_text in self._mlx_lm.stream_generate(
                    self._model, self._tokenizer, prompt=prompt,
                    max_tokens=max_tokens, temp=temperature):
                    text = token_text if isinstance(token_text, str) else str(token_text)
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens.append(text)
                        token_count += 1
            else:
                # Fallback to batch generate
                response = self._mlx_lm.generate(
                    self._model, self._tokenizer, prompt=prompt,
                    max_tokens=max_tokens, temp=temperature)
                first_token_time = time.perf_counter()
                output_tokens.append(response)
                # Estimate token count
                token_count = len(self._tokenizer.encode(response))

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
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        self._model_loaded = False
        self._current_model = None
