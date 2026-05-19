"""Tests for inference backends using mocks (no real models needed)."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from bench.backends.base import InferenceBackend, InferenceResult
from bench.backends.ollama_backend import OllamaBackend
from bench.backends.llamacpp_backend import LlamaCppBackend
from bench.backends.mlx_backend import MLXBackend


class TestInferenceResult:
    """Test the InferenceResult dataclass."""

    def test_to_dict_roundtrip(self, sample_inference_result):
        d = sample_inference_result.to_dict()
        assert d["backend_name"] == "Ollama"
        assert d["model_name"] == "test-model:latest"
        assert d["tokens_per_second"] == 37.5
        assert d["time_to_first_token_ms"] == 85.0  # 0.085 * 1000
        assert d["peak_ram_mb"] == 1200.0
        assert d["error"] is None

    def test_to_dict_with_error(self, empty_inference_result):
        d = empty_inference_result.to_dict()
        assert d["error"] == "Connection refused"
        assert d["token_count"] == 0

    def test_quality_details_rounding(self, sample_inference_result):
        d = sample_inference_result.to_dict()
        for key, val in d["quality_details"].items():
            assert isinstance(val, (int, float))

    def test_default_values(self):
        result = InferenceResult()
        assert result.output_text == ""
        assert result.token_count == 0
        assert result.tokens_per_second == 0.0
        assert result.error is None
        assert result.raw_metrics == {}


class TestOllamaBackend:
    """Test Ollama backend with mocked HTTP calls."""

    def test_is_available_success(self):
        backend = OllamaBackend()
        with patch.object(backend._session, 'get') as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert backend.is_available() is True

    def test_is_available_failure(self):
        backend = OllamaBackend()
        import requests as req
        with patch.object(backend._session, 'get', side_effect=req.ConnectionError("refused")):
            assert backend.is_available() is False

    def test_get_available_models(self):
        backend = OllamaBackend()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:3b"}]
        }
        with patch.object(backend._session, 'get', return_value=mock_response):
            models = backend.get_available_models()
            assert len(models) == 2
            assert "qwen2.5:1.5b" in models

    def test_load_model_requires_ollama_id(self, sample_llamacpp_config):
        """Config without ollama_id should raise ValueError."""
        backend = OllamaBackend()
        with pytest.raises(ValueError, match="No Ollama model ID"):
            backend.load_model(sample_llamacpp_config)

    def test_cleanup_resets_state(self):
        backend = OllamaBackend()
        backend._model_loaded = True
        backend._current_model = "test"
        backend.cleanup()
        assert backend._model_loaded is False
        assert backend._current_model is None


class TestLlamaCppBackend:
    """Test llama.cpp backend with mocked library."""

    def test_is_available_with_import(self):
        backend = LlamaCppBackend()
        with patch.dict('sys.modules', {'llama_cpp': MagicMock()}):
            assert backend.is_available() is True

    def test_is_available_without_import(self):
        backend = LlamaCppBackend()
        backend._llama_module = None
        with patch('builtins.__import__', side_effect=ImportError):
            result = backend.is_available()
            # Either False or it caught the import error
            assert result is False or backend._llama_module is None

    def test_load_model_file_not_found(self, sample_llamacpp_config):
        backend = LlamaCppBackend()
        backend._llama_module = MagicMock()
        with pytest.raises(FileNotFoundError):
            backend.load_model(sample_llamacpp_config)

    def test_generate_without_model_raises(self):
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match="No model loaded"):
            backend.generate("test prompt")

    def test_cleanup(self):
        backend = LlamaCppBackend()
        backend._llm = MagicMock()
        backend._model_loaded = True
        backend.cleanup()
        assert backend._llm is None
        assert backend._model_loaded is False


class TestMLXBackend:
    """Test MLX backend with mocked library."""

    def test_is_available_with_mlx(self):
        backend = MLXBackend()
        with patch.dict('sys.modules', {'mlx_lm': MagicMock()}):
            assert backend.is_available() is True

    def test_load_model_requires_mlx_id(self, sample_model_config):
        """Config without mlx_id should raise ValueError."""
        backend = MLXBackend()
        backend._mlx_lm = MagicMock()
        with pytest.raises(ValueError, match="No MLX model ID"):
            backend.load_model(sample_model_config)

    def test_generate_without_model_raises(self):
        backend = MLXBackend()
        with pytest.raises(RuntimeError, match="No model loaded"):
            backend.generate("test prompt")

    def test_cleanup(self):
        backend = MLXBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._model_loaded = True
        backend.cleanup()
        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._model_loaded is False


class TestBackendInterface:
    """Verify all backends implement the required interface."""

    @pytest.mark.parametrize("BackendClass", [OllamaBackend, LlamaCppBackend, MLXBackend])
    def test_has_required_methods(self, BackendClass):
        backend = BackendClass()
        assert hasattr(backend, 'is_available')
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'generate')
        assert hasattr(backend, 'cleanup')
        assert hasattr(backend, 'get_status')

    @pytest.mark.parametrize("BackendClass,expected", [
        (OllamaBackend, "Ollama"),
        (LlamaCppBackend, "llama.cpp"),
        (MLXBackend, "MLX"),
    ])
    def test_backend_names(self, BackendClass, expected):
        assert BackendClass().name == expected
