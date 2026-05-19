"""Tests for the benchmark runner's aggregation and statistics logic."""

import pytest
import statistics
from bench.runner import BenchmarkRunner
from bench.config import BenchmarkConfig, BenchmarkMode


class TestBuildOutput:
    """Test the _build_output aggregation logic with synthetic run data."""

    def _make_runner_with_results(self, results):
        runner = BenchmarkRunner(BenchmarkConfig())
        runner.results = results
        return runner

    def _make_run(self, backend="Ollama", model="test", quant="Q4_0",
                  tps=50.0, ttft_ms=100.0, ram=1200.0, quality=70.0,
                  error=None, iteration=1):
        return {
            "backend_name": backend, "model_name": model,
            "quantization": quant, "size_label": "1.5B",
            "tokens_per_second": tps,
            "time_to_first_token_ms": ttft_ms,
            "total_generation_time_s": 5.0,
            "peak_ram_mb": ram,
            "quality_score": quality,
            "model_load_time_s": 1.0,
            "error": error,
            "iteration": iteration,
        }

    def test_single_run_aggregation(self):
        runner = self._make_runner_with_results([
            self._make_run(tps=50.0, ttft_ms=100.0, ram=1200.0, quality=70.0)
        ])
        output = runner._build_output()
        summaries = output["summaries"]
        assert len(summaries) == 1
        assert summaries[0]["tokens_per_second"]["mean"] == 50.0
        assert summaries[0]["ttft_ms"]["mean"] == 100.0

    def test_multiple_runs_mean(self):
        runner = self._make_runner_with_results([
            self._make_run(tps=40.0, iteration=1),
            self._make_run(tps=50.0, iteration=2),
            self._make_run(tps=60.0, iteration=3),
        ])
        output = runner._build_output()
        s = output["summaries"][0]
        assert s["tokens_per_second"]["mean"] == 50.0
        assert s["tokens_per_second"]["min"] == 40.0
        assert s["tokens_per_second"]["max"] == 60.0

    def test_std_deviation_calculation(self):
        runs = [self._make_run(tps=v) for v in [40.0, 50.0, 60.0]]
        runner = self._make_runner_with_results(runs)
        output = runner._build_output()
        expected_std = statistics.stdev([40.0, 50.0, 60.0])
        actual_std = output["summaries"][0]["tokens_per_second"]["std"]
        assert abs(actual_std - expected_std) < 0.01

    def test_error_runs_excluded(self):
        runner = self._make_runner_with_results([
            self._make_run(tps=50.0),
            self._make_run(tps=0.0, error="timeout"),
            self._make_run(tps=60.0),
        ])
        output = runner._build_output()
        s = output["summaries"][0]
        assert s["num_runs"] == 2  # Error run excluded
        assert s["tokens_per_second"]["mean"] == 55.0  # (50+60)/2

    def test_all_errors_no_summary(self):
        runner = self._make_runner_with_results([
            self._make_run(error="fail1"),
            self._make_run(error="fail2"),
        ])
        output = runner._build_output()
        assert len(output["summaries"]) == 0

    def test_multiple_models_separate_summaries(self):
        runner = self._make_runner_with_results([
            self._make_run(model="model-a", tps=50.0),
            self._make_run(model="model-b", tps=70.0),
        ])
        output = runner._build_output()
        assert len(output["summaries"]) == 2

    def test_metadata_present(self):
        runner = self._make_runner_with_results([self._make_run()])
        output = runner._build_output()
        assert "metadata" in output
        assert "benchmark_version" in output["metadata"]
        assert "system" in output["metadata"]

    def test_raw_results_preserved(self):
        runs = [self._make_run(iteration=i) for i in range(3)]
        runner = self._make_runner_with_results(runs)
        output = runner._build_output()
        assert len(output["raw_results"]) == 3


class TestBenchmarkConfig:
    """Test benchmark configuration defaults and properties."""

    def test_default_mode(self):
        config = BenchmarkConfig()
        assert config.mode == BenchmarkMode.STANDARD

    def test_num_iterations_quick(self):
        config = BenchmarkConfig(mode=BenchmarkMode.QUICK)
        assert config.num_iterations == 1

    def test_num_iterations_standard(self):
        config = BenchmarkConfig(mode=BenchmarkMode.STANDARD)
        assert config.num_iterations == 2

    def test_num_iterations_full(self):
        config = BenchmarkConfig(mode=BenchmarkMode.FULL)
        assert config.num_iterations == 3

    def test_default_temperature(self):
        config = BenchmarkConfig()
        assert config.temperature == 0.7
