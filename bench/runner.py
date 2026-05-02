"""
Main benchmark orchestrator.
Runs inference across all configured model × runtime × quantization combinations.
"""

import json
import os
import time
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .config import BenchmarkConfig, ModelConfig, BenchmarkMode, get_all_models, get_models_by_runtime
from .prompts import get_prompts, BenchmarkPrompt
from .quality import evaluate_quality
from .metrics import MemoryTracker, get_system_info, get_current_ram_mb
from .backends.base import InferenceResult

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class BenchmarkRunner:
    """Orchestrates benchmark runs across multiple backends and models."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[Dict[str, Any]] = []
        self.system_info = get_system_info()
        self.console = Console() if HAS_RICH else None
        self._backends = {}

    def _init_backends(self, runtimes: Optional[List[str]] = None):
        """Initialize available backends."""
        from .backends.ollama_backend import OllamaBackend
        from .backends.llamacpp_backend import LlamaCppBackend
        from .backends.mlx_backend import MLXBackend

        all_backends = {
            "ollama": OllamaBackend,
            "llama.cpp": LlamaCppBackend,
            "mlx": MLXBackend,
        }

        for name, cls in all_backends.items():
            if runtimes and name not in runtimes:
                continue
            try:
                backend = cls()
                if backend.is_available():
                    self._backends[name] = backend
                    self._log(f"  ✓ {name} backend available", style="green")
                else:
                    self._log(f"  ✗ {name} backend not available", style="yellow")
            except Exception as e:
                self._log(f"  ✗ {name} backend error: {e}", style="red")

    def _log(self, msg: str, style: str = ""):
        if self.console and HAS_RICH:
            self.console.print(msg, style=style)
        else:
            print(msg)

    def _run_single(self, backend, model_config: ModelConfig,
                    prompt: BenchmarkPrompt, iteration: int) -> Dict[str, Any]:
        """Run a single inference and collect all metrics."""
        # Track memory during inference
        with MemoryTracker(interval=self.config.memory_sample_interval) as mem:
            result = backend.generate(
                prompt=prompt.text,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        # Set memory metrics
        result.baseline_ram_mb = mem.baseline_mb
        result.peak_ram_mb = mem.peak_mb
        result.ram_delta_mb = mem.delta_mb

        # Evaluate quality
        if result.output_text and not result.error:
            quality = evaluate_quality(
                output=result.output_text,
                prompt=prompt.text,
                category=prompt.category,
                expected_keywords=prompt.expected_keywords,
                min_tokens=prompt.min_tokens,
                max_tokens=prompt.max_tokens,
            )
            result.quality_score = quality["composite"]
            result.quality_details = quality

        # Set metadata
        result.quantization = model_config.quantization
        result.prompt_category = prompt.category

        entry = result.to_dict()
        entry["iteration"] = iteration
        entry["prompt_text"] = prompt.text
        entry["size_label"] = model_config.size_label
        return entry

    def run(self, runtimes: Optional[List[str]] = None,
            models: Optional[List[str]] = None,
            mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full benchmark suite.
        Returns complete results dict ready for JSON export.
        """
        if mode:
            self.config.mode = BenchmarkMode(mode)

        self._log("\n")
        if self.console and HAS_RICH:
            self.console.print(Panel.fit(
                "[bold cyan]LLM Inference Bench — Apple Silicon[/bold cyan]\n"
                f"Mode: {self.config.mode.value} | Max Tokens: {self.config.max_tokens}",
                border_style="cyan"))
        else:
            self._log("=" * 60)
            self._log("LLM Inference Bench — Apple Silicon")
            self._log(f"Mode: {self.config.mode.value} | Max Tokens: {self.config.max_tokens}")
            self._log("=" * 60)

        # Show system info
        self._log(f"\n  System: {self.system_info.chip} | {self.system_info.total_ram_gb:.0f}GB RAM | {self.system_info.os_version}")

        # Initialize backends
        self._log("\n[bold]Initializing backends...[/bold]" if HAS_RICH else "\nInitializing backends...")
        self._init_backends(runtimes)

        if not self._backends:
            self._log("No backends available! Install ollama, llama-cpp-python, or mlx-lm.", style="red bold")
            return {"error": "No backends available"}

        # Get model configs
        all_models = get_all_models()
        if models:
            all_models = [m for m in all_models if any(n.lower() in m.name.lower() for n in models)]

        # Filter to available backends
        active_models = [m for m in all_models if m.runtime in self._backends]

        if not active_models:
            self._log("No models to benchmark with available backends.", style="yellow")
            return {"error": "No models available"}

        # Get prompts
        prompts = get_prompts(self.config.mode.value)
        iterations = self.config.num_iterations
        total_runs = len(active_models) * len(prompts) * iterations

        self._log(f"\n  Models: {len(active_models)} | Prompts: {len(prompts)} | "
                  f"Iterations: {iterations} | Total runs: {total_runs}\n")

        # Run benchmarks
        self.results = []
        run_idx = 0

        for model_config in active_models:
            backend = self._backends[model_config.runtime]
            self._log(f"\n{'─' * 50}")
            self._log(f"  Model: {model_config.name}", style="bold cyan" if HAS_RICH else "")

            # Load model
            try:
                load_time = backend.load_model(model_config)
                self._log(f"  Loaded in {load_time:.1f}s", style="green")
            except Exception as e:
                self._log(f"  Failed to load: {e}", style="red")
                continue

            # Warmup
            if self.config.warmup_runs > 0:
                self._log("  Warming up...", style="dim")
                try:
                    backend.generate(prompts[0].text, max_tokens=10, temperature=0.7)
                except Exception:
                    pass

            # Run prompts
            for prompt in prompts:
                for i in range(iterations):
                    run_idx += 1
                    self._log(
                        f"  [{run_idx}/{total_runs}] {prompt.category} (iter {i+1})",
                        style="dim")

                    try:
                        entry = self._run_single(backend, model_config, prompt, i + 1)
                        entry["model_load_time_s"] = round(load_time, 3)
                        self.results.append(entry)

                        if not entry.get("error"):
                            self._log(
                                f"    → {entry['tokens_per_second']:.1f} tok/s | "
                                f"TTFT: {entry['time_to_first_token_ms']:.0f}ms | "
                                f"RAM: {entry['peak_ram_mb']:.0f}MB | "
                                f"Quality: {entry['quality_score']:.0f}",
                                style="green" if HAS_RICH else "")
                        else:
                            self._log(f"    ✗ Error: {entry['error']}", style="red")
                    except Exception as e:
                        self._log(f"    ✗ Exception: {e}", style="red")

            # Cleanup between models
            try:
                backend.cleanup()
            except Exception:
                pass

        # Build final results
        output = self._build_output()
        return output

    def _build_output(self) -> Dict[str, Any]:
        """Build the final results dictionary."""
        # Aggregate by model
        aggregated = {}
        for r in self.results:
            key = f"{r['backend_name']}|{r['model_name']}|{r['quantization']}"
            if key not in aggregated:
                aggregated[key] = {
                    "backend": r["backend_name"],
                    "model": r["model_name"],
                    "quantization": r["quantization"],
                    "size_label": r.get("size_label", ""),
                    "runs": [],
                }
            aggregated[key]["runs"].append(r)

        # Calculate statistics
        summaries = []
        for key, data in aggregated.items():
            runs = data["runs"]
            valid = [r for r in runs if not r.get("error")]
            if not valid:
                continue

            summary = {
                "backend": data["backend"],
                "model": data["model"],
                "quantization": data["quantization"],
                "size_label": data["size_label"],
                "num_runs": len(valid),
                "tokens_per_second": {
                    "mean": statistics.mean(r["tokens_per_second"] for r in valid),
                    "median": statistics.median(r["tokens_per_second"] for r in valid),
                    "std": statistics.stdev(r["tokens_per_second"] for r in valid) if len(valid) > 1 else 0,
                    "min": min(r["tokens_per_second"] for r in valid),
                    "max": max(r["tokens_per_second"] for r in valid),
                },
                "ttft_ms": {
                    "mean": statistics.mean(r["time_to_first_token_ms"] for r in valid),
                    "median": statistics.median(r["time_to_first_token_ms"] for r in valid),
                },
                "total_time_s": {
                    "mean": statistics.mean(r["total_generation_time_s"] for r in valid),
                },
                "peak_ram_mb": {
                    "mean": statistics.mean(r["peak_ram_mb"] for r in valid),
                    "max": max(r["peak_ram_mb"] for r in valid),
                },
                "quality_score": {
                    "mean": statistics.mean(r["quality_score"] for r in valid),
                },
                "model_load_time_s": valid[0].get("model_load_time_s", 0),
            }
            summaries.append(summary)

        return {
            "metadata": {
                "benchmark_version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": self.config.mode.value,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.system_info.to_dict(),
            },
            "summaries": summaries,
            "raw_results": self.results,
        }

    def save_results(self, output: Dict, filepath: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if not filepath:
            os.makedirs(self.config.results_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config.results_dir, f"bench_{ts}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self._log(f"\n  Results saved to: {filepath}", style="bold green")
        return filepath

    def print_summary(self, output: Dict):
        """Print a formatted summary table."""
        summaries = output.get("summaries", [])
        if not summaries:
            self._log("No results to display.")
            return

        if self.console and HAS_RICH:
            table = Table(title="Benchmark Results Summary", border_style="cyan")
            table.add_column("Backend", style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("Quant", style="yellow")
            table.add_column("tok/s", justify="right", style="green")
            table.add_column("TTFT (ms)", justify="right")
            table.add_column("RAM (MB)", justify="right")
            table.add_column("Quality", justify="right")

            for s in sorted(summaries, key=lambda x: x["tokens_per_second"]["mean"], reverse=True):
                table.add_row(
                    s["backend"],
                    s["model"],
                    s["quantization"],
                    f"{s['tokens_per_second']['mean']:.1f}",
                    f"{s['ttft_ms']['mean']:.0f}",
                    f"{s['peak_ram_mb']['mean']:.0f}",
                    f"{s['quality_score']['mean']:.0f}",
                )
            self.console.print(table)
        else:
            print("\n" + "=" * 80)
            print(f"{'Backend':<10} {'Model':<25} {'Quant':<8} {'tok/s':>8} {'TTFT':>8} {'RAM':>8} {'Qual':>6}")
            print("-" * 80)
            for s in sorted(summaries, key=lambda x: x["tokens_per_second"]["mean"], reverse=True):
                print(f"{s['backend']:<10} {s['model']:<25} {s['quantization']:<8} "
                      f"{s['tokens_per_second']['mean']:>7.1f} "
                      f"{s['ttft_ms']['mean']:>7.0f} "
                      f"{s['peak_ram_mb']['mean']:>7.0f} "
                      f"{s['quality_score']['mean']:>5.0f}")
