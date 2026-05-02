#!/usr/bin/env python3
"""
LLM Inference Bench — CLI Entry Point
======================================
Benchmark LLM inference performance on Apple Silicon Macs.

Usage:
    python run_benchmark.py --quick                    # Fast sanity check
    python run_benchmark.py --full                     # Comprehensive benchmark
    python run_benchmark.py --runtime ollama           # Single runtime
    python run_benchmark.py --dashboard                # Open dashboard
    python run_benchmark.py --quick --runtime ollama   # Quick + specific runtime
"""

import argparse
import sys
import os
import webbrowser
import http.server
import threading

from bench.config import BenchmarkConfig, BenchmarkMode
from bench.runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference Bench — Apple Silicon Mac Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --quick                  Quick sanity check (1 prompt, 1 iter)
  python run_benchmark.py --full                   Full benchmark (all prompts, 3 iters)
  python run_benchmark.py --runtime ollama         Benchmark only Ollama
  python run_benchmark.py --runtime mlx            Benchmark only MLX
  python run_benchmark.py --max-tokens 128         Limit generation length
  python run_benchmark.py --dashboard              Open interactive dashboard
  python run_benchmark.py --check                  Check runtime availability
        """,
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true",
                            help="Quick mode: 1 prompt, 1 iteration")
    mode_group.add_argument("--full", action="store_true",
                            help="Full mode: all prompts, 3 iterations")
    mode_group.add_argument("--dashboard", action="store_true",
                            help="Open the interactive results dashboard")
    mode_group.add_argument("--check", action="store_true",
                            help="Check runtime dependencies and exit")

    # Filters
    parser.add_argument("--runtime", type=str, nargs="+",
                        choices=["ollama", "llama.cpp", "mlx"],
                        help="Run only specific runtime(s)")
    parser.add_argument("--model", type=str, nargs="+",
                        help="Filter models by name substring")

    # Parameters
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--output", type=str,
                        help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()


def run_dashboard():
    """Serve the dashboard on a local HTTP server."""
    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")

    if not os.path.exists(dashboard_dir):
        print("Dashboard directory not found!")
        sys.exit(1)

    port = 8765
    os.chdir(dashboard_dir)

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("localhost", port), handler)

    print(f"\n  🌐 Dashboard running at: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.shutdown()


def run_check():
    """Check runtime availability."""
    print("\n  🔍 Checking runtime dependencies...\n")

    from bench.metrics import get_system_info
    info = get_system_info()
    print(f"  System: {info.chip} | {info.total_ram_gb:.0f}GB RAM | {info.os_version}")
    print(f"  Python: {info.python_version}\n")

    # Check Ollama
    try:
        from bench.backends.ollama_backend import OllamaBackend
        backend = OllamaBackend()
        if backend.is_available():
            models = backend.get_available_models()
            print(f"  ✅ Ollama: Available ({len(models)} models)")
            for m in models[:5]:
                print(f"      • {m}")
            if len(models) > 5:
                print(f"      ... and {len(models) - 5} more")
        else:
            print("  ❌ Ollama: Server not running (start with 'ollama serve')")
    except Exception as e:
        print(f"  ❌ Ollama: {e}")

    # Check llama.cpp
    try:
        from bench.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        if backend.is_available():
            print("  ✅ llama.cpp: llama-cpp-python installed")
        else:
            print("  ❌ llama.cpp: pip install llama-cpp-python")
    except Exception as e:
        print(f"  ❌ llama.cpp: {e}")

    # Check MLX
    try:
        from bench.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        if backend.is_available():
            print("  ✅ MLX: mlx-lm installed")
        else:
            print("  ❌ MLX: pip install mlx-lm")
    except Exception as e:
        print(f"  ❌ MLX: {e}")

    print()


def main():
    args = parse_args()

    if args.dashboard:
        run_dashboard()
        return

    if args.check:
        run_check()
        return

    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"
    else:
        mode = "standard"

    # Build config
    config = BenchmarkConfig(
        mode=BenchmarkMode(mode),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    output = runner.run(
        runtimes=args.runtime,
        models=args.model,
        mode=mode,
    )

    if "error" not in output:
        # Save results
        filepath = runner.save_results(output, args.output)
        runner.print_summary(output)

        print(f"\n  📊 View results in the dashboard:")
        print(f"     python run_benchmark.py --dashboard")
        print(f"     Then load: {filepath}\n")
    else:
        print(f"\n  ❌ Benchmark failed: {output['error']}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
