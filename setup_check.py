#!/usr/bin/env python3
"""
Setup Check — Verify runtime dependencies for LLM Inference Bench.
Run this before benchmarking to ensure everything is properly configured.
"""

import sys
import platform
import subprocess


def main():
    print("\n" + "=" * 55)
    print("  LLM Inference Bench — Setup Check")
    print("=" * 55)

    # System Info
    print("\n  📱 System Information")
    print(f"     OS: macOS {platform.mac_ver()[0]}")
    print(f"     Python: {platform.python_version()}")
    print(f"     Arch: {platform.machine()}")

    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**3)
        print(f"     RAM: {ram:.0f} GB")
    except ImportError:
        print("     RAM: (install psutil)")

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5)
        print(f"     CPU: {result.stdout.strip()}")
    except Exception:
        pass

    # Core Dependencies
    print("\n  📦 Core Dependencies")
    for pkg in ["psutil", "requests", "tqdm", "rich"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"     ✅ {pkg} ({ver})")
        except ImportError:
            print(f"     ❌ {pkg} — pip install {pkg}")

    # Optional Backends
    print("\n  🔧 Inference Backends")

    # Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"     ✅ Ollama server running ({len(models)} models)")
        else:
            print("     ⚠️  Ollama server responded but with error")
    except Exception:
        print("     ❌ Ollama — not running (ollama serve)")

    # llama-cpp-python
    try:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "?")
        print(f"     ✅ llama-cpp-python ({ver})")
    except ImportError:
        print("     ❌ llama-cpp-python — pip install llama-cpp-python")

    # MLX
    try:
        import mlx_lm
        print(f"     ✅ mlx-lm installed")
    except ImportError:
        print("     ❌ mlx-lm — pip install mlx-lm")

    print("\n" + "=" * 55)
    print("  Ready to benchmark? Run:")
    print("  python run_benchmark.py --quick --runtime ollama")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
