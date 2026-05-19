"""
Apple Silicon specific hardware metrics collection.
Gathers GPU core count, Neural Engine status, memory bandwidth,
and optional energy monitoring data.
"""

import subprocess
import platform
from typing import Dict, Any, Optional


def get_apple_silicon_details() -> Dict[str, Any]:
    """
    Collect detailed Apple Silicon hardware capabilities.
    Returns a dict with GPU cores, Neural Engine, memory bandwidth etc.
    """
    details = {
        "gpu_cores": _get_gpu_cores(),
        "neural_engine": _check_neural_engine(),
        "memory_bandwidth_gbps": _get_memory_bandwidth(),
        "metal_support": _check_metal_support(),
        "chip_generation": _detect_chip_generation(),
    }
    return details


def _get_gpu_cores() -> Optional[int]:
    """Get number of GPU cores on Apple Silicon."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            if "Total Number of Cores" in line and "GPU" not in line:
                continue
            if "Cores" in line and ("GPU" in line or "Metal" in line):
                parts = line.strip().split(":")
                if len(parts) > 1:
                    try:
                        return int(parts[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass

    # Fallback: estimate from chip model
    chip_estimates = {
        "M1": 8, "M1 Pro": 16, "M1 Max": 32, "M1 Ultra": 64,
        "M2": 10, "M2 Pro": 19, "M2 Max": 38, "M2 Ultra": 76,
        "M3": 10, "M3 Pro": 18, "M3 Max": 40, "M3 Ultra": 80,
        "M4": 10, "M4 Pro": 20, "M4 Max": 40,
    }
    chip = _detect_chip_generation()
    return chip_estimates.get(chip)


def _check_neural_engine() -> bool:
    """Check if Neural Engine is present (all Apple Silicon has it)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5)
        brand = result.stdout.strip().lower()
        # All Apple Silicon (M1+) chips have Neural Engine
        return any(f"m{i}" in brand for i in range(1, 10))
    except Exception:
        return False


def _get_memory_bandwidth() -> Optional[float]:
    """
    Estimate memory bandwidth in GB/s based on chip model.
    Apple doesn't expose this via sysctl, so we use known specs.
    """
    bandwidth_map = {
        "M1": 68.25, "M1 Pro": 200.0, "M1 Max": 400.0, "M1 Ultra": 800.0,
        "M2": 100.0, "M2 Pro": 200.0, "M2 Max": 400.0, "M2 Ultra": 800.0,
        "M3": 100.0, "M3 Pro": 150.0, "M3 Max": 300.0, "M3 Ultra": 800.0,
        "M4": 120.0, "M4 Pro": 273.0, "M4 Max": 546.0,
    }
    chip = _detect_chip_generation()
    return bandwidth_map.get(chip)


def _check_metal_support() -> bool:
    """Check if Metal GPU API is available."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10)
        return "Metal" in result.stdout
    except Exception:
        return False


def _detect_chip_generation() -> str:
    """Detect the Apple Silicon chip generation."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5)
        brand = result.stdout.strip().lower()

        for gen in range(4, 0, -1):
            prefix = f"m{gen}"
            if prefix in brand:
                if "ultra" in brand:
                    return f"M{gen} Ultra"
                elif "max" in brand:
                    return f"M{gen} Max"
                elif "pro" in brand:
                    return f"M{gen} Pro"
                else:
                    return f"M{gen}"
    except Exception:
        pass
    return "Unknown"
