"""
Memory and system metrics collection.
Uses psutil for process-level RAM monitoring with a background sampling thread.
"""

import psutil
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SystemInfo:
    """System hardware and software information."""
    chip: str = ""
    cpu_cores: int = 0
    cpu_brand: str = ""
    total_ram_gb: float = 0.0
    os_version: str = ""
    python_version: str = ""
    gpu_cores: Optional[int] = None
    neural_engine: bool = False
    memory_bandwidth_gbps: Optional[float] = None
    metal_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chip": self.chip,
            "cpu_cores": self.cpu_cores,
            "cpu_brand": self.cpu_brand,
            "total_ram_gb": round(self.total_ram_gb, 1),
            "os_version": self.os_version,
            "python_version": self.python_version,
        }
        if self.gpu_cores:
            d["gpu_cores"] = self.gpu_cores
        if self.neural_engine:
            d["neural_engine"] = True
        if self.memory_bandwidth_gbps:
            d["memory_bandwidth_gbps"] = self.memory_bandwidth_gbps
        if self.metal_support:
            d["metal_support"] = True
        return d


def get_system_info() -> SystemInfo:
    """Collect Apple Silicon system information."""
    from .apple_metrics import get_apple_silicon_details, _detect_chip_generation

    info = SystemInfo()
    info.cpu_cores = psutil.cpu_count(logical=True)
    info.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    info.os_version = f"macOS {platform.mac_ver()[0]}"
    info.python_version = platform.python_version()

    # Get Apple Silicon chip info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5)
        info.cpu_brand = result.stdout.strip()
        info.chip = _detect_chip_generation()
        if info.chip == "Unknown":
            info.chip = info.cpu_brand
    except Exception:
        info.chip = "Unknown Apple Silicon"

    # Get Apple Silicon specific details
    try:
        details = get_apple_silicon_details()
        info.gpu_cores = details.get("gpu_cores")
        info.neural_engine = details.get("neural_engine", False)
        info.memory_bandwidth_gbps = details.get("memory_bandwidth_gbps")
        info.metal_support = details.get("metal_support", False)
    except Exception:
        pass

    return info


class MemoryTracker:
    """
    Context manager for tracking memory usage during inference.
    Runs a background thread that samples RSS at regular intervals.
    """

    def __init__(self, pid: Optional[int] = None, interval: float = 0.1):
        self.pid = pid or psutil.Process().pid
        self.interval = interval
        self._samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0

    def _sample_loop(self):
        try:
            proc = psutil.Process(self.pid)
            while self._running:
                try:
                    mem = proc.memory_info()
                    rss_mb = mem.rss / (1024 ** 2)
                    self._samples.append(rss_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(self.interval)
        except Exception:
            pass

    def __enter__(self):
        try:
            proc = psutil.Process(self.pid)
            self.baseline_mb = proc.memory_info().rss / (1024 ** 2)
        except Exception:
            self.baseline_mb = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._samples:
            self.peak_mb = max(self._samples)
        else:
            self.peak_mb = self.baseline_mb

    @property
    def delta_mb(self) -> float:
        return max(0, self.peak_mb - self.baseline_mb)

    @property
    def samples(self) -> List[float]:
        return self._samples.copy()


def get_current_ram_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 ** 2)
