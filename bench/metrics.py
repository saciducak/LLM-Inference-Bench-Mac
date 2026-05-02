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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chip": self.chip,
            "cpu_cores": self.cpu_cores,
            "cpu_brand": self.cpu_brand,
            "total_ram_gb": round(self.total_ram_gb, 1),
            "os_version": self.os_version,
            "python_version": self.python_version,
        }


def get_system_info() -> SystemInfo:
    """Collect Apple Silicon system information."""
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
        # Extract chip name
        brand = info.cpu_brand.lower()
        if "m1" in brand:
            if "pro" in brand: info.chip = "M1 Pro"
            elif "max" in brand: info.chip = "M1 Max"
            elif "ultra" in brand: info.chip = "M1 Ultra"
            else: info.chip = "M1"
        elif "m2" in brand:
            if "pro" in brand: info.chip = "M2 Pro"
            elif "max" in brand: info.chip = "M2 Max"
            elif "ultra" in brand: info.chip = "M2 Ultra"
            else: info.chip = "M2"
        elif "m3" in brand:
            if "pro" in brand: info.chip = "M3 Pro"
            elif "max" in brand: info.chip = "M3 Max"
            elif "ultra" in brand: info.chip = "M3 Ultra"
            else: info.chip = "M3"
        elif "m4" in brand:
            if "pro" in brand: info.chip = "M4 Pro"
            elif "max" in brand: info.chip = "M4 Max"
            else: info.chip = "M4"
        else:
            info.chip = info.cpu_brand
    except Exception:
        info.chip = "Unknown Apple Silicon"

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
