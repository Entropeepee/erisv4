"""
eris/server/system_stats.py
===========================
Host system telemetry for the cockpit (Tier 7): CPU / RAM / GPU / VRAM /
temperatures, so you can see how hard the machine is working while Eris runs.

Everything degrades gracefully:
  * CPU% / RAM        -> psutil (cross-platform).
  * GPU util/VRAM/temp -> `nvidia-smi` (you have an RTX 5080); parsed from CSV.
  * CPU temperature   -> psutil.sensors_temperatures() where supported (Linux),
                         else LibreHardwareMonitor's WMI namespace on Windows if
                         present; otherwise reported as None.
Any unavailable metric is returned as None rather than raising.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _nvidia_smi() -> List[Dict[str, Any]]:
    """Return a list of per-GPU dicts via nvidia-smi, or [] if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    q = ("utilization.gpu,memory.used,memory.total,temperature.gpu,"
         "power.draw,power.limit,name")
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        ).stdout.strip()
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        def f(x):
            try: return float(x)
            except Exception: return None
        gpus.append({
            "name": parts[6],
            "gpu_util_pct": f(parts[0]),
            "vram_used_mb": f(parts[1]),
            "vram_total_mb": f(parts[2]),
            "gpu_temp_c": f(parts[3]),
            "power_draw_w": f(parts[4]),
            "power_limit_w": f(parts[5]),
        })
    return gpus


def _cpu_temp_c() -> Optional[float]:
    # Linux / some hardware via psutil
    if psutil and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures() or {}
            for key in ("coretemp", "k10temp", "zenpower", "acpitz"):
                if temps.get(key):
                    return float(temps[key][0].current)
            for arr in temps.values():
                if arr:
                    return float(arr[0].current)
        except Exception:
            pass
    # Windows: LibreHardwareMonitor / OpenHardwareMonitor WMI namespace, if running
    try:
        import wmi  # type: ignore
        for ns in ("root\\LibreHardwareMonitor", "root\\OpenHardwareMonitor"):
            try:
                w = wmi.WMI(namespace=ns)
                cores = [s for s in w.Sensor()
                         if s.SensorType == "Temperature" and "CPU" in (s.Name or "")]
                if cores:
                    return float(sorted(cores, key=lambda s: s.Value or 0)[-1].Value)
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_system_stats() -> Dict[str, Any]:
    """One snapshot of host telemetry for the /api/system endpoint."""
    stats: Dict[str, Any] = {
        "cpu_pct": None, "cpu_temp_c": None,
        "ram_used_gb": None, "ram_total_gb": None, "ram_pct": None,
        "gpus": [], "ok": True,
    }
    if psutil:
        try:
            stats["cpu_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats["ram_used_gb"] = round(vm.used / 1024**3, 2)
            stats["ram_total_gb"] = round(vm.total / 1024**3, 2)
            stats["ram_pct"] = vm.percent
        except Exception:
            pass
    stats["cpu_temp_c"] = _cpu_temp_c()
    stats["gpus"] = _nvidia_smi()
    return stats
