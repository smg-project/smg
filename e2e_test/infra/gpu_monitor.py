"""GPU utilization monitoring for benchmarks.

This module provides a low-impact GPU monitor that runs in a separate process
and collects utilization samples using NVML or AMD SMI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def _percentile(sorted_samples: list[float], p: float) -> float:
    """Calculate percentile from pre-sorted samples."""
    if not sorted_samples:
        return 0.0
    idx = max(
        0,
        min(len(sorted_samples) - 1, int(round((p / 100.0) * (len(sorted_samples) - 1)))),
    )
    return float(sorted_samples[idx])


def _compute_stats(
    samples: list[float], trim_start: int = 0, trim_end: int = 0
) -> dict[str, float]:
    """Compute statistics for a list of samples.

    Args:
        samples: List of GPU utilization samples
        trim_start: Number of samples to exclude from the beginning (startup period)
        trim_end: Number of samples to exclude from the end (shutdown period)
    """
    if not samples:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "count": 0,
        }

    # Trim startup and shutdown samples
    end_idx = len(samples) - trim_end if trim_end > 0 else len(samples)
    trimmed = samples[trim_start:end_idx]

    if not trimmed:
        trimmed = samples  # Fallback to original if trimming removes all samples

    sorted_trimmed = sorted(trimmed)
    return {
        "mean": sum(trimmed) / len(trimmed),
        "min": sorted_trimmed[0],
        "max": sorted_trimmed[-1],
        "p5": _percentile(sorted_trimmed, 5),
        "p10": _percentile(sorted_trimmed, 10),
        "p25": _percentile(sorted_trimmed, 25),
        "p50": _percentile(sorted_trimmed, 50),
        "p75": _percentile(sorted_trimmed, 75),
        "p90": _percentile(sorted_trimmed, 90),
        "p95": _percentile(sorted_trimmed, 95),
        "count": len(trimmed),
    }


def _monitor_loop(pid: int, output_path: str, interval: float) -> None:
    """Main monitoring loop - runs in separate process.

    Monitors GPU utilization until the target process exits, then writes
    results to output_path as JSON.
    """
    # Lower process priority to minimize impact on benchmark
    try:
        os.nice(10)
    except Exception:
        pass

    # Initialize NVML
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as e:
        logger.warning("Failed to initialize NVML: %s", e)
        _write_empty_result(output_path)
        return

    # Get GPU handles
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
    except Exception as e:
        logger.warning("Failed to get GPU handles: %s", e)
        _write_empty_result(output_path)
        _shutdown_nvml()
        return

    # Collect samples
    per_gpu_samples: dict[str, list[float]] = {str(i): [] for i in range(device_count)}
    overall_samples: list[float] = []

    try:
        while _process_alive(pid):
            try:
                gpu_utils = []
                for idx, handle in enumerate(handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        gpu_utils.append(float(util))
                        per_gpu_samples[str(idx)].append(float(util))
                    except Exception:
                        continue

                if gpu_utils:
                    avg = sum(gpu_utils) / len(gpu_utils)
                    overall_samples.append(avg)
            except Exception:
                pass

            time.sleep(interval)
    finally:
        # Write results
        _write_result(output_path, pid, interval, overall_samples, per_gpu_samples)
        _shutdown_nvml()


def _bdf(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[01][0-9a-fA-F]\.[0-7]", value
    ):
        raise ValueError(f"Invalid full PCI BDF: {value!r}")
    return value.lower()


def _amd_worker_bdfs(torch, gpu_ids: list[int]) -> dict[int, str]:
    # Worker._build_env overrides CUDA per worker. An inherited parent mask
    # would make its logical indices ambiguous here; do not guess through it.
    if any(
        os.environ.get(key, "").strip()
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    ):
        raise ValueError("Worker GPU mapping requires an unmasked parent HIP view")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Missing or duplicate worker GPU IDs")
    if not torch.version.hip:
        raise ValueError("AMD monitoring requires the workers' HIP runtime")
    count = torch.cuda.device_count()
    selected = {}
    for gpu in gpu_ids:
        if type(gpu) is not int or not 0 <= gpu < count:
            raise ValueError("Worker GPU ID outside the actual HIP view")
        props = torch.cuda.get_device_properties(gpu)
        for field, limit in (
            ("pci_domain_id", 0xFFFF),
            ("pci_bus_id", 0xFF),
            ("pci_device_id", 0x1F),
        ):
            value = getattr(props, field)
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError("Incomplete HIP PCI identity")
        # These exact HIP properties are also used by the device identity
        # preflight. GPU PCI functions are zero; SMI roundtrip must agree.
        selected[gpu] = _bdf(
            f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        )
    if len(set(selected.values())) != len(selected):
        raise ValueError("Duplicate worker physical PCI identities")
    return selected


def _amd_handles(smi, selected: dict[int, str]) -> dict[int, object]:
    if not selected or len(set(selected.values())) != len(selected):
        raise ValueError("Missing or duplicate selected physical GPUs")
    by_bdf = {}
    for handle in smi.amdsmi_get_processor_handles():
        bdf = _bdf(smi.amdsmi_get_gpu_device_bdf(handle))
        if bdf in by_bdf:
            raise ValueError("Duplicate AMD SMI PCI identity")
        by_bdf[bdf] = handle
    result = {}
    for gpu, bdf in selected.items():
        handle = by_bdf[bdf]  # Missing selected devices must not become zeroes.
        if smi.amdsmi_get_processor_type(handle) != {"processor_type": "AMD_GPU"}:
            raise ValueError("Selected SMI processor is not an AMD GPU")
        if (
            _bdf(smi.amdsmi_get_gpu_device_bdf(smi.amdsmi_get_processor_handle_from_bdf(bdf)))
            != bdf
        ):
            raise ValueError("AMD SMI PCI identity roundtrip mismatch")
        result[gpu] = handle
    return result


def _amd_sample(smi, handles: dict[int, object]) -> dict[str, float]:
    values = {}
    for gpu, handle in handles.items():
        value = smi.amdsmi_get_gpu_busy_percent(handle)
        # 27.0's UINT32_MAX sentinel is the string "N/A". Do not accept bool,
        # missing data, an unsupported API, or out-of-range values as idle.
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError(f"GPU {gpu} busy percent unavailable: {value!r}")
        values[str(gpu)] = float(value)
    if not values:
        raise ValueError("No selected AMD GPUs")
    return values


def _amd_monitor_loop(pid: int, output_path: str, interval: float, gpu_ids: list[int]) -> None:
    initialized = False
    samples: list[float] = []
    per_gpu: dict[str, list[float]] = {}
    metadata: dict[str, Any] = {
        "backend": "amdsmi",
        "status": "unavailable",
        "sample_times_monotonic": [],
    }
    try:
        # Only the sampler child imports the runtime. CPU tests inject fakes.
        import amdsmi
        import torch

        selected = _amd_worker_bdfs(torch, gpu_ids)
        amdsmi.amdsmi_init()
        initialized = True
        handles = _amd_handles(amdsmi, selected)
        metadata.update(
            package_version=amdsmi.__version__,
            module_origin=amdsmi.__file__,
            library_version=amdsmi.amdsmi_get_lib_version(),
            selected_bdfs={str(gpu): bdf for gpu, bdf in selected.items()},
        )
        per_gpu = {str(gpu): [] for gpu in handles}
        while _process_alive(pid):
            values = _amd_sample(amdsmi, handles)
            # A tick contributes only when every selected worker GPU is valid.
            for gpu, value in values.items():
                per_gpu[gpu].append(value)
            samples.append(sum(values.values()) / len(values))
            metadata["sample_times_monotonic"].append(time.monotonic())
            time.sleep(interval)
        if not samples:
            raise ValueError("AMD SMI produced no complete selected-GPU samples")
        metadata["status"] = "available"
    except Exception as exc:
        metadata.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        logger.warning("AMD GPU utilization unavailable: %s", exc)
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception as exc:
                metadata.update(status="unavailable", reason=f"SMI shutdown failed: {exc}")
        _write_result(output_path, pid, interval, samples, per_gpu, metadata=metadata)


def _process_alive(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _write_empty_result(path: str) -> None:
    """Write empty result file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "count": 0,
                    "overall": {"mean": 0.0},
                    "per_gpu": {},
                    "raw": {"overall": [], "per_gpu": {}},
                },
                f,
            )
    except Exception:
        pass


def _write_result(
    path: str,
    pid: int,
    interval: float,
    overall_samples: list[float],
    per_gpu_samples: dict[str, list[float]],
    *,
    metadata: dict | None = None,
) -> None:
    """Write monitoring results to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            result = {
                "bench_pid": pid,
                "interval_sec": interval,
                "count": len(overall_samples),
                "overall": _compute_stats(overall_samples),
                "per_gpu": {k: _compute_stats(v) for k, v in per_gpu_samples.items()},
                "raw": {"overall": overall_samples, "per_gpu": per_gpu_samples},
            }
            if metadata:
                result.update(metadata)
                if metadata["status"] != "available":
                    result["overall"] = None
                    result["per_gpu"] = {}
            json.dump(result, f)
    except Exception as e:
        logger.warning("Failed to write GPU monitor results: %s", e)


def _shutdown_nvml() -> None:
    """Shutdown NVML."""
    try:
        import pynvml

        pynvml.nvmlShutdown()
    except Exception:
        pass


class GPUMonitor:
    """GPU utilization monitor for benchmarks.

    Usage:
        monitor = GPUMonitor(output_dir="benchmark_results")
        monitor.start(target_pid=12345)
        # ... run benchmark ...
        result = monitor.stop()
        misses = monitor.threshold_misses({"gpu_util_p50_min": 99})
    """

    def __init__(
        self,
        output_dir: str | Path = ".",
        interval: float = 2.0,
        trim_start: int = 3,
        trim_end: int = 2,
        *,
        backend: str = "nvml",
        gpu_ids: list[int] | None = None,
    ):
        if backend not in ("nvml", "amdsmi"):
            raise ValueError(f"Unknown GPU monitor backend: {backend}")
        if interval <= 0 or not interval < float("inf"):
            raise ValueError("GPU sample interval must be positive and finite")
        self.backend = backend
        self.gpu_ids = list(gpu_ids or [])
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.trim_start = trim_start
        self.trim_end = trim_end
        self._process: Process | None = None
        self._output_path: str | None = None
        self._result: dict[str, Any] | None = None

    @property
    def output_path(self) -> str | None:
        """Path to the GPU utilization JSON file."""
        return self._output_path

    def start(self, target_pid: int) -> None:
        """Start monitoring GPU utilization for the target process."""
        self._output_path = str(self.output_dir / "gpu_utilization.json")
        self._result = None

        self._process = Process(
            target=_amd_monitor_loop if self.backend == "amdsmi" else _monitor_loop,
            args=(target_pid, self._output_path, self.interval, self.gpu_ids)
            if self.backend == "amdsmi"
            else (target_pid, self._output_path, self.interval),
            daemon=True,
        )
        self._process.start()
        logger.debug("Started GPU monitor for PID %d", target_pid)

    def stop(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """Stop monitoring and return results."""
        if self._process is None:
            return None

        try:
            self._process.join(timeout=timeout)
        except Exception:
            pass

        if self._process.is_alive():
            try:
                self._process.terminate()
                self._process.join(timeout=timeout)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=timeout)
            except ProcessLookupError:
                pass
        if self._process.is_alive():
            raise RuntimeError("Owned GPU sampler did not exit")

        self._process = None
        self._result = self._read_result()
        return self._result

    def _read_result(self) -> dict[str, Any] | None:
        """Read results from output file and apply trimming.

        Trims startup/shutdown samples and recomputes stats to exclude
        periods where GPU is idle during initialization/teardown.
        """
        if not self._output_path or not os.path.exists(self._output_path):
            return None
        try:
            with open(self._output_path) as f:
                result = json.load(f)

            if result.get("status") == "unavailable":
                return result

            # Apply trimming to raw samples and recompute stats
            raw_samples = result.get("raw", {}).get("overall", [])
            if raw_samples and len(raw_samples) > (self.trim_start + self.trim_end + 2):
                # Recompute overall stats with trimming
                result["overall"] = _compute_stats(raw_samples, self.trim_start, self.trim_end)

                # Also trim per-GPU stats
                raw_per_gpu = result.get("raw", {}).get("per_gpu", {})
                if raw_per_gpu:
                    result["per_gpu"] = {
                        k: _compute_stats(v, self.trim_start, self.trim_end)
                        for k, v in raw_per_gpu.items()
                    }

                logger.debug(
                    "GPU stats after trimming %d start, %d end samples: mean=%.2f%% p50=%.2f%%",
                    self.trim_start,
                    self.trim_end,
                    result["overall"].get("mean", 0.0),
                    result["overall"].get("p50", 0.0),
                )

            return result
        except Exception as e:
            logger.warning("Failed to read GPU monitor result: %s", e)
            return None

    def log_summary(self) -> None:
        """Log a summary of GPU utilization."""
        result = self._result or self._read_result()
        if not result or result.get("status") == "unavailable" or result.get("count", 0) <= 0:
            logger.warning(
                "GPU utilization monitor unavailable: %s",
                (result or {}).get("reason", "no samples"),
            )
            return

        overall = result.get("overall", {})
        logger.info(
            "GPU utilization: mean=%.2f%% p50=%.2f%% (samples=%d)",
            overall.get("mean", 0.0),
            overall.get("p50", 0.0),
            result.get("count", 0),
        )

    def threshold_misses(self, thresholds: dict[str, float] | None) -> list[str]:
        """Return one line per GPU-utilization threshold that was missed.

        Supported thresholds:
            - gpu_util_mean_min: Minimum mean GPU utilization %
            - gpu_util_p50_min: Minimum p50 GPU utilization %
        """
        if not thresholds:
            return []

        result = self._result or self._read_result()
        if self.backend == "amdsmi" and (
            not result or result.get("status") != "available" or result.get("count", 0) <= 0
        ):
            return [
                "GPU utilization unavailable: "
                + str((result or {}).get("reason", "no complete samples"))
            ]
        if not result or result.get("count", 0) <= 0:
            logger.warning("GPU utilization monitor produced no samples")
            return []

        overall = result.get("overall", {})
        misses: list[str] = []

        mean_threshold = thresholds.get("gpu_util_mean_min")
        if mean_threshold is not None:
            mean_value = overall.get("mean", 0.0)
            if mean_value < mean_threshold:
                misses.append(f"GPU utilization mean: {mean_value:.2f}% < {mean_threshold}%")

        p50_threshold = thresholds.get("gpu_util_p50_min")
        if p50_threshold is not None:
            p50_value = overall.get("p50")
            if p50_value is not None:
                if p50_value < p50_threshold:
                    misses.append(f"GPU utilization p50: {p50_value:.2f}% < {p50_threshold}%")
        return misses


def should_monitor(thresholds: dict[str, Any] | None) -> bool:
    """Check if GPU monitoring should be enabled.

    Returns True if:
    - thresholds contains gpu_util_mean_min or gpu_util_p50_min, OR
    - GPU_UTIL_LOG environment variable is truthy
    """
    if thresholds:
        if thresholds.get("gpu_util_mean_min") is not None:
            return True
        if thresholds.get("gpu_util_p50_min") is not None:
            return True

    return os.environ.get("GPU_UTIL_LOG", "").lower() in ("1", "true", "yes")
