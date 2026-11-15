"""Complexity: parameter count, FLOPs per image and measured inference latency.

FLOPs come from PyTorch's own ``torch.utils.flop_counter.FlopCounterMode`` (no extra
dependency); it counts one multiply-add as 2 FLOPs and covers matrix multiplications and
convolutions (elementwise operations are not counted). Latency is always reported with
the platform it was measured on (:func:`platform_info`).
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(fn: Callable[..., Any], *inputs: Any) -> int:
    """FLOPs of one call ``fn(*inputs)``; divide by the batch size for FLOPs per image."""
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn(*inputs)
    return counter.get_total_flops()


def platform_info() -> dict[str, str]:
    info = {"os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cpu": platform.processor() or platform.machine(),
            "threads": str(torch.get_num_threads())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
    return info


def measure_latency(fn: Callable[..., Any], *inputs: Any, warmup: int = 3,
                    repeats: int = 20) -> dict[str, Any]:
    """Wall-clock latency of ``fn(*inputs)`` in ms (mean, std, median) and the platform."""
    cuda = any(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs)
    times = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(*inputs)
            if cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms": statistics.mean(times), "std_ms": statistics.stdev(times),
            "median_ms": statistics.median(times), "repeats": repeats,
            "device": "cuda" if cuda else "cpu", "platform": platform_info()}
