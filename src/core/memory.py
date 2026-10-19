"""Sampled process-memory measurements for expensive indexing stages."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import psutil

logger = logging.getLogger(__name__)
_SAMPLE_INTERVAL_SECONDS = 0.1


@contextmanager
def measure_memory(stage: str, **context: Any) -> Iterator[None]:
    """Log start, end, and sampled peak RSS while a stage runs."""
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    sample_error: list[Exception] = []
    stop_sampling = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop_sampling.wait(_SAMPLE_INTERVAL_SECONDS):
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception as exc:  # pragma: no cover - process exits during measurement
                sample_error.append(exc)
                stop_sampling.set()

    sampler = threading.Thread(target=sample_rss, name="memory-sampler", daemon=True)
    started = perf_counter()
    status = "completed"
    logger.info(
        "stage=%s status=started memory_rss_mb=%.1f %s",
        stage,
        start_rss / (1024 * 1024),
        " ".join(f"{key}={value}" for key, value in context.items()),
    )
    sampler.start()
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        stop_sampling.set()
        sampler.join()
        end_rss = process.memory_info().rss
        if sample_error:
            raise RuntimeError("Unable to sample process memory during indexing") from sample_error[0]
        peak_rss = max(peak_rss, end_rss)
        logger.info(
            "stage=%s status=%s duration_seconds=%.3f memory_rss_mb=%.1f "
            "memory_peak_rss_mb=%.1f memory_delta_mb=%.1f %s",
            stage,
            status,
            perf_counter() - started,
            end_rss / (1024 * 1024),
            peak_rss / (1024 * 1024),
            (peak_rss - start_rss) / (1024 * 1024),
            " ".join(f"{key}={value}" for key, value in context.items()),
        )
