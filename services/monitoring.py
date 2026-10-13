"""System monitoring for the session context: CPU/memory load and network RTT to the cloud LLM."""
import time

import httpx
import psutil


class ProbeError(RuntimeError):
    """Raised when the RTT probe gets no HTTP response in time."""


class SystemMonitor:
    """Reads device load with psutil and probes RTT with one HTTP HEAD request."""

    def cpu_mem(self) -> tuple[float, float]:
        """Return (cpu, mem) utilisation as fractions in [0, 1]."""
        cpu = psutil.cpu_percent(interval=0.1) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        return min(max(cpu, 0.0), 1.0), min(max(mem, 0.0), 1.0)

    def net_bytes(self) -> int | None:
        """System-wide bytes sent + received since boot (psutil); None if unavailable."""
        try:
            counters = psutil.net_io_counters()
            return int(counters.bytes_sent + counters.bytes_recv)
        except Exception:
            return None

    def probe_rtt(self, url: str, timeout_s: float) -> float:
        """Return the time in ms for one HEAD request; any HTTP status counts as reachable.

        The first request includes DNS + TLS setup, which is part of what a learner's
        first cloud call would pay too.
        """
        start = time.perf_counter()
        try:
            httpx.head(url, timeout=timeout_s, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise ProbeError(f"RTT probe to {url} failed: {type(exc).__name__}: {exc}") from exc
        return (time.perf_counter() - start) * 1000.0
