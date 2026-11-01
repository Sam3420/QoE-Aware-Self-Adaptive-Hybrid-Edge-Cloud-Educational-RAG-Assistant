"""Measure the network this machine is on (Result 2): RTT, jitter and loss to the cloud LLM, plus bandwidth.

    .venv\\Scripts\\python -m evaluation.benchmarks.network_profile --label college-wifi
    .venv\\Scripts\\python -m evaluation.benchmarks.network_profile --label phone-hotspot-4g --no-speedtest

Run it once per network you want in the report (college Wi-Fi, home broadband, phone hotspot, weak signal, ...),
ideally a few times at different hours. Each run appends one row to evaluation/results/network_profiles.csv; the
network model (network_model.py) builds its emulated profiles from these rows.

- RTT/jitter/loss: TCP connects to the Groq API host (what every cloud answer pays first); jitter is the mean
  absolute difference of consecutive RTTs (RFC 3550 style); a connect that does not complete in 2 s counts as lost.
- HTTP RTT: the app's own probe (one HEAD request to PROBE_URL), as collect_context measures it.
- Bandwidth: speedtest.net (nearest server), unless --no-speedtest.
"""
import argparse
import csv
import socket
import statistics
import time
from datetime import datetime, timezone

import httpx

from evaluation.benchmarks.isolation import RESULTS

OUT = RESULTS / "network_profiles.csv"
HOST, PORT = "api.groq.com", 443


def tcp_rtts(host: str, port: int, n: int, timeout_s: float = 2.0, gap_s: float = 0.2) -> list[float | None]:
    addr = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4]  # resolve once: DNS is not RTT
    out: list[float | None] = []
    for _ in range(n):
        start = time.perf_counter()
        try:
            with socket.create_connection(addr[:2], timeout=timeout_s):
                out.append((time.perf_counter() - start) * 1000)
        except OSError:
            out.append(None)
        time.sleep(gap_s)
    return out


def http_rtts(url: str, n: int) -> list[float]:
    out = []
    for _ in range(n):
        start = time.perf_counter()
        try:
            httpx.head(url, timeout=5, follow_redirects=False)
            out.append((time.perf_counter() - start) * 1000)
        except httpx.HTTPError:
            pass
    return out


def speedtest_mbps() -> dict[str, float | None]:
    try:
        import speedtest

        st = speedtest.Speedtest(secure=True)
        st.get_best_server()
        down, up = st.download() / 1e6, st.upload(pre_allocate=False) / 1e6
        if down <= 0 or up <= 0:  # e.g. a campus firewall: the library reports 0 instead of failing
            raise RuntimeError(f"speedtest returned {down:.2f}/{up:.2f} Mbps")
        return {"down_mbps": round(down, 2), "up_mbps": round(up, 2), "speedtest_ping_ms": round(st.results.ping, 1),
                "speedtest_server": st.results.server.get("sponsor")}
    except Exception as exc:  # blocked by the network, no server, library error: time a plain HTTPS transfer
        return {**cloudflare_mbps(), "speedtest_ping_ms": None, "speedtest_server": f"cloudflare (speedtest: {exc})"[:80]}


def cloudflare_mbps(down_bytes: int = 25_000_000, up_bytes: int = 5_000_000) -> dict[str, float | None]:
    """Bandwidth from one timed download and upload to speed.cloudflare.com (best of 3 each)."""
    def best(fn) -> float | None:
        rates = []
        for _ in range(3):
            try:
                rates.append(fn())
            except httpx.HTTPError:
                pass
        return round(max(rates), 2) if rates else None

    def down() -> float:
        start = time.perf_counter()
        r = httpx.get(f"https://speed.cloudflare.com/__down?bytes={down_bytes}", timeout=60)
        r.raise_for_status()
        return len(r.content) * 8 / (time.perf_counter() - start) / 1e6

    def up() -> float:
        start = time.perf_counter()
        httpx.post("https://speed.cloudflare.com/__up", content=b"0" * up_bytes, timeout=60).raise_for_status()
        return up_bytes * 8 / (time.perf_counter() - start) / 1e6

    return {"down_mbps": best(down), "up_mbps": best(up)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="e.g. college-wifi, home-broadband, phone-hotspot-4g")
    parser.add_argument("--pings", type=int, default=50)
    parser.add_argument("--no-speedtest", action="store_true")
    args = parser.parse_args()

    from config import settings

    rtts = tcp_rtts(HOST, PORT, args.pings)
    ok = [r for r in rtts if r is not None]
    jitter = statistics.mean(abs(a - b) for a, b in zip(ok, ok[1:])) if len(ok) > 1 else None
    probe = http_rtts(settings.probe_url, 10)
    row = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": args.label,
        "host": HOST, "pings": len(rtts), "loss": round(1 - len(ok) / len(rtts), 4),
        "rtt_median_ms": round(statistics.median(ok), 1) if ok else None,
        "rtt_p95_ms": round(sorted(ok)[int(0.95 * (len(ok) - 1))], 1) if ok else None,
        "jitter_ms": round(jitter, 1) if jitter is not None else None,
        "probe_url": settings.probe_url, "probe_rtt_median_ms": round(statistics.median(probe), 1) if probe else None,
        **({} if args.no_speedtest else speedtest_mbps()),
    }
    new = not OUT.exists()
    fields = ["measured_at", "label", "host", "pings", "loss", "rtt_median_ms", "rtt_p95_ms", "jitter_ms", "probe_url",
              "probe_rtt_median_ms", "down_mbps", "up_mbps", "speedtest_ping_ms", "speedtest_server"]
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
