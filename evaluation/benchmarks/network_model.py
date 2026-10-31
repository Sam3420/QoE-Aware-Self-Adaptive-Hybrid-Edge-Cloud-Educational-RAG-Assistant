"""Application-level network emulation for the replay (Results 3-4): what a cloud call costs on a given network.

The answer matrix measured every cloud call on the benchmark machine's real network (the base profile). Under an
emulated profile p a call takes its measured time plus the difference the network makes:

    extra = R·(rtt_p − rtt_base) + Σ_R jitter + transfer_p − transfer_base + lost packets × RTO

R = 4 round trips per call (DNS, TCP, TLS 1.3, request: the cloud client opens a new connection per call); transfer
= bytes × 8 / bandwidth; each of the call's packets is lost with probability `loss` and costs one retransmission
timeout RTO = max(300 ms, 2·rtt). The edge model runs on the device, so local answers are unaffected.

Outages: "offline" (no route: the call fails within ~50 ms) and "blackhole" (connected but no internet, e.g. a
captive portal: the call waits CLOUD_LLM_TIMEOUT_S). Either way the app then falls back to the local model.
The model is checked against live injected delays by validate_network_model.py.
"""
import math
import random
from dataclasses import dataclass

ROUND_TRIPS = 4
HEADER_BYTES = 1500  # request + response headers, TLS records
CHARS_TO_BYTES = 1.1  # mostly ASCII JSON


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    rtt_ms: float
    jitter_ms: float = 0.0
    loss: float = 0.0
    down_mbps: float = 100.0
    up_mbps: float = 50.0
    outage: str | None = None  # None | "offline" | "blackhole"
    source: str = "assumed"  # "measured: <label>" when built from network_profiles.csv


# Defaults until measured profiles exist (python -m evaluation.benchmarks.network_profile): typical fixed broadband,
# 4G and weak rural mobile conditions. The report must state which profiles were measured and which assumed.
DEFAULT_PROFILES = {
    "good": NetworkProfile("good", rtt_ms=40, jitter_ms=5, loss=0.0, down_mbps=50, up_mbps=20),
    "moderate": NetworkProfile("moderate", rtt_ms=120, jitter_ms=30, loss=0.005, down_mbps=10, up_mbps=4),
    "poor": NetworkProfile("poor", rtt_ms=600, jitter_ms=200, loss=0.03, down_mbps=0.5, up_mbps=0.25),
    "offline": NetworkProfile("offline", rtt_ms=0, outage="offline"),
    "blackhole": NetworkProfile("blackhole", rtt_ms=0, outage="blackhole"),
}
OFFLINE_FAIL_MS = 50.0


def transfer_ms(up_bytes: float, down_bytes: float, p: NetworkProfile) -> float:
    return (up_bytes * 8 / (p.up_mbps * 1e6) + down_bytes * 8 / (p.down_mbps * 1e6)) * 1000


def cloud_call_extra_ms(prompt_chars: float, answer_chars: float, p: NetworkProfile, base: NetworkProfile,
                        rng: random.Random) -> float:
    """Extra time one cloud call takes on profile p compared with the base network it was measured on."""
    up = prompt_chars * CHARS_TO_BYTES + HEADER_BYTES
    down = answer_chars * CHARS_TO_BYTES + HEADER_BYTES
    rtts = sum(max(0.0, p.rtt_ms + rng.gauss(0.0, p.jitter_ms)) for _ in range(ROUND_TRIPS))
    extra = rtts - ROUND_TRIPS * base.rtt_ms + transfer_ms(up, down, p) - transfer_ms(up, down, base)
    packets = 2 * ROUND_TRIPS + math.ceil((up + down) / 1460)
    lost = sum(rng.random() < p.loss for _ in range(packets))
    return max(0.0, extra + lost * max(300.0, 2 * p.rtt_ms))


def cloud_fail_ms(p: NetworkProfile, timeout_s: float) -> float | None:
    """Time until a cloud call fails on this profile; None when it goes through."""
    if p.outage == "offline":
        return OFFLINE_FAIL_MS
    if p.outage == "blackhole":
        return timeout_s * 1000
    return None


def probe_rtt_ms(p: NetworkProfile, rng: random.Random, fail_ms: float) -> float:
    """The raw time of collect_context's RTT probe on this profile (fail_ms when unreachable); apply the probe
    timeout with probe_seen()."""
    if p.outage:
        return fail_ms
    # one HEAD request: DNS + TCP + TLS + request ≈ 4 RTTs, as the app's probe measures it
    return sum(max(0.0, p.rtt_ms + rng.gauss(0.0, p.jitter_ms)) for _ in range(ROUND_TRIPS))


def probe_seen(raw_ms: float, timeout_s: float, fail_ms: float) -> float:
    """What the context records: the probe time, or RTT_FAIL_MS when the probe did not finish within its timeout."""
    return fail_ms if raw_ms > timeout_s * 1000 else raw_ms


def profiles_from_csv(path: str, mapping: dict[str, str]) -> dict[str, NetworkProfile]:
    """Measured profiles: mapping {profile name: label in network_profiles.csv} (median over that label's rows)."""
    import pandas as pd

    rows = pd.read_csv(path)
    out = dict(DEFAULT_PROFILES)
    for name, label in mapping.items():
        r = rows[rows["label"] == label]
        if r.empty:
            raise ValueError(f"no rows labelled {label!r} in {path}")
        med = r.median(numeric_only=True)
        out[name] = NetworkProfile(
            name, rtt_ms=float(med["rtt_median_ms"]), jitter_ms=float(med.get("jitter_ms") or 0.0),
            loss=float(med.get("loss") or 0.0),
            down_mbps=float(med["down_mbps"]) if not math.isnan(med.get("down_mbps", math.nan)) else 10.0,
            up_mbps=float(med["up_mbps"]) if not math.isnan(med.get("up_mbps", math.nan)) else 4.0,
            source=f"measured: {label} (n={len(r)})")
    return out
