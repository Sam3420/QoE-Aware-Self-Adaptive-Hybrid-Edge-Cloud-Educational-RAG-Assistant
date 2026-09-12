"""Deterministic evaluation of HybridRoutingService policy decisions.

Run from the repository root:
    python evaluation/hybrid_routing_evaluation.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.hybrid_routing_service import HybridRoutingService


EVALUATION_DIR = Path(__file__).resolve().parent
SCENARIOS_FILE = EVALUATION_DIR / "hybrid_routing_scenarios.json"
RESULTS_FILE = EVALUATION_DIR / "hybrid_routing_results.json"
SUMMARY_FILE = EVALUATION_DIR / "hybrid_routing_summary.csv"


def load_scenarios(path: Path = SCENARIOS_FILE) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        router = HybridRoutingService(
            local_provider_available=bool(scenario["local_provider_available"]),
            cloud_provider_available=bool(scenario["cloud_provider_available"]),
        )
        actual_provider = router.decide_provider(
            strategy=str(scenario["strategy"]),
            current_runtime_configuration=scenario.get("current_runtime_configuration"),
            latency_ms=scenario.get("latency_ms"),
            qoe_score=scenario.get("qoe_score"),
            recent_qoe_scores=scenario.get("recent_qoe_scores"),
        )
        expected_provider = str(scenario["expected_provider"])
        results.append(
            {
                "id": scenario["id"],
                "difficulty": scenario["difficulty"],
                "description": scenario["description"],
                "strategy": scenario["strategy"],
                "expected_provider": expected_provider,
                "actual_provider": actual_provider,
                "passed": actual_provider == expected_provider,
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result["difficulty"])].append(result)

    rows: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        group = groups.get(difficulty, [])
        count = len(group)
        passed = sum(bool(result["passed"]) for result in group)
        rows.append(
            {
                "difficulty": difficulty,
                "scenario_count": count,
                "passed": passed,
                "failed": count - passed,
                "pass_rate": round(passed / count, 3) if count else 0.0,
            }
        )
    return rows


def write_outputs(results: list[dict[str, Any]]) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with SUMMARY_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["difficulty", "scenario_count", "passed", "failed", "pass_rate"],
        )
        writer.writeheader()
        writer.writerows(summarize(results))


def main() -> int:
    results = evaluate_scenarios(load_scenarios())
    write_outputs(results)
    failures = [result for result in results if not result["passed"]]
    for row in summarize(results):
        print(row)
    if failures:
        print("Failed scenarios:")
        for result in failures:
            print(f"- {result['id']}: expected {result['expected_provider']}, got {result['actual_provider']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
