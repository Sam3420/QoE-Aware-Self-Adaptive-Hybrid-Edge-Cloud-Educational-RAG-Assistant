"""QoE, SRE, utility, reward and entropy-weight computations (pure functions, no I/O, no config import).

All scores are in [0, 1] with higher = better. Weight groups are validated (exact keys, non-negative,
sum to 1). See docs/adaptive_policy.md for the equations and the missing-data policy.
"""
