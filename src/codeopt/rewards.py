"""Correctness-gated execution rewards."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    syntax_error: float = -1.0
    failed: float = 0.0
    correct: float = 0.5
    faster: float = 1.0
    minimum_speedup: float = 1.05

    def __post_init__(self):
        if self.minimum_speedup <= 1:
            raise ValueError("minimum_speedup must exceed 1")


def execution_reward(candidate, baseline, config=None):
    config = config or RewardConfig()
    if not baseline.accepted or not baseline.runtime_s or baseline.runtime_s <= 0:
        raise ValueError("The baseline must pass a nonempty test suite before reward computation")
    if candidate.verdict == "syntax_error":
        return config.syntax_error
    if not candidate.accepted:
        return config.failed
    if not candidate.runtime_s or candidate.runtime_s <= 0:
        raise ValueError("Accepted candidate has no valid timing")
    speedup = baseline.runtime_s / candidate.runtime_s
    return config.faster if speedup >= config.minimum_speedup else config.correct
