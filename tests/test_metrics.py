import copy

import pytest

from codeopt.evaluation import compare_reports, summarize
from codeopt.execution import ExecutionResult
from codeopt.rewards import execution_reward


def row(baseline, candidate, verdict="accepted"):
    return {
        "baseline": {"verdict": "accepted", "runtime_s": baseline},
        "candidate": {"verdict": verdict, "runtime_s": candidate},
        "baseline_lines": 4,
        "candidate_lines": 2,
    }


def test_slowdowns_are_included_in_runtime_average():
    result = summarize([row(1, 0.5), row(1, 2), row(1, None, "wrong_answer")])
    assert result["correctness_percent"] == pytest.approx(200 / 3)
    assert result["geometric_mean_speedup"] == pytest.approx(1)
    assert result["mean_runtime_reduction_percent"] == pytest.approx(-25)
    assert result["faster_percent_of_comparable"] == 50
    assert result["comparable_samples"] == 2


def test_empty_metrics_are_explicitly_unavailable():
    result = summarize([])
    assert result["samples"] == 0
    assert result["correctness_percent"] is None
    assert result["geometric_mean_speedup"] is None


@pytest.mark.parametrize(
    "verdict,runtime,reward",
    [
        ("syntax_error", None, -1),
        ("wrong_answer", None, 0),
        ("accepted", 0.99, 0.5),
        ("accepted", 0.8, 1),
        ("accepted", 2, 0.5),
    ],
)
def test_reward_is_correctness_gated_with_noise_margin(verdict, runtime, reward):
    baseline = ExecutionResult("accepted", 1, 1, 1)
    candidate = ExecutionResult(verdict, runtime, 1 if verdict == "accepted" else 0, 1)
    assert execution_reward(candidate, baseline) == reward


def test_invalid_baseline_cannot_produce_speedup_reward():
    with pytest.raises(ValueError):
        execution_reward(
            ExecutionResult("accepted", 1, 1, 1), ExecutionResult("no_tests", None, 0, 0)
        )


@pytest.mark.parametrize("field", ["dataset_sha256", "tests_sha256", "execution"])
def test_comparison_rejects_mismatched_conditions(field):
    base = {"dataset_sha256": "a", "tests_sha256": "b", "execution": {}, "records": [row(1, 1)]}
    changed = copy.deepcopy(base)
    changed[field] = "different"
    with pytest.raises(ValueError, match=field):
        compare_reports({"base": base, "changed": changed})
