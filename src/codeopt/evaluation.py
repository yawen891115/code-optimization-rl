"""Comparable baseline/candidate execution records and unbiased summary metrics."""

import math
import statistics
from dataclasses import asdict

from .data import dataset_digest
from .execution import discover_tests, test_suite_digest


def count_lines(code):
    return sum(bool(line.strip()) for line in code.splitlines())


def summarize(records):
    accepted = [row for row in records if row["candidate"]["verdict"] == "accepted"]
    comparable = [
        row
        for row in accepted
        if row["baseline"]["verdict"] == "accepted"
        and (row["baseline"]["runtime_s"] or 0) > 0
        and (row["candidate"]["runtime_s"] or 0) > 0
    ]
    ratios = [row["baseline"]["runtime_s"] / row["candidate"]["runtime_s"] for row in comparable]
    reductions = [100 * (1 - 1 / ratio) for ratio in ratios]
    line_reductions = [
        100 * (1 - row["candidate_lines"] / row["baseline_lines"])
        for row in comparable
        if row["baseline_lines"] > 0
    ]
    return {
        "samples": len(records),
        "correct": len(accepted),
        "correctness_percent": 100 * len(accepted) / len(records) if records else None,
        "comparable_samples": len(comparable),
        "geometric_mean_speedup": math.exp(statistics.mean(map(math.log, ratios)))
        if ratios
        else None,
        "mean_runtime_reduction_percent": statistics.mean(reductions) if reductions else None,
        "faster_percent_of_comparable": 100 * sum(r > 1 for r in ratios) / len(ratios)
        if ratios
        else None,
        "mean_nonblank_line_reduction_percent": statistics.mean(line_reductions)
        if line_reductions
        else None,
    }


def evaluate(examples, predictions, runner, test_roots):
    expected = {e.sample_id for e in examples}
    if set(predictions) != expected:
        raise ValueError(
            f"Prediction IDs mismatch: {len(expected - set(predictions))} missing, {len(set(predictions) - expected)} extra"
        )
    test_map = {e.problem_id: discover_tests(test_roots, e.problem_id) for e in examples}
    if any(not tests for tests in test_map.values()):
        missing = sorted(pid for pid, tests in test_map.items() if not tests)
        raise ValueError(f"Missing test cases for: {missing}")
    records = []
    for example in examples:
        generated = predictions[example.sample_id]
        if not isinstance(generated, str):
            raise TypeError("Every generated program must be a string")
        cases = test_map[example.problem_id]
        baseline = runner.run(example.input, cases)
        candidate = runner.run(generated, cases)
        records.append(
            {
                "sample_id": example.sample_id,
                "problem_id": example.problem_id,
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "baseline_lines": count_lines(example.input),
                "candidate_lines": count_lines(generated),
                "generated": generated,
            }
        )
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_digest(examples),
        "tests_sha256": test_suite_digest(test_map),
        "execution": runner.fingerprint(),
        "metrics": summarize(records),
        "records": records,
    }


def compare_reports(reports):
    if len(reports) < 2:
        raise ValueError("Provide at least two labeled evaluation reports")
    first = next(iter(reports.values()))
    for label, report in reports.items():
        for field in ("dataset_sha256", "tests_sha256", "execution"):
            if report[field] != first[field]:
                raise ValueError(f"{label}: {field} differs; comparison would be invalid")
    return {label: summarize(report["records"]) for label, report in reports.items()}
