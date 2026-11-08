"""A handwritten correctness demo; no model training or benchmark claims."""

from pathlib import Path

from .data import Example, write_json
from .evaluation import evaluate
from .execution import RunConfig, Runner


def run_demo(output_dir):
    root = Path(output_dir)
    examples = [
        Example(
            "sum_squares",
            "n = int(input())\ntotal = 0\nfor i in range(1, n + 1):\n    total += i * i\nprint(total)\n",
            "n = int(input())\nprint(n * (n + 1) * (2 * n + 1) // 6)\n",
        ),
        Example(
            "distinct",
            "values = list(map(int, input().split()))\nseen = []\nfor value in values:\n    if value not in seen:\n        seen.append(value)\nprint(len(seen))\n",
            "print(len(set(map(int, input().split()))))\n",
        ),
    ]
    cases = {
        "sum_squares": [("1\n", "1\n"), ("10\n", "385\n"), ("1000\n", "333833500\n")],
        "distinct": [("1 2 2 3\n", "3\n"), ("7 7 7\n", "1\n"), ("-1 0 1 -1\n", "3\n")],
    }
    tests = root / "tests"
    for pid, pairs in cases.items():
        directory = tests / pid
        directory.mkdir(parents=True, exist_ok=True)
        for index, (stdin, expected) in enumerate(pairs):
            (directory / f"input.{index}.txt").write_text(stdin)
            (directory / f"output.{index}.txt").write_text(expected)
    write_json(root / "examples.json", [e.to_dict() for e in examples])
    predictions = {e.sample_id: e.output for e in examples}
    write_json(
        root / "predictions.json",
        [{"sample_id": key, "generated": value} for key, value in predictions.items()],
    )
    report = evaluate(
        examples, predictions, Runner(RunConfig(backend="local", repeats=2, warmups=0)), [tests]
    )
    report["experiment"] = "Handwritten demo programs; no learned model was evaluated"
    write_json(root / "evaluation.json", report)
    return report
