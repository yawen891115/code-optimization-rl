"""Validated instruction pairs and deterministic splits by problem ID."""

import ast
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

INSTRUCTION = (
    "Optimize this Python program for runtime while preserving its stdin/stdout behavior. "
    "Return only the complete Python code."
)


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Example:
    problem_id: str
    input: str
    output: str
    instruction: str = INSTRUCTION

    def __post_init__(self):
        object.__setattr__(self, "problem_id", str(self.problem_id))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.problem_id):
            raise ValueError("problem_id must contain only letters, digits, underscores or hyphens")
        for field in ("input", "output", "instruction"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")

    @property
    def sample_id(self):
        return digest(asdict(self))[:24]

    def to_dict(self):
        return {"sample_id": self.sample_id, **asdict(self)}


def load_examples(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array or JSONL file")
    examples, seen = [], set()
    for row in rows:
        example = Example(
            problem_id=row["problem_id"],
            input=row["input"],
            output=row["output"],
            instruction=row.get("instruction", INSTRUCTION),
        )
        if "sample_id" in row and row["sample_id"] != example.sample_id:
            raise ValueError("sample_id does not match example content")
        if example.sample_id not in seen:
            examples.append(example)
            seen.add(example.sample_id)
    return examples


def convert_csv(path, delimiter="\t", limit=None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    examples, seen = [], set()
    counts = {"read": 0, "non_python": 0, "invalid_python": 0, "duplicates": 0}
    csv.field_size_limit(16 * 1024 * 1024)
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        required = {"problem_id", "language", "code_v0", "code_v1"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Dataset needs columns: {sorted(required)}")
        for row in reader:
            counts["read"] += 1
            if not row["language"].lower().startswith("python"):
                counts["non_python"] += 1
                continue
            example = Example(row["problem_id"], row["code_v0"], row["code_v1"])
            try:
                ast.parse(example.input)
                ast.parse(example.output)
            except (SyntaxError, ValueError):
                counts["invalid_python"] += 1
                continue
            if example.sample_id in seen:
                counts["duplicates"] += 1
                continue
            seen.add(example.sample_id)
            examples.append(example)
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise ValueError("No usable Python pairs found")
    return examples, {**counts, "kept": len(examples)}


def split_by_problem(examples, test_fraction=0.2, seed=42):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    groups = sorted({e.problem_id for e in examples})
    if len(groups) < 2:
        raise ValueError("At least two distinct problems are needed for a held-out split")
    random.Random(seed).shuffle(groups)
    test_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * test_fraction)))
    held_out = set(groups[:test_count])
    return (
        [e for e in examples if e.problem_id not in held_out],
        [e for e in examples if e.problem_id in held_out],
    )


def dataset_digest(examples):
    return digest(sorted(e.sample_id for e in examples))
