import csv

import pytest

from codeopt.data import Example, convert_csv, load_examples, split_by_problem, write_json


def test_group_split_has_no_leakage_and_is_reproducible():
    examples = [
        Example(str(group), f"print({variant})", "print(1)")
        for group in range(10)
        for variant in range(3)
    ]
    train, test = split_by_problem(examples, 0.3, 42)
    assert {e.problem_id for e in train}.isdisjoint(e.problem_id for e in test)
    assert len(train) + len(test) == len(examples)
    assert len({e.problem_id for e in test}) == 3
    assert (train, test) == split_by_problem(examples, 0.3, 42)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_bad_split_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        split_by_problem([Example("a", "print(1)", "print(1)")], fraction)


def test_single_problem_cannot_be_split():
    with pytest.raises(ValueError, match="two distinct"):
        split_by_problem([Example("a", "print(1)", "print(1)")])


def test_ids_are_content_bound_and_duplicate_rows_removed(tmp_path):
    example = Example("p001", "print(1)", "print(1)")
    path = tmp_path / "data.json"
    write_json(path, [example.to_dict(), example.to_dict()])
    assert load_examples(path) == [example]
    modified = {**example.to_dict(), "input": "print(2)"}
    write_json(path, [modified])
    with pytest.raises(ValueError, match="sample_id"):
        load_examples(path)


def test_problem_id_cannot_escape_test_root():
    with pytest.raises(ValueError):
        Example("../private", "print(1)", "print(1)")


def test_csv_filters_language_syntax_and_duplicates(tmp_path):
    path = tmp_path / "pairs.tsv"
    fields = ["problem_id", "language", "code_v0", "code_v1"]
    rows = [
        ["1", "Python", "print(1)", "print(1)"],
        ["1", "Python", "print(1)", "print(1)"],
        ["2", "C++", "int main() {}", "int main() {}"],
        ["3", "Python", "print 1", "print(1)"],
    ]
    with path.open("w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)
    examples, counts = convert_csv(path)
    assert len(examples) == 1
    assert counts == {"read": 4, "non_python": 1, "invalid_python": 1, "duplicates": 1, "kept": 1}
