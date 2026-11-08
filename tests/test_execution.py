import pytest

from codeopt.execution import RunConfig, Runner, discover_tests
from codeopt.execution import TestCase as Case


@pytest.fixture
def runner():
    return Runner(RunConfig(backend="local", timeout=0.15, repeats=1, warmups=0))


@pytest.mark.parametrize(
    "code,verdict",
    [
        ("print(int(input()) + 1)", "accepted"),
        ("print(0)", "wrong_answer"),
        ("if:", "syntax_error"),
        ("raise ValueError('test')", "runtime_error"),
        ("while True: pass", "timeout"),
    ],
)
def test_execution_verdicts(runner, code, verdict):
    result = runner.run(code, [Case("one", "1\n", "2\n")])
    assert result.verdict == verdict
    assert result.total == 1
    if verdict == "accepted":
        assert result.runtime_s > 0
        assert result.passed == 1
    else:
        assert result.runtime_s is None


def test_missing_tests_are_never_success(runner):
    assert runner.run("print(1)", []).verdict == "no_tests"


def test_missing_output_rejected_and_only_filename_prefix_replaced(tmp_path):
    root = tmp_path / "input_collection"
    directory = root / "p1"
    directory.mkdir(parents=True)
    (directory / "input.0.txt").write_text("1\n")
    with pytest.raises(ValueError, match="Missing expected output"):
        discover_tests([root], "p1")
    (directory / "output.0.txt").write_text("2\n")
    assert discover_tests([root], "p1")[0].expected == "2\n"


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        discover_tests([tmp_path], "../../outside")
