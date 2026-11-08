import json

import pytest

from codeopt.cli import main
from codeopt.data import Example, write_json


def test_prepare_cli_outputs_disjoint_splits(tmp_path, capsys):
    source = tmp_path / "pairs.json"
    write_json(source, [Example(str(i), "print(1)", "print(1)").to_dict() for i in range(5)])
    output = tmp_path / "prepared"
    main(["prepare", "--input", str(source), "--output-dir", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert set(report["train_problems"]).isdisjoint(report["test_problems"])
    assert (output / "train.json").is_file() and (output / "test.json").is_file()


def test_invalid_options_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--input", "does-not-exist", "--output-dir", "unused"])
    assert error.value.code == 2
    assert "codeopt:" in capsys.readouterr().err
