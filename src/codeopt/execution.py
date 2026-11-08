"""A Docker runner by default; explicit local execution for trusted programs."""

import ast
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import digest, write_json


@dataclass(frozen=True)
class TestCase:
    name: str
    stdin: str
    expected: str


@dataclass(frozen=True)
class RunConfig:
    backend: str = "docker"
    timeout: float = 3.0
    repeats: int = 3
    warmups: int = 1
    memory_mb: int = 256
    output_bytes: int = 1_048_576
    image: str = "python:3.11-slim"

    def __post_init__(self):
        if self.backend not in {"docker", "local"}:
            raise ValueError("backend must be docker or local")
        if self.timeout <= 0 or self.repeats < 1 or self.warmups < 0:
            raise ValueError("Invalid timeout or repeat count")
        if self.memory_mb < 32 or self.output_bytes < 1024:
            raise ValueError("Resource limits are too small")


@dataclass(frozen=True)
class ExecutionResult:
    verdict: str
    runtime_s: float | None
    passed: int
    total: int
    detail: str = ""

    @property
    def accepted(self):
        return self.verdict == "accepted"


def discover_tests(roots, problem_id):
    problem_id = str(problem_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", problem_id):
        raise ValueError("Invalid problem ID")
    cases = []
    for index, root in enumerate(roots):
        directory = Path(root).expanduser() / problem_id
        for source in sorted(directory.glob("input.*.txt")):
            expected = source.with_name("output." + source.name[len("input.") :])
            if not expected.is_file():
                raise ValueError(f"Missing expected output: {expected}")
            cases.append(
                TestCase(
                    f"{index}/{source.name}",
                    source.read_text(encoding="utf-8"),
                    expected.read_text(encoding="utf-8"),
                )
            )
    return cases


class Runner:
    def __init__(self, config=None):
        self.config = config or RunConfig()

    def fingerprint(self):
        result = {
            **asdict(self.config),
            "platform": platform.system(),
            "machine": platform.machine(),
        }
        if self.config.backend == "local":
            result["python"] = platform.python_version()
        else:
            check = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", self.config.image],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            result["image_id"] = check.stdout.strip()
        return result

    def run(self, code, cases):
        if not cases:
            return ExecutionResult("no_tests", None, 0, 0, "No test cases were found")
        try:
            ast.parse(code)
        except (SyntaxError, ValueError) as error:
            return ExecutionResult("syntax_error", None, 0, len(cases), str(error))
        if self.config.backend == "docker" and not shutil.which("docker"):
            raise RuntimeError(
                "Docker is required. Use --allow-local-execution only for trusted code."
            )
        container_name = "codeopt-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="codeopt-") as directory:
            root = Path(directory)
            root.chmod(0o755)
            (root / "program.py").write_text(code, encoding="utf-8")
            shutil.copyfile(Path(__file__).with_name("_worker.py"), root / "worker.py")
            write_json(
                root / "request.json",
                {
                    "config": asdict(self.config),
                    "cases": [asdict(case) for case in cases],
                },
            )
            for file in root.iterdir():
                file.chmod(0o644)
            if self.config.backend == "local":
                command = [sys.executable, "-I", "-B", str(root / "worker.py"), str(root)]
            else:
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--name",
                    container_name,
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=64",
                    f"--memory={self.config.memory_mb + 64}m",
                    "--cpus=1",
                    "--user=65534:65534",
                    "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
                    "-w",
                    "/tmp",
                    "-v",
                    f"{root}:/work:ro",
                    self.config.image,
                    "python",
                    "-I",
                    "-B",
                    "/work/worker.py",
                    "/work",
                ]
            deadline = (
                len(cases) * (self.config.warmups + self.config.repeats) * self.config.timeout + 30
            )
            try:
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=deadline, check=False
                )
            except subprocess.TimeoutExpired as error:
                if self.config.backend == "docker":
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                raise RuntimeError("Execution worker exceeded its overall deadline") from error
            if process.returncode:
                raise RuntimeError(f"Execution worker failed: {process.stderr[-1200:]}")
            return ExecutionResult(**json.loads(process.stdout))


def test_suite_digest(test_map):
    return digest(
        {pid: [asdict(case) for case in cases] for pid, cases in sorted(test_map.items())}
    )
