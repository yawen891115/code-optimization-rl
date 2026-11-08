"""Standalone worker copied into an isolated execution directory."""

import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def execute(program, stdin, config):
    def limits():
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cpu = max(1, math.ceil(config["timeout"]) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (config["output_bytes"], config["output_bytes"]))
        memory = config["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

    with (
        tempfile.TemporaryDirectory() as working,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        start = time.perf_counter()
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", str(program)],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=working,
            env={"PATH": os.defpath, "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"},
            start_new_session=True,
            preexec_fn=limits if os.name == "posix" else None,
        )
        verdict = "accepted"
        try:
            child.communicate(stdin.encode(), timeout=config["timeout"])
        except subprocess.TimeoutExpired:
            verdict = "timeout"
        finally:
            if os.name == "posix":
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.kill()
            child.wait()
        elapsed = time.perf_counter() - start
        stdout.seek(0)
        output = stdout.read(config["output_bytes"] + 1).decode("utf-8", errors="replace")
        stderr.seek(0)
        error = stderr.read(1000).decode("utf-8", errors="replace")
        if verdict == "accepted" and child.returncode:
            verdict = "runtime_error"
        return verdict, elapsed, output, error


def main(directory):
    root = Path(directory)
    request = json.loads((root / "request.json").read_text())
    config, cases = request["config"], request["cases"]
    measurements = []
    result = {
        "verdict": "accepted",
        "runtime_s": None,
        "passed": 0,
        "total": len(cases),
        "detail": "",
    }
    for case in cases:
        timings = []
        for repeat in range(config["warmups"] + config["repeats"]):
            verdict, elapsed, output, error = execute(root / "program.py", case["stdin"], config)
            if verdict == "accepted" and output.strip() != case["expected"].strip():
                verdict = "wrong_answer"
            if verdict != "accepted":
                result.update(verdict=verdict, detail=f"{case['name']}: {error[:600]}")
                print(json.dumps(result))
                return
            if repeat >= config["warmups"]:
                timings.append(elapsed)
        measurements.append(statistics.median(timings))
        result["passed"] += 1
    if not measurements:
        result["verdict"] = "no_tests"
    else:
        result["runtime_s"] = statistics.mean(measurements)
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])
