"""Guard SDK workers against package auto-installers and telemetry side effects."""
import sys
from pathlib import Path


class OfflineViolation(RuntimeError):
    pass


def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "os.system", "os.exec", "os.posix_spawn", "os.spawn"}:
        raise OfflineViolation(f"offline SDK worker blocked {event}")
    if event == "subprocess.Popen":
        raise OfflineViolation("offline SDK worker blocked subprocess (including bare pip)")
    if event == "import" and str(args[0]).startswith("MNN.tools"):
        raise OfflineViolation("MNN.tools imports telemetry; use native _tools directly")


def install():
    if sys.prefix == sys.base_prefix:
        raise OfflineViolation("run only with an isolated virtual environment")
    sys.addaudithook(audit)


def isolated_environment(workspace):
    """Allow the one intended project sandbox, without using a pre-existing venv."""
    expected = (Path(workspace) / "runs/phase2-edge/.venv").resolve()
    if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
        raise OfflineViolation("use runs/phase2-edge/.venv, not the project or system interpreter")


def self_test():
    import json
    import os
    import socket
    import subprocess

    install()
    blocked = []
    actions = {
        "bare_pip": lambda: subprocess.run(["pip", "--version"]),
        "network": lambda: socket.create_connection(("example.com", 443)),
        "shell": lambda: os.system("pip --version"),
        "telemetry_import": lambda: __import__("MNN.tools.mnnconvert"),
    }
    for name, action in actions.items():
        try:
            action()
        except OfflineViolation:
            blocked.append(name)
        else:
            raise AssertionError(f"guard failed: {name}")
    print(json.dumps({"blocked": blocked, "status": "PASS"}))


if __name__ == "__main__":
    self_test()
