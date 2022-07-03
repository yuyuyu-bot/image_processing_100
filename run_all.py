#!/bin/python3

import re
import subprocess
import typing

IMPL = ["cpp", "neon", "cuda"]


def check(impl: str, lines: str) -> bool:
    for line in lines:
        match = re.search(rf".*{impl}.*", line)
        if (match is not None):
            return True
    return False


def main() -> None:
    dirs = subprocess.check_output(
        "find -maxdepth 1 -type d | "
        "grep -x \"\./[0-9]*\" | "
        "sort --version-sort", shell=True
    ).decode("utf-8").split("\n")
    dirs.remove("")

    missing_implementations: typing.Dict[str, typing.List[str]] = {}

    for d in dirs:
        print(d)
        stdout = subprocess.check_output("make run", shell=True, cwd=d)
        lines = stdout.decode("utf-8").split("\n")
        lines.remove("")

        for line in lines:
            print(line)

        missing_implementations[d] = []
        for impl in IMPL:
            if (not check(impl, lines)):
                missing_implementations[d].append(impl)

    for d in missing_implementations.keys():
        if (len(missing_implementations[d]) > 0):
            for impl in missing_implementations[d]:
                print(f"{d} does not contain {impl} implementation.")


if __name__ == "__main__":
    main()
