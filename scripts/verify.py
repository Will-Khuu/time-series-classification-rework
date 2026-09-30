"""Run every check on the codebase and write reports/verification.md.

Checks: the test suite, dataset integrity, the duplicate scan, and mutation checks.
A mutation check breaks the source on purpose and confirms the tests catch it, which
shows the tests can fail. Source files are restored after each mutation.

Usage: python scripts/verify.py
"""

import hashlib
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "verification.md"
sys.path.insert(0, str(ROOT / "data"))

import download  # noqa: E402
from check_independence import duplicate_pairs  # noqa: E402

from arwsn.loading import COPIES, DATA_DIR  # noqa: E402


@dataclass
class Mutation:
    file: str
    before: str
    after: str
    breaks: str
    tests: str


MUTATIONS = [
    Mutation("src/arwsn/loading.py", '    "lying/dataset13": "lying/dataset7",\n', "",
             "forget one duplicate file", "tests/test_loading.py"),
    Mutation("src/arwsn/loading.py", 'sep=r"[,\\s]+"', 'sep=","',
             "parse commas only, losing the space-separated file", "tests/test_loading.py"),
    Mutation("src/arwsn/loading.py", 'n_test = 2 if folder.startswith("bending") else 3',
             'n_test = 3 if folder.startswith("bending") else 3',
             "change the notebook split sizes", "tests/test_loading.py"),
    Mutation("src/arwsn/features.py", "transpose(2, 0, 1)", "transpose(0, 2, 1)",
             "order features segment-first, as the notebook's names wrongly assumed", "tests/test_features.py"),
    Mutation("src/arwsn/features.py", "std(axis=0, ddof=1)", "std(axis=0, ddof=0)",
             "use population std instead of sample std", "tests/test_features.py"),
    Mutation("src/arwsn/features.py", "[0.25, 0.5, 0.75]", "[0.2, 0.5, 0.75]",
             "compute the wrong first quartile", "tests/test_features.py"),
]


def run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)


def pytest_cases(*targets: str) -> tuple[list[tuple[str, str, str]], str]:
    """(test file stem, test name, outcome) per test case, plus pytest's output tail if it crashed."""
    with TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        result = run(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *targets)
        if not xml.exists():
            return [], (result.stdout + result.stderr)[-500:]
        cases = []
        for case in ET.parse(xml).getroot().iter("testcase"):
            failed = case.find("failure") is not None or case.find("error") is not None
            outcome = "fail" if failed else "skip" if case.find("skipped") is not None else "pass"
            cases.append((case.get("classname").rpartition(".")[2], case.get("name"), outcome))
    return cases, ""


def code_fingerprint() -> str:
    files = sorted(p for d in ["src", "tests", "data", "scripts"] for p in (ROOT / d).rglob("*.py"))
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()[:16]


def environment() -> list[str]:
    sha = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
    libs = ", ".join(f"{p} {version(p)}" for p in ["numpy", "pandas", "scipy", "scikit-learn", "pytest"])
    return [
        f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        f"- Code fingerprint: `{code_fingerprint()}` (SHA-256 of every `.py` file in `src`, `tests`, `data`, `scripts`)",
        f"- Checked on top of commit `{sha}`; the fingerprint identifies the exact code",
        f"- Python {platform.python_version()} on {platform.system()} {platform.machine()}",
        f"- {libs}",
    ]


def test_suite() -> tuple[bool, list[str]]:
    cases, crash = pytest_cases()
    if crash:
        return False, ["pytest crashed before producing results:", "", "```", crash, "```"]
    by_file: dict[str, list[int]] = {}
    for stem, _, outcome in cases:
        by_file.setdefault(stem, [0, 0, 0])[["pass", "fail", "skip"].index(outcome)] += 1
    lines = ["| Test file | Passed | Failed | Skipped |", "|---|---|---|---|"]
    lines += [f"| `tests/{stem}.py` | {p} | {f} | {k} |" for stem, (p, f, k) in sorted(by_file.items())]
    lines += [f"- FAILED `tests/{stem}.py::{name}`" for stem, name, outcome in cases if outcome == "fail"]
    total = [sum(c[i] for c in by_file.values()) for i in range(3)]
    ok = total[1] == 0 and total[2] == 0 and total[0] > 0
    return ok, [f"{total[0]} passed, {total[1]} failed, {total[2]} skipped.", "", *lines]


def dataset_integrity() -> tuple[bool, list[str]]:
    found = download.problems(DATA_DIR)
    summary = f"{sum(download.EXPECTED_FILES.values())} files across {len(download.EXPECTED_FILES)} folders"
    return not found, [f"`data/download.py` file and row counts: {'all match' if not found else 'problems'} ({summary})."] + [
        f"- {p}" for p in found
    ]


def duplicates() -> tuple[bool, list[str]]:
    pairs = duplicate_pairs(DATA_DIR)
    copies = {b if b in COPIES else a for a, b, _ in pairs}
    ok = copies == set(COPIES)
    lines = [f"`data/check_independence.py` found {len(pairs)} duplicate pairs covering {len(copies)} copied files.",
             f"They {'match' if ok else 'do NOT match'} the `COPIES` constant in `src/arwsn/loading.py`."]
    return ok, lines


def mutations() -> tuple[bool, list[str]]:
    lines = ["| Deliberate bug | File | Caught by |", "|---|---|---|"]
    all_caught = True
    for m in MUTATIONS:
        path = ROOT / m.file
        original = path.read_text()
        if original.count(m.before) != 1:
            lines.append(f"| {m.breaks} | `{m.file}` | SETUP ERROR: target text not found exactly once |")
            all_caught = False
            continue
        try:
            path.write_text(original.replace(m.before, m.after))
            cases, crash = pytest_cases(m.tests)
        finally:
            path.write_text(original)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original.encode()).digest()
        caught_by = [name for stem, name, outcome in cases if outcome == "fail" and m.tests.endswith(f"{stem}.py")]
        all_caught &= bool(caught_by) and not crash
        verdict = f"`{caught_by[0]}`" + (f" and {len(caught_by) - 1} more" if len(caught_by) > 1 else "")
        lines.append(f"| {m.breaks} | `{m.file}` | {verdict if caught_by and not crash else 'NOT CAUGHT'} |")
    return all_caught, lines


def main() -> None:
    sections = [
        ("Test suite", test_suite),
        ("Dataset integrity", dataset_integrity),
        ("Duplicate recordings", duplicates),
        ("Mutation checks", mutations),
    ]
    body, verdicts = [], []
    for title, check in sections:
        ok, lines = check()
        verdicts.append((title, ok))
        body += [f"## {title}: {'PASS' if ok else 'FAIL'}", "", *lines, ""]
    overall = all(ok for _, ok in verdicts)
    header = [
        "# Verification report",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}.** Regenerate with `python scripts/verify.py`.",
        "",
        *environment(),
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(header + body))
    for title, ok in verdicts:
        print(f"{'PASS' if ok else 'FAIL'}  {title}")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
