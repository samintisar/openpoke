"""Checksum of eval files an experiment loop must never edit.

Update only by deliberate human commit:  python -m evals.integrity --update
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_FILE = "evals/PROTECTED.sha256"
PROTECTED = (
    "evals/cases.py",
    "evals/grading.py",
    "evals/metrics.py",
    "evals/layers/retrieval.py",
    "evals/layers/routing.py",
    "evals/worlds/model.py",
    "evals/worlds/generator.py",
    "evals/worlds/handwritten.py",
    "evals/worlds/handwritten_cases.yaml",
    "evals/suites/regression.txt",
    "evals/run.py",
    "evals/llm.py",
    "evals/integrity.py",
    "evals/systems/baseline.py",
    "evals/systems/frozen/baseline_system_prompt.md",
    "evals/systems/frozen/baseline_tools.json",
    "evals/worlds/lived/model.py",
    "evals/worlds/lived/plan.py",
    "evals/worlds/lived/writer.py",
    "evals/worlds/lived/grow.py",
    "evals/worlds/lived/probes.py",
    "evals/worlds/lived/store.py",
    "evals/worlds/lived/world-s1.json",
)
_MISSING = b"<missing>"


def compute_checksum(root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for rel in PROTECTED:
        digest.update(rel.encode("utf-8") + b"\0")
        path = root / rel
        data = path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else _MISSING  # e.g. world not generated yet
        digest.update(data + b"\0")
    return digest.hexdigest()


def expected_checksum(root: Path = REPO_ROOT) -> str | None:
    path = root / EXPECTED_FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def expected_file_committed(root: Path = REPO_ROOT) -> bool:
    """False when PROTECTED.sha256 differs from HEAD; True if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", EXPECTED_FILE], cwd=root, capture_output=True
        )
    except OSError:
        return True
    return result.returncode == 0


def write_expected(root: Path = REPO_ROOT) -> str:
    value = compute_checksum(root)
    path = root / EXPECTED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the current checksum as expected")
    args = parser.parse_args(argv)
    if args.update:
        print(write_expected())
        return 0
    current, expected = compute_checksum(), expected_checksum()
    print(f"current  {current}\nexpected {expected}")
    return 0 if current == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
