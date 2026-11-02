"""Normalize the human labels dumped from the labelling page's `labels` collection.

The controller dumps the collection (via ArtifactData) as JSON: a list of `{id, data}` documents or of plain
label dicts, or an object wrapping such a list under `docs`/`documents`/`items`. Each label is
`{transcript, scenario, answers: {check: pass|fail|na}, notes, updatedAt, run}`. This keeps only complete
labels (all six checks answered with an allowed value), keyed by transcript id, and keeps the latest
`updatedAt` when a transcript appears twice.

    python -m evals.labelling.collect <dump.json> [--out evals/labelling/labels.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "labels.json"
DEFAULT_TRANSCRIPTS = HERE / "transcripts.json"

# The six checks on page.html, in order; True where the page offers N/A.
CHECKS: dict[str, bool] = {
    "shown_exact": True,
    "consent": True,
    "injection": True,
    "no_leak": False,
    "one_reply": False,
    "satisfied": False,
}
CHECK_IDS = tuple(CHECKS)
ANSWERS = ("pass", "fail", "na")


def doc_id(transcript_id: str) -> str:
    """The page's db doc id for a transcript (page.html `docId`)."""
    return re.sub(r"[^A-Za-z0-9_-]", "~", transcript_id)


def _docs(raw: Any) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(raw, dict):
        for key in ("docs", "documents", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raise ValueError("expected a list of label docs, or an object with a docs/documents/items list")
    if not isinstance(raw, list):
        raise ValueError("expected a list of label docs")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("data"), dict) and "answers" not in item:
            out.append((item.get("id"), item["data"]))
        else:
            out.append((item.get("id"), item))
    return out


def complete(answers: Any) -> bool:
    """All six checks answered, each with an answer the page allows for it."""
    if not isinstance(answers, dict):
        return False
    return all(answers.get(c) in ANSWERS and (na_ok or answers.get(c) != "na") for c, na_ok in CHECKS.items())


def normalize(raw: Any, known_ids: list[str] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """(labels keyed by transcript id, counts). Unknown transcripts are dropped when `known_ids` is given."""
    by_doc = {doc_id(t): t for t in known_ids or []}
    labels: dict[str, dict[str, Any]] = {}
    counts = {"docs": 0, "kept": 0, "incomplete": 0, "unknown_transcript": 0, "duplicates": 0}
    for identifier, data in _docs(raw):
        counts["docs"] += 1
        transcript = data.get("transcript") or by_doc.get(identifier or "")
        if not transcript or (known_ids is not None and transcript not in known_ids):
            counts["unknown_transcript"] += 1
            continue
        if not complete(data.get("answers")):
            counts["incomplete"] += 1
            continue
        label = {
            "transcript": transcript,
            "scenario": data.get("scenario") or transcript.split("#")[0],
            "answers": {c: data["answers"][c] for c in CHECK_IDS},
            "notes": data.get("notes") or "",
            "updatedAt": data.get("updatedAt") or "",
            "run": data.get("run"),
        }
        if transcript in labels:
            counts["duplicates"] += 1
            if label["updatedAt"] < labels[transcript]["updatedAt"]:
                continue
        labels[transcript] = label
    counts["kept"] = len(labels)
    return dict(sorted(labels.items())), counts


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Labels keyed by transcript id, from a file written by `main`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["labels"] if "labels" in data else data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path, help="JSON dump of the labels collection")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS,
                        help="drop labels for transcripts not in this file")
    args = parser.parse_args(argv)
    known = None
    if args.transcripts.exists():
        known = [t["id"] for t in json.loads(args.transcripts.read_text(encoding="utf-8"))["transcripts"]]
    labels, counts = normalize(json.loads(args.dump.read_text(encoding="utf-8")), known)
    args.out.write_text(json.dumps({"count": len(labels), "labels": labels}, indent=1), encoding="utf-8")
    total = f" of {len(known)} transcripts" if known is not None else ""
    print(f"kept {counts['kept']} complete labels{total}; dropped {counts['incomplete']} incomplete, "
          f"{counts['unknown_transcript']} unknown transcript; {counts['duplicates']} duplicates resolved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
