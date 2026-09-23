"""Load fixed contrast experiment testset (SPEC §15.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TESTSET_PATH = Path("data/experiment/testset.json")


class TestsetError(ValueError):
    """Raised when testset.json is missing or malformed."""


def load_testset(path: str | Path = DEFAULT_TESTSET_PATH) -> list[dict[str, Any]]:
    """Return list of cases with id/query/positives/negatives."""
    p = Path(path)
    if not p.is_file():
        raise TestsetError(f"testset missing: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        cases = raw.get("cases")
    elif isinstance(raw, list):
        cases = raw
    else:
        raise TestsetError("testset must be a list or an object with 'cases'")

    if not isinstance(cases, list) or not cases:
        raise TestsetError("testset cases must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(cases):
        if not isinstance(row, dict):
            raise TestsetError(f"cases[{i}] must be an object")
        for key in ("id", "query", "positives", "negatives"):
            if key not in row:
                raise TestsetError(f"cases[{i}] missing field: {key}")
        positives = row["positives"]
        negatives = row["negatives"]
        if not isinstance(positives, list) or not positives:
            raise TestsetError(f"cases[{i}].positives must be a non-empty list")
        if not isinstance(negatives, list) or not negatives:
            raise TestsetError(f"cases[{i}].negatives must be a non-empty list")
        if not all(isinstance(x, str) and x.strip() for x in positives):
            raise TestsetError(f"cases[{i}].positives must be non-empty strings")
        if not all(isinstance(x, str) and x.strip() for x in negatives):
            raise TestsetError(f"cases[{i}].negatives must be non-empty strings")
        normalized.append(
            {
                "id": str(row["id"]),
                "query": str(row["query"]).strip(),
                "positives": [str(x).strip() for x in positives],
                "negatives": [str(x).strip() for x in negatives],
            }
        )
    return normalized
