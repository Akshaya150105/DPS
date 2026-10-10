"""Attack evaluation: computes accuracy metrics against ground-truth keyword map."""

from __future__ import annotations

import csv
import pathlib
import time
from typing import Any


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy(
    predictions: dict[int, str | None],
    ground_truth: dict[int, str],
) -> float:
    """Compute the fraction of queries predicted correctly.

    A prediction of ``None`` always counts as wrong.  Query ids present in
    *ground_truth* but missing from *predictions* are also counted as wrong.

    Args:
        predictions:  Dict mapping query_id -> predicted keyword (or ``None``).
        ground_truth: Dict mapping query_id -> true keyword string.

    Returns:
        Float in [0.0, 1.0].  Returns 1.0 when *ground_truth* is empty
        (vacuously correct).
    """
    if not ground_truth:
        return 1.0

    correct = 0
    total   = len(ground_truth)

    for qid, true_kw in ground_truth.items():
        predicted = predictions.get(qid)   # None if missing
        if predicted is not None and predicted == true_kw:
            correct += 1

    return correct / total


# ---------------------------------------------------------------------------
# CSV result logger
# ---------------------------------------------------------------------------

# Fixed column order for the results CSV.
_COLUMNS = [
    "dataset",
    "keyword_count",
    "attack",
    "equality",
    "tau",
    "lambda",
    "gamma",
    "alpha",
    "eta",
    "noise",
    "correct",
    "total",
    "accuracy",
    "runtime_s",
    "seed",
]


def log_result(row_dict: dict[str, Any], path: str | pathlib.Path) -> None:
    """Append a single result row to a CSV file at *path*.

    If the file does not yet exist (or is empty) a header row is written first.
    Columns are always written in the fixed order defined by :data:`_COLUMNS`.
    Any key in *row_dict* that is not in :data:`_COLUMNS` is silently ignored;
    any column in :data:`_COLUMNS` absent from *row_dict* is written as an
    empty string.

    The parent directory is created automatically if it does not exist.

    Args:
        row_dict: Mapping of column name -> value for this experiment run.
                  Recognised keys::

                      dataset, keyword_count, attack, equality, tau, lambda,
                      gamma, alpha, eta, noise, correct, total, accuracy,
                      runtime_s, seed

        path:     Filesystem path to the target CSV file.  A ``str`` or
                  :class:`pathlib.Path` is accepted.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Write header only when the file is new or empty.
    write_header = not p.exists() or p.stat().st_size == 0

    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=_COLUMNS,
            extrasaction="ignore",    # silently drop unknown keys
        )
        if write_header:
            writer.writeheader()

        # Fill missing columns with empty string so DictWriter does not raise.
        row = {col: row_dict.get(col, "") for col in _COLUMNS}
        writer.writerow(row)
