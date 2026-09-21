"""Execute all notebooks under notebooks/ and report failures.

Run with: uv run python scripts/run_all_notebooks.py

By default this runs in quick mode (NNZERO_QUICK_RUN=1) so training loops use
only a small number of steps. To run the full training loops, unset the
environment variable:

    NNZERO_QUICK_RUN=0 uv run python scripts/run_all_notebooks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
TIMEOUT_SECONDS = 600


def run_notebook(path: Path) -> None:
    print(f"Executing {path.name} ...", flush=True)
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Run with the repo root as the working directory so that relative paths
    # like "data/names.txt" resolve correctly.
    ep = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": path.parent.parent}})

    print(f"  {path.name}: OK")


def main() -> int:
    # Quick-run mode keeps CI execution times reasonable.
    if "NNZERO_QUICK_RUN" not in os.environ:
        os.environ["NNZERO_QUICK_RUN"] = "1"
        print("NNZERO_QUICK_RUN=1 set automatically. Use NNZERO_QUICK_RUN=0 for full training.")

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOK_DIR}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for nb_path in notebooks:
        try:
            run_notebook(nb_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{nb_path.name}: {exc}")
            print(f"  {nb_path.name}: FAILED ({exc})")

    print()
    if failures:
        print("Failures:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(f"All {len(notebooks)} notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
