"""Download datasets used by the notebooks.

Run with: uv run python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).parent.parent / "data"
DATASETS = {
    "names.txt": "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
}


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, dest.open("wb") as f:  # noqa: S310
        f.write(response.read())
    print(f"  saved {dest.stat().st_size:,} bytes")


def main() -> None:
    for filename, url in DATASETS.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"{dest} already exists, skipping")
            continue
        download(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
