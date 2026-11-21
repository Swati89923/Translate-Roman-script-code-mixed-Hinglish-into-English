"""Download the PHINC corpus (Hinglish -> English parallel sentences).

PHINC: Srivastava & Singh, "PHINC: A Parallel Hinglish Social Media Code-Mixed
Corpus for Machine Translation", W-NUT 2020 (https://arxiv.org/abs/2004.09447).
The original release lives on Zenodo: https://doi.org/10.5281/zenodo.3605597

Usage:
    python src/download_data.py

The file is saved to data/raw/phinc.csv. The data/ folder is gitignored, so
every team member runs this script once on their own machine / Colab.
"""

import urllib.request
from pathlib import Path

# Direct link to the single CSV file of the Zenodo record.
URL = (
    "https://zenodo.org/records/3605597/files/"
    "English-Hindi%20code-mixed%20parallel%20corpus.csv?download=1"
)
OUT_PATH = Path("data/raw/phinc.csv")


def download(url: str = URL, out_path: Path = OUT_PATH) -> Path:
    """Download `url` to `out_path` unless the file already exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)  # create data/raw/ if needed
    if out_path.exists():
        print(f"{out_path} already exists, skipping download.")
        return out_path
    print(f"Downloading PHINC to {out_path} ...")
    # Download to a temporary .part file and rename only when it is complete. If the
    # download is interrupted, out_path never exists, so the next run retries.
    part_path = out_path.with_name(out_path.name + ".part")
    urllib.request.urlretrieve(url, part_path)
    part_path.replace(out_path)
    print(f"Done ({out_path.stat().st_size / 1e6:.1f} MB).")
    return out_path


if __name__ == "__main__":
    download()
