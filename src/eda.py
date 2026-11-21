"""Quick exploratory statistics for the processed PHINC splits.

Usage (after download_data.py and preprocess.py):
    python src/eda.py

Prints and saves to results/eda.txt: split sizes, sentence-length statistics,
a rough code-mixing indicator and a few example pairs. Paste the numbers into
the README / final report.
"""

import random
import statistics
from pathlib import Path

from preprocess import read_tsv  # run as a script from the repo root: python src/eda.py

SPLIT_DIR = Path("data/processed")


def overlap_ratio(src: str, tgt: str) -> float:
    """Share of source words that also appear (as-is) in the English sentence.

    This is only a ROUGH proxy for code-mixing: English words written in
    Roman script (e.g. "movie", "party") appear unchanged in both sides, while
    Hindi words written in Roman script (e.g. "kaha", "hai") do not.
    """
    s, t = src.split(), set(tgt.split())
    return sum(w in t for w in s) / len(s) if s else 0.0


def main() -> None:
    lines = []
    for name in ("train", "val", "test"):
        pairs = read_tsv(SPLIT_DIR / f"{name}.tsv")
        src_len = [len(s.split()) for s, _ in pairs]
        tgt_len = [len(t.split()) for _, t in pairs]
        ov = [overlap_ratio(s, t) for s, t in pairs]
        lines.append(
            f"{name:5s} n={len(pairs):5d}  "
            f"src_len mean={statistics.mean(src_len):.1f} max={max(src_len)}  "
            f"tgt_len mean={statistics.mean(tgt_len):.1f} max={max(tgt_len)}  "
            f"word-overlap mean={statistics.mean(ov):.2f}"
        )

    lines.append("\nExample training pairs:")
    train = read_tsv(SPLIT_DIR / "train.tsv")
    for s, t in random.Random(0).sample(train, 5):
        lines.append(f"  HI-EN: {s}\n  EN   : {t}\n")

    report = "\n".join(lines)
    print(report)
    Path("results").mkdir(exist_ok=True)
    Path("results/eda.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
