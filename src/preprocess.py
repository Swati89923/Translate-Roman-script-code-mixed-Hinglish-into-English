"""Clean the raw PHINC corpus and write fixed train / val / test splits.

Everyone on the team must train and evaluate on the SAME splits, otherwise the
BLEU / chrF numbers of the different models cannot be compared. This script is
deterministic: the same raw file and the same seed always give the same files.

Usage:
    python src/preprocess.py                 # defaults: seed 42, 80/10/10 split
    python src/preprocess.py --seed 7

Output (tab-separated, one pair per line, header "src<TAB>tgt"):
    data/processed/train.tsv
    data/processed/val.tsv
    data/processed/test.tsv
"""

import argparse
import csv
import random
import re
from pathlib import Path

RAW_PATH = Path("data/raw/phinc.csv")
OUT_DIR = Path("data/processed")

# Column names used in the raw PHINC csv.
SRC_COL = "Sentence"            # Roman-script Hinglish (the model input)
TGT_COL = "English_Translation"  # English reference translation

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Trailing "..." or the unicode ellipsis that tweets often end with after a cut-off URL.
TRAILING_DOTS_RE = re.compile(r"(\.{2,}|…)\s*$")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalise one sentence.

    Policy (documented here so every model uses the same one):
      * URLs are replaced by the token "<url>" (the exact link carries no
        translation signal and would only blow up the vocabulary).
      * Trailing ellipsis characters are removed.
      * Runs of whitespace (incl. tabs / newlines) collapse to a single space,
        which also keeps the TSV files well-formed.
      * Text is lowercased by default. Mentions (@user), hashtags and emojis
        are kept as they are, because the English reference keeps them too.
    """
    text = URL_RE.sub("<url>", text)
    text = TRAILING_DOTS_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text.lower() if lowercase else text


def load_pairs(raw_path: Path = RAW_PATH, lowercase: bool = True):
    """Read the raw csv and return a list of cleaned (src, tgt) pairs.

    Rows with an empty side after cleaning are dropped, and exact duplicate
    pairs are removed (keeping the first one) so that the same sentence cannot
    end up in both train and test.
    """
    pairs, seen = [], set()
    with open(raw_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            src = clean_text(row.get(SRC_COL) or "", lowercase)
            tgt = clean_text(row.get(TGT_COL) or "", lowercase)
            if not src or not tgt or (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            pairs.append((src, tgt))
    return pairs


def split_pairs(pairs, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.1):
    """Shuffle with a fixed seed and cut into train / val / test lists."""
    pairs = list(pairs)  # copy, so the caller's list is not reordered
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_frac)
    n_test = int(len(pairs) * test_frac)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def write_tsv(pairs, path: Path) -> None:
    """Write pairs as a 2-column TSV with a header line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("src\ttgt\n")
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")


def read_tsv(path):
    """Read a split file written by write_tsv; returns a list of (src, tgt).

    Other modules (LSTM, Transformer, fine-tuning) should use this function so
    that everybody reads the splits the same way.
    """
    with open(path, encoding="utf-8") as f:
        next(f)  # skip the header
        return [tuple(line.rstrip("\n").split("\t")) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-case", action="store_true", help="do not lowercase")
    args = parser.parse_args()

    pairs = load_pairs(args.raw, lowercase=not args.keep_case)
    train, val, test = split_pairs(pairs, seed=args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        write_tsv(part, args.out / f"{name}.tsv")
        print(f"{name}: {len(part)} pairs -> {args.out / (name + '.tsv')}")


if __name__ == "__main__":
    main()
