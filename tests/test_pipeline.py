"""Small tests for the data pipeline and the evaluation module.

Run from the repo root:   python -m pytest
They use tiny in-memory examples, so no dataset download is needed.
"""

import sys
from pathlib import Path

# Make `src/` importable when pytest is started from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from evaluate import evaluate
from preprocess import clean_text, read_tsv, split_pairs, write_tsv
from vocab import Vocab


def test_clean_text_replaces_urls_and_whitespace():
    assert clean_text("Look  at\tthis http://x.com/a?b=1 ...") == "look at this <url>"


def test_clean_text_keep_case():
    assert clean_text("Hello World", lowercase=False) == "Hello World"


def test_split_is_deterministic_and_disjoint():
    pairs = [(f"s{i}", f"t{i}") for i in range(100)]
    a = split_pairs(pairs, seed=1)
    b = split_pairs(pairs, seed=1)
    assert a == b                                   # same seed -> same split
    assert [len(x) for x in a] == [80, 10, 10]      # 80/10/10
    all_items = a[0] + a[1] + a[2]
    assert len(set(all_items)) == 100               # no pair appears twice -> no leakage


def test_tsv_round_trip(tmp_path):
    pairs = [("mera naam", "my name"), ("kya hua", "what happened")]
    write_tsv(pairs, tmp_path / "x.tsv")
    assert read_tsv(tmp_path / "x.tsv") == pairs


def test_vocab_encode_decode():
    v = Vocab(["kya hua kya", "kya baat"], min_freq=1)
    ids = v.encode("kya zzz")                       # "zzz" was never seen
    assert ids[0] == v.bos_id and ids[-1] == v.eos_id
    assert v.decode(ids) == "kya"                   # <unk> is dropped when decoding


def test_evaluate_perfect_and_bad():
    refs = ["i am going home", "what is your name"]
    perfect = evaluate(refs, refs)
    assert perfect["bleu"] == 100.0 and perfect["chrf"] == 100.0
    bad = evaluate(["zzz qqq", "xxx"], refs)
    assert bad["bleu"] < 5


def test_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a", "b"])
