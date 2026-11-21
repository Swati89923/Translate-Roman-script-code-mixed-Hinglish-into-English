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
from preprocess import clean_text, has_letter, is_truncated, load_pairs, read_tsv, split_pairs, write_tsv
from vocab import Vocab


def test_clean_text_replaces_urls_and_whitespace():
    assert clean_text("Look  at\tthis http://x.com/a?b=1 ...") == "look at this <url>"


def test_clean_text_keep_case():
    assert clean_text("Hello World", lowercase=False) == "Hello World"


def test_clean_text_decodes_html_codes():
    assert clean_text("rock &amp; roll &lt;3") == "rock & roll <3"


def test_has_letter():
    assert not has_letter(")") and not has_letter("$") and not has_letter("-) 3:)")
    assert has_letter("ok :)") and has_letter("नमस्ते")


def test_load_pairs_drops_junk_translations(tmp_path):
    raw = tmp_path / "raw.csv"
    lines = [
        "Sentence,English_Translation",
        "kya hua bhai,what happened brother",
        "haha,)",                          # English side is only a symbol -> dropped
        "lol,$",                           # dropped
        "wah,-)",                          # dropped
        "tom &amp; jerry,tom &amp; jerry",  # kept, html decoded
        "kya hua bhai,what happened brother",  # exact duplicate -> dropped
    ]
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert load_pairs(raw) == [
        ("kya hua bhai", "what happened brother"),
        ("tom & jerry", "tom & jerry"),
    ]


def test_is_truncated():
    assert is_truncated("are you ready to shop the new collection from h&")
    assert is_truncated("@someuser &  ")
    assert not is_truncated("shopping at h&m today")  # "&" inside a sentence is fine


def test_load_pairs_drops_cut_tweets(tmp_path):
    lines = [
        "Sentence,English_Translation",
        "shop from h&amp;,m tomorrow from h&amp;",   # raw tweet cut at "&amp;" -> dropped
        "kal h&amp;m jaunga,i will go to h&amp;m tomorrow",  # "&" inside -> kept
    ]
    raw = tmp_path / "raw.csv"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert load_pairs(raw) == [("kal h&m jaunga", "i will go to h&m tomorrow")]


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
    assert v.decode(ids) == "kya <unk>"             # <unk> stays visible for error analysis


def test_sentencepiece_keeps_url_token(tmp_path):
    pytest.importorskip("sentencepiece")
    import sentencepiece as spm
    from vocab import train_sentencepiece

    sentences = [f"check this <url> number {i} kya hua bhai" for i in range(200)]
    model = train_sentencepiece(sentences, str(tmp_path / "spm"), vocab_size=60)
    sp = spm.SentencePieceProcessor(model_file=model)
    assert sp.encode("a <url> b", out_type=str).count("<url>") == 1  # one token, not split


def test_download_uses_part_file(tmp_path, monkeypatch):
    import download_data

    def fake_retrieve(url, dest):  # pretend the download finished
        assert str(dest).endswith(".part")          # written to .part first
        Path(dest).write_text("data", encoding="utf-8")

    monkeypatch.setattr(download_data.urllib.request, "urlretrieve", fake_retrieve)
    out = tmp_path / "phinc.csv"
    download_data.download("http://example.invalid", out)
    assert out.exists() and not (tmp_path / "phinc.csv.part").exists()


def test_download_failure_leaves_no_final_file(tmp_path, monkeypatch):
    import download_data

    def broken_retrieve(url, dest):  # interrupted download
        Path(dest).write_text("partial", encoding="utf-8")
        raise OSError("connection lost")

    monkeypatch.setattr(download_data.urllib.request, "urlretrieve", broken_retrieve)
    out = tmp_path / "phinc.csv"
    with pytest.raises(OSError):
        download_data.download("http://example.invalid", out)
    assert not out.exists()                         # so the next run retries


def test_evaluate_perfect_and_bad():
    refs = ["i am going home", "what is your name"]
    perfect = evaluate(refs, refs)
    assert perfect["bleu"] == 100.0 and perfect["chrf"] == 100.0
    bad = evaluate(["zzz qqq", "xxx"], refs)
    assert bad["bleu"] < 5


def test_evaluate_ignores_case():
    # Pretrained models output capitalised English; references are lowercased.
    # Note: BLEU needs at least 4 words (it counts 4-grams); a 2-word sentence always gives BLEU 0.
    scores = evaluate(["Hello World Is Here Now"], ["hello world is here now"])
    assert scores["bleu"] == 100.0 and scores["chrf"] == 100.0
    assert evaluate(["Hello World"], ["hello world"])["chrf"] == 100.0


def test_save_errors_ranks_with_lowercase(tmp_path):
    from evaluate import save_errors

    # A correct-but-capitalised prediction must not be ranked as a bad translation.
    n = save_errors(["a", "b"], ["Hello World", "zzz"], ["hello world", "hello world"],
                    tmp_path / "e.csv")
    first_data_row = (tmp_path / "e.csv").read_text(encoding="utf-8").splitlines()[1]
    assert n == 2 and first_data_row.startswith("b,zzz")  # the really wrong one is first


def test_evaluate_length_mismatch():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a", "b"])
