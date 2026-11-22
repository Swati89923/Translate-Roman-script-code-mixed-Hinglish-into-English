"""Tests for src/compare.py using small fake result files (no real training needed)."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare import (category_table, count_categories, load_scores, normalise_category,
                     scores_table)


def write_result(folder, name, bleu, chrf):
    (folder / f"{name}.json").write_text(
        json.dumps({"model": name, "bleu": bleu, "chrf": chrf, "num_sentences": 1345}),
        encoding="utf-8")


def write_errors(folder, name, tags):
    with open(folder / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "prediction", "reference", "chrf", "error_category"])
        for i, tag in enumerate(tags):
            w.writerow([f"src {i}", f"pred {i}", f"ref {i}", 1.0, tag])


def test_normalise_category_drops_note_and_capitalises():
    assert normalise_category("junk output (learned from noisy data)") == "Junk output"
    assert normalise_category("wrong meaning") == "Wrong meaning"


def test_scores_table_is_sorted_by_bleu_and_lists_missing_models(tmp_path):
    write_result(tmp_path, "lstm_attn", 8.97, 27.71)
    write_result(tmp_path, "finetune_mbart", 36.76, 57.87)
    table = scores_table(load_scores(tmp_path))
    lines = table.splitlines()
    assert "mBART-50 fine-tuned" in lines[2] and "36.76" in lines[2]   # best first
    assert "LSTM" in lines[3] and "8.97" in lines[3]
    assert "not run" in table                                           # models without a json


def test_count_categories_ignores_untagged_rows(tmp_path):
    write_errors(tmp_path, "m", ["wrong meaning", "wrong meaning", "", "dropped words"])
    counts = count_categories(tmp_path / "m.csv")
    assert counts == {"Wrong meaning": 2, "Dropped words": 1}


def test_category_table_has_one_column_per_tagged_model(tmp_path):
    write_errors(tmp_path, "lstm_attn_errors", ["wrong meaning", "repetition"])
    write_errors(tmp_path, "transformer_errors", ["wrong meaning"])
    table = category_table(tmp_path)
    header = table.splitlines()[0]
    assert "LSTM" in header and "Transformer" in header
    assert "| Wrong meaning | 1 | 1 |" in table
    assert "| **Tagged rows** | **2** | **1** |" in table
