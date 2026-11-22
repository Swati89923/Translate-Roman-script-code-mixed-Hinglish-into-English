"""Tests for the report generator in docs/report (no browser or LaTeX needed)."""

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
sys.path.insert(0, str(REPORT_DIR))

import build_report  # noqa: E402
import content  # noqa: E402


def test_content_numbers_come_from_results_files():
    doc = content.build()
    mbart = content.score("finetune_mbart")
    assert f"{mbart['bleu']:.2f}" in doc["abstract"]
    tables = [b[1] for b in doc["blocks"] if b[0] == "table"]
    results = next(t for t in tables if t["label"] == "results")
    assert results["rows"][0][1] == "mBART-50 fine-tuned"          # best system first
    assert results["rows"][0][3] == f"{mbart['bleu']:.2f}"


def test_every_citation_key_exists():
    doc = content.build()
    refs = doc["references"]
    import re
    text = " ".join(str(b[1]) for b in doc["blocks"] if b[0] in ("p", "ul")) + doc["abstract"]
    keys = {k.strip() for m in re.findall(r"\[[@+]([^\]]+)\]", text) for k in m.split(";")}
    assert keys and keys <= set(refs)


def test_tables_and_figures_are_numbered_in_order():
    doc = content.build()
    num = build_report.Numbering([b for b in doc["blocks"] if b[0] in ("table", "fig")]).num
    tabs = sorted(v for k, v in num.items() if k.startswith("tab:"))
    assert tabs == list(range(1, len(tabs) + 1))
    assert "fig:pipeline" in num and "tab:results" in num


def test_inline_markup_html_and_latex():
    refs, num = {"a": ("Smith and Lee", "2020", "x")}, {"tab:t": 3}
    html_out = build_report.inline_html("**b** *i* `<url>` [@a] [+a] [[tab:t]]", refs, num)
    assert "<b>b</b>" in html_out and "<i>i</i>" in html_out and "<code>&lt;url&gt;</code>" in html_out
    assert "(Smith and Lee 2020)" in html_out and "Smith and Lee (2020)" in html_out and "Table&nbsp;3" in html_out
    tex_out = build_report.inline_tex("50% of `a_b` [[tab:t]]", refs, num)
    assert r"50\% of" in tex_out and r"\texttt{a\_b}" in tex_out and r"\ref{tab:t}" in tex_out


def test_decimal_points_are_centred_only_in_numbers():
    assert build_report.centre_decimals("36.76") == "36·76"
    assert build_report.centre_decimals("10,762") == "10,762"
    assert build_report.centre_decimals("see Fig. 2.5") == "see Fig. 2.5"
