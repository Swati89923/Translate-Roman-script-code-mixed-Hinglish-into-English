"""Final comparison of all models, built from the files in results/.

Run from the repo root after the models have written their results:

    python src/compare.py

Writes:
    results/comparison.md         test BLEU / chrF of every model that has a result file
    results/comparison.png        bar chart of the same numbers
    results/error_categories.md   how many of the 50 hand-tagged worst translations per model
                                  fall into each error category
    results/error_examples.md     a few real tagged examples per model

Why a script: the numbers in the README and the report are copied from this output, so they
always match results/*.json. A model whose file is missing is simply left out (or shown as
"not run"), so the script also works while some models are still unfinished.
"""

import csv
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

RESULTS = Path("results")

# (label, owner, result json, errors csv, how it was trained). Order = order in the table.
MODELS = [
    ("LSTM + Bahdanau attention (beam 5)", "Pinky", "lstm_attn", "lstm_attn_errors", "from scratch"),
    ("Transformer, 6 layers (beam 5)", "Ujjval", "transformer", "transformer_errors", "from scratch"),
    ("mT5-small zero-shot", "Vinay", "finetune_mt5_zeroshot", None, "pretrained, no training"),
    ("mT5-small fine-tuned", "Vinay", "finetune_mt5", "finetune_mt5_errors", "pretrained + fine-tuned"),
    ("mBART-50 zero-shot", "Vinay", "finetune_mbart_zeroshot", None, "pretrained, no training"),
    ("mBART-50 fine-tuned", "Vinay", "finetune_mbart", "finetune_mbart_errors", "pretrained + fine-tuned"),
]


def load_scores(results_dir: Path = RESULTS):
    """Return a list of dicts (label, owner, kind, bleu, chrf, n) for the models that have a json."""
    rows = []
    for label, owner, name, _, kind in MODELS:
        path = results_dir / f"{name}.json"
        if not path.exists():
            rows.append({"label": label, "owner": owner, "kind": kind, "bleu": None, "chrf": None, "n": None})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"label": label, "owner": owner, "kind": kind,
                     "bleu": data["bleu"], "chrf": data["chrf"], "n": data.get("num_sentences")})
    return rows


def scores_table(rows) -> str:
    """Markdown table sorted by BLEU (best first); models without a result go last as "not run"."""
    done = sorted((r for r in rows if r["bleu"] is not None), key=lambda r: -r["bleu"])
    missing = [r for r in rows if r["bleu"] is None]
    lines = ["| Rank | Model | Trained by | How | BLEU | chrF |", "| --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(done, 1):
        lines.append(f"| {i} | {r['label']} | {r['owner']} | {r['kind']} | {r['bleu']:.2f} | {r['chrf']:.2f} |")
    for r in missing:
        lines.append(f"| - | {r['label']} | {r['owner']} | {r['kind']} | not run | not run |")
    return "\n".join(lines) + "\n"


def normalise_category(tag: str) -> str:
    """"junk output (learned from noisy data)" -> "Junk output": drop the note, capitalise."""
    tag = re.sub(r"\s*\(.*?\)", "", tag).strip()
    return tag[:1].upper() + tag[1:]


def count_categories(csv_path: Path) -> Counter:
    """Count the hand-tagged rows of one errors csv by category (untagged rows are ignored)."""
    counts = Counter()
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tag = (row.get("error_category") or "").strip()
            if tag:
                counts[normalise_category(tag)] += 1
    return counts


def category_table(results_dir: Path = RESULTS) -> str:
    """One markdown table: rows = error categories, columns = models with tagged errors."""
    per_model = OrderedDict()
    for label, _, _, errors, _ in MODELS:
        path = results_dir / f"{errors}.csv" if errors else None
        if path and path.exists():
            counts = count_categories(path)
            if counts:
                per_model[label] = counts
    if not per_model:
        return "No tagged error files found.\n"
    total = Counter()
    for c in per_model.values():
        total.update(c)
    categories = [c for c, _ in total.most_common()]
    short = [re.sub(r" \(.*\)", "", m) for m in per_model]  # drop "(beam 5)" from the headers
    lines = ["| Error category | " + " | ".join(short) + " |",
             "| --- | " + " | ".join("---" for _ in per_model) + " |"]
    for cat in categories:
        lines.append(f"| {cat} | " + " | ".join(str(c.get(cat, "")) for c in per_model.values()) + " |")
    lines.append("| **Tagged rows** | " + " | ".join(f"**{sum(c.values())}**" for c in per_model.values()) + " |")
    return "\n".join(lines) + "\n"


def examples_markdown(results_dir: Path = RESULTS, per_category: int = 1, max_len: int = 110) -> str:
    """A few real tagged examples per model (the worst-scoring row of each category)."""
    out = []
    for label, _, _, errors, _ in MODELS:
        path = results_dir / f"{errors}.csv" if errors else None
        if not (path and path.exists()):
            continue
        seen = Counter()
        rows = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):  # the csv is sorted worst first
                tag = (row.get("error_category") or "").strip()
                if not tag or seen[tag] >= per_category:
                    continue
                if max(len(row["source"]), len(row["prediction"]), len(row["reference"])) > max_len:
                    continue  # keep the examples short
                seen[tag] += 1
                rows.append((normalise_category(tag), row))
        if not rows:
            continue
        out.append(f"### {re.sub(r' [(].*[)]', '', label)}\n")
        out.append("| Category | Hinglish | Model output | Reference |\n| --- | --- | --- | --- |")
        for cat, r in rows:
            cells = [x.replace("|", "\\|") for x in (r["source"], r["prediction"], r["reference"])]
            out.append(f"| {cat} | `{cells[0]}` | `{cells[1]}` | `{cells[2]}` |")
        out.append("")
    return "\n".join(out) + "\n"


def bar_chart(rows, path: Path) -> None:
    """Horizontal bar chart of BLEU and chrF for the models that have results."""
    import matplotlib
    matplotlib.use("Agg")  # no window needed, we only save a png
    import matplotlib.pyplot as plt

    done = sorted((r for r in rows if r["bleu"] is not None), key=lambda r: r["bleu"])
    labels = [re.sub(r" \(.*\)", "", r["label"]) for r in done]
    ys = range(len(done))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(done) + 1.6))
    ax.barh([y + 0.2 for y in ys], [r["chrf"] for r in done], height=0.4, label="chrF", color="#9ecae1")
    ax.barh([y - 0.2 for y in ys], [r["bleu"] for r in done], height=0.4, label="BLEU", color="#3182bd")
    for y, r in zip(ys, done):
        ax.text(r["bleu"] + 0.5, y - 0.2, f"{r['bleu']:.1f}", va="center", fontsize=8)
        ax.text(r["chrf"] + 0.5, y + 0.2, f"{r['chrf']:.1f}", va="center", fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.set_xlabel("score on the test set (1,345 sentences), higher is better")
    ax.set_xlim(0, max(r["chrf"] for r in done) + 8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load_scores()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "comparison.md").write_text(scores_table(rows), encoding="utf-8")
    (RESULTS / "error_categories.md").write_text(category_table(), encoding="utf-8")
    (RESULTS / "error_examples.md").write_text(examples_markdown(), encoding="utf-8")
    if any(r["bleu"] is not None for r in rows):
        bar_chart(rows, RESULTS / "comparison.png")
    print(scores_table(rows))
    print(category_table())
    print("wrote results/comparison.md, comparison.png, error_categories.md, error_examples.md")


if __name__ == "__main__":
    main()
