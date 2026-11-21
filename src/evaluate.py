"""Shared evaluation: BLEU and chrF for every model in the project.

All models (LSTM, Transformer, mBART / mT5 / IndicTrans2) MUST score their test
predictions through `evaluate()` below. Using one function and one sacreBLEU
configuration is what makes the numbers comparable in the final report.

Use from Python (put src/ on the path first, same style as the README):
    import sys; sys.path.insert(0, "src")
    from evaluate import evaluate, save_result
    scores = evaluate(hypotheses, references)      # two lists of strings
    save_result(scores, "results/lstm_attn.json", model="lstm_attn")

Use from the command line (one sentence per line in each file):
    python src/evaluate.py --hyp preds.txt --ref refs.txt --out results/my_model.json
"""

import argparse
import json
from pathlib import Path


def evaluate(hypotheses, references):
    """Return corpus-level BLEU and chrF.

    hypotheses: list of predicted English sentences
    references: list of reference English sentences (same length / order)

    Both sides are lowercased before scoring. preprocess.py lowercases the
    references, but pretrained models (mBART, mT5, IndicTrans2) output normal
    capitalised English; without this they would be penalised for casing alone
    and the pretrained-vs-scratch comparison would be biased.
    """
    import sacrebleu  # imported lazily so importing this module stays cheap

    if len(hypotheses) != len(references):
        raise ValueError(
            f"Got {len(hypotheses)} hypotheses but {len(references)} references."
        )
    if not hypotheses:
        raise ValueError("Cannot evaluate an empty list.")

    hypotheses = [h.lower() for h in hypotheses]
    references = [r.lower() for r in references]  # no-op for the default (lowercased) splits

    # sacreBLEU expects a LIST of reference streams (one per reference set).
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "num_sentences": len(hypotheses),
    }


def save_result(scores: dict, path, model: str = "") -> None:
    """Write the scores (plus the model name) to a json file in results/."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, **scores}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_errors(sources, hypotheses, references, path, max_rows: int = 100) -> int:
    """Write the worst translations to a csv for the manual error analysis.

    Sentences are ranked by their sentence-level chrF (lowest first), so the
    file starts with the most wrong translations. Each model's author reads
    at least 50 rows and tags them with an error category (e.g. "untranslated
    hindi word", "wrong word choice", "dropped words"); the team then compares
    the categories across models. Returns the number of rows written.
    """
    import csv

    import sacrebleu

    rows = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        # Same lowercasing as evaluate(), so the ranking matches the reported scores.
        score = sacrebleu.sentence_chrf(hyp.lower(), [ref.lower()]).score
        rows.append((score, src, hyp, ref))
    rows.sort(key=lambda r: r[0])  # worst first

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # "error_category" is left empty on purpose: fill it in by hand.
        writer.writerow(["source", "prediction", "reference", "chrf", "error_category"])
        for score, src, hyp, ref in rows[:max_rows]:
            writer.writerow([src, hyp, ref, round(score, 1), ""])
    return min(len(rows), max_rows)


def _read_lines(path):
    # splitlines() keeps blank lines in the middle, so an empty prediction still
    # counts as one (wrong) sentence and the line numbers stay aligned.
    return Path(path).read_text(encoding="utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BLEU and chrF.")
    parser.add_argument("--hyp", required=True, help="file with predictions, one per line")
    parser.add_argument("--ref", required=True, help="file with references, one per line")
    parser.add_argument("--out", help="optional json file to save the scores to")
    parser.add_argument("--model", default="", help="model name stored in the json")
    args = parser.parse_args()

    scores = evaluate(_read_lines(args.hyp), _read_lines(args.ref))
    print(json.dumps(scores, indent=2))
    if args.out:
        save_result(scores, args.out, model=args.model)


if __name__ == "__main__":
    main()
