# Hinglish to English Machine Translation

Translate Roman-script, code-mixed Hinglish (Hindi + English written in the Latin alphabet)
into English. Course project, Group 8 (P08).

**Research question:** how much does pretraining help on a low-resource, code-mixed
translation task compared with training from scratch? We compare three model families on the
same data and the same metrics (BLEU, chrF) and look at the kinds of errors each one makes.

| Step | Model | Issue |
| --- | --- | --- |
| 1 | Data pipeline + shared evaluation | #1 |
| 2 | Seq2seq LSTM with Bahdanau attention (from scratch) | #2 |
| 3 | Transformer (from scratch) | #3 |
| 4 | Fine-tuned mBART / mT5 / IndicTrans2 (pretrained) | #4 |

## Team

| Member | Issue |
| --- | --- |
| Swati Kumari | #1 data pipeline, evaluation, final comparison |
| Pinky Rana | #2 LSTM + attention |
| Ujjval Baijal | #3 Transformer from scratch |
| Vinay Kumar | #4 pretrained fine-tuning |

## Data

PHINC: about 13.7k Hinglish-English sentence pairs of social media text.
Srivastava and Singh, *PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine
Translation*, W-NUT 2020 ([paper](https://arxiv.org/abs/2004.09447),
[Zenodo](https://doi.org/10.5281/zenodo.3605597)). The `data/` folder is not committed, so each
person recreates it with the commands below.

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py     # downloads data/raw/phinc.csv
python src/preprocess.py        # writes data/processed/{train,val,test}.tsv (seed 42, 80/10/10)
python src/eda.py               # optional: statistics, saved to results/eda.txt
python -m pytest                # tests for the pipeline and evaluation
```

The same tests run automatically on every pull request and on pushes to `main` with GitHub
Actions (`.github/workflows/tests.yml`); the `tests` check must be green before merging.

After cleaning (13,738 raw rows): 10,762 train, 1,345 validation, 1,345 test pairs.
If you trained a model on an older split (10,991 / 1,373 / 1,373 or 10,803 / 1,350 / 1,350), run `python src/preprocess.py`
again and retrain, so that all models use exactly the same data.

### Cleaning rules (same for every model)

- HTML codes such as `&amp;` and `&lt;` are decoded with `html.unescape` (done first).
- URLs become `<url>`; trailing "..." removed; whitespace collapsed.
- Everything is lowercased (use `--keep-case` to turn this off).
- Mentions, hashtags and emojis are kept, because the English reference keeps them too.
- Pairs where either side has no letter at all are dropped. The raw file has 234 junk translations
  that are only `)`, `$` or `-)`; a model trained on them learns to output `)`.
- Pairs where either side ends with a bare `&` are dropped (51 pairs). The raw tweets were cut at
  `&amp;`, so source and reference are two halves of the same sentence (e.g. `...from h&` / `m tomorrow`).
- Exact duplicate (Hinglish, English) pairs are dropped, so no identical pair is in two splits.
  The same Hinglish sentence can still appear with a different translation (1 validation
  sentence and 1 test sentence also occur in train).
- `evaluate()` lowercases predictions and references before scoring, so models that output
  capitalised English (mBART, mT5) are not penalised for casing.

## How to use the shared code in your own model

```python
import sys; sys.path.insert(0, "src")           # or run from the repo root with `src.` imports
from preprocess import read_tsv
from vocab import Vocab
from evaluate import evaluate, save_result

train = read_tsv("data/processed/train.tsv")    # list of (hinglish, english)
test = read_tsv("data/processed/test.tsv")

vocab_src = Vocab([s for s, _ in train])        # build vocabularies from TRAIN only
vocab_tgt = Vocab([t for _, t in train])

# ... train your model, then translate the test sentences:
hypotheses = [...]                              # one English string per test pair
references = [t for _, t in test]
scores = evaluate(hypotheses, references)       # {"bleu": ..., "chrf": ..., "num_sentences": ...}
save_result(scores, "results/<your_model>.json", model="<your_model>")
```

Rules so the results stay comparable:

1. Always train on `train.tsv`, tune on `val.tsv`, and report on `test.tsv` once at the end.
2. Always score with `evaluate()`; do not compute BLEU yourself.
3. Save the scores in `results/<model>.json` and your wrong translations in
   `results/<model>_errors.csv` for the error analysis.

## Model 1: LSTM + Bahdanau attention (#2)

Encoder-decoder LSTM with additive attention, following Bahdanau, Cho and Bengio,
*Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015
([paper](https://arxiv.org/abs/1409.0473)). Bidirectional LSTM encoder, LSTM decoder that
attends over all source words at every step, trained from scratch with word-level vocabularies
built from `train.tsv`.

```bash
python src/train_lstm.py                          # train + test (GPU recommended, e.g. Colab)
python src/train_lstm.py --eval-only              # test again from checkpoints/lstm_attn.pt
python src/train_lstm.py --epochs 1 --limit 300   # quick smoke test on CPU
```

On Colab, open `notebooks/lstm_attn_colab.ipynb` and run all cells.
Hyperparameters: `configs/lstm_attn.json` (embedding 256, encoder 2x256, decoder 512, dropout 0.3,
Adam 1e-3, batch 64, gradient clipping 1.0, label smoothing 0.1, early stopping on validation loss
with patience 4, beam size 5, `<unk>` blocked during decoding because ~12% of the English
training words are rare and become `<unk>`).
Outputs: `results/lstm_attn.json` (test BLEU/chrF with beam search, greedy for comparison),
training curves, 3 attention heatmaps and `results/lstm_attn_errors.csv`.

## Repo layout

```
src/download_data.py   download PHINC from Zenodo
src/preprocess.py      cleaning + fixed train/val/test split
src/vocab.py           word-level vocabulary and SentencePiece helper
src/evaluate.py        BLEU + chrF (sacreBLEU), used by all models
src/eda.py             dataset statistics
src/models/lstm_attn.py  LSTM + Bahdanau attention model (#2)
src/train_lstm.py      training / evaluation script for #2
configs/               hyperparameters per model
notebooks/             Colab notebooks
tests/                 pytest tests
results/               score json files and error csvs (small files, committed)
```

## Results

Filled in once issues #2, #3 and #4 are done (BLEU / chrF on the test set).

| Model | BLEU | chrF |
| --- | --- | --- |
| LSTM + Bahdanau attention (beam 5) | 8.97 | 27.71 |
| Transformer (scratch) | | |
| Pretrained (fine-tuned) | | |
