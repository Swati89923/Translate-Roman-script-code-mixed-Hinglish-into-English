# Hinglish to English Machine Translation

[![tests](https://github.com/Swati89923/Translate-Roman-script-code-mixed-Hinglish-into-English/actions/workflows/tests.yml/badge.svg)](https://github.com/Swati89923/Translate-Roman-script-code-mixed-Hinglish-into-English/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10-blue)
![task](https://img.shields.io/badge/task-machine%20translation-informational)

Translate Roman-script, code-mixed Hinglish (Hindi and English mixed in one sentence, written in
the Latin alphabet) into English. Course project, **Group 8 (P08)**.

**Jump to:** [Overview](#overview) · [Team](#team) · [Status](#project-status) · [Data](#data) ·
[Quick start](#quick-start) · [Evaluation](#evaluation-protocol) · [Models](#models) ·
[Results](#results) · [Limitations](#limitations-and-next-steps) · [How we work](#how-we-work)

## Overview

Social media users mix Hindi and English freely and spell Hindi words in Latin letters with no
fixed spelling. That is hard for machine translation: there is little parallel data, spellings
vary (`kya`, `kyaa`, `kyaaa`), and many English words appear unchanged inside Hindi sentences.

| Hinglish (input) | English (reference) |
| --- | --- |
| `@singhlicious ye sach hai milord` | `@singhlicious this is the truth, my lord!` |
| `family kaha hai umar kam-akal bhai <url>` | `where is family less intelligent bro <url>` |

**Research question.** How much does **pretraining** help on a low-resource, code-mixed
translation task compared with **training from scratch**? We train three model families on the
same data, score them with the same metrics (BLEU and chrF) and compare the kinds of errors each
one makes.

```mermaid
flowchart LR
    A[PHINC raw csv] --> B[preprocess.py<br/>clean + fixed split]
    B --> C[train / val / test tsv]
    C --> D1[LSTM + attention<br/>from scratch]
    C --> D2[Transformer<br/>from scratch]
    C --> D3[mBART / mT5 / IndicTrans2<br/>fine-tuned]
    D1 --> E[evaluate.py<br/>BLEU + chrF]
    D2 --> E
    D3 --> E
    E --> F[results/*.json<br/>+ error csv per model]
    F --> G[comparison + error analysis]
```

## Team

| Member | GitHub | Issue | Task |
| --- | --- | --- | --- |
| Swati Kumari | [@Swati89923](https://github.com/Swati89923) | #1 | Project setup, PHINC data pipeline, shared BLEU/chrF evaluation, CI, final comparison and report |
| Pinky Rana | [@Pinkyrana398](https://github.com/Pinkyrana398) | #2 | Baseline: seq2seq LSTM with Bahdanau attention, trained from scratch |
| Vinay Kumar | [@Vinays-08](https://github.com/Vinays-08) | #4 | Fine-tuning pretrained models: mBART-50, mT5, IndicTrans2 |
| Ujjval Baijal | [@Ujjval-1](https://github.com/Ujjval-1) | #3 | Transformer encoder-decoder trained from scratch |

## Project status

| Step | What | Owner | Issue | Status |
| --- | --- | --- | --- | --- |
| 1 | Data pipeline, shared evaluation, tests | Swati | #1, #7 | Done (merged) |
| 1b | CI: pytest on every pull request | Swati | #9 | Done (merged) |
| 2 | LSTM + Bahdanau attention | Pinky | #2 | Done, results in `results/` |
| 3 | Fine-tune mBART / mT5 / IndicTrans2 | Vinay | #4 | Code merged, full runs and scores pending |
| 4 | Transformer from scratch | Ujjval | #3 | In progress |
| 5 | Final comparison table, error analysis, report | Swati and team | | After steps 3 and 4 |

## Data

**PHINC**: about 13.7k Hinglish-English sentence pairs of social media text.
Srivastava and Singh, *PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine
Translation*, W-NUT 2020 ([paper](https://arxiv.org/abs/2004.09447),
[Zenodo](https://doi.org/10.5281/zenodo.3605597)). The `data/` folder is not committed, so each
person recreates it with the commands in [Quick start](#quick-start).

After cleaning (13,738 raw rows) the fixed split (80/10/10, seed 42) is:

| Split | Pairs | Hinglish words per sentence (mean / max) | English words per sentence (mean / max) |
| --- | --- | --- | --- |
| Train | 10,762 | 13.3 / 40 | 12.5 / 46 |
| Validation | 1,345 | 13.1 / 37 | 12.5 / 37 |
| Test | 1,345 | 13.1 / 37 | 12.2 / 37 |

About 28% of the Hinglish words also appear unchanged in the English sentence (English words,
names, mentions, hashtags), a rough indicator of how much English is mixed in. Full statistics
are in `results/eda.txt`.

If you trained a model on an older split (10,991 / 1,373 / 1,373 or 10,803 / 1,350 / 1,350), run
`python src/preprocess.py` again and retrain, so that all models use exactly the same data.

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

## Quick start

```bash
git clone https://github.com/Swati89923/Translate-Roman-script-code-mixed-Hinglish-into-English.git
cd Translate-Roman-script-code-mixed-Hinglish-into-English
pip install -r requirements.txt
python src/download_data.py     # downloads data/raw/phinc.csv
python src/preprocess.py        # writes data/processed/{train,val,test}.tsv
python src/eda.py               # optional: statistics, saved to results/eda.txt
python -m pytest                # tests for the pipeline, evaluation and the LSTM model
```

The same tests run automatically on every pull request and on pushes to `main` with GitHub
Actions (`.github/workflows/tests.yml`); the `tests` check must be green before merging.
Training the models needs a GPU (for example a free Colab or Kaggle T4); the smoke-test options
in the model sections run on a CPU.

## Evaluation protocol

All models are scored the same way, otherwise the numbers cannot be compared:

1. Train on `train.tsv`, tune on `val.tsv`, report once on `test.tsv`.
2. Score with `evaluate()` from `src/evaluate.py` (sacreBLEU **BLEU** and **chrF**). It lowercases
   predictions and references, so models that output capitalised English (mBART, mT5) are not
   penalised for casing.
3. Save scores in `results/<model>.json` and the worst translations in `results/<model>_errors.csv`.
4. Error analysis: tag at least 50 rows of the errors csv per model with an error category
   (the `error_category` column), then compare the categories across models.

```python
import sys; sys.path.insert(0, "src")
from preprocess import read_tsv
from evaluate import evaluate, save_result

test = read_tsv("data/processed/test.tsv")       # list of (hinglish, english)
hypotheses = [...]                               # one English string per test pair
scores = evaluate(hypotheses, [t for _, t in test])   # {"bleu": ..., "chrf": ..., "num_sentences": ...}
save_result(scores, "results/<your_model>.json", model="<your_model>")
```

## Models

### 1. LSTM + Bahdanau attention (Pinky, #2)

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

On Colab, open `notebooks/lstm_attn_colab.ipynb` and run all cells. Hyperparameters are in
`configs/lstm_attn.json` (embedding 256, encoder 2x256, decoder 512, dropout 0.3, Adam 1e-3,
batch 64, gradient clipping 1.0, label smoothing 0.1, early stopping with patience 4, beam size 5).
About 12% of the English training words are rare and become `<unk>`, so `<unk>` is blocked during
decoding (this raised BLEU from 6.50 to 8.97).

![LSTM training curves](results/lstm_attn_curves.png)

Training loss keeps falling while validation loss stays flat after about epoch 5 to 10, so the
model starts to overfit; early stopping ended training at epoch 15. The attention heatmap below
shows the model aligning English words with the right Hinglish words (for example `very` with
`bahut`). More heatmaps: `results/lstm_attn_attention_2.png` and `_3.png`.

<img src="results/lstm_attn_attention_1.png" alt="Attention heatmap" width="380">

### 2. Transformer from scratch (Ujjval, #3)

Transformer encoder-decoder trained from random initialisation (no pretrained weights), to compare
with the LSTM and with the pretrained models. In progress; this section is filled in when the
pull request is merged.

### 3. Fine-tuned pretrained models (Vinay, #4)

`src/finetune.py` fine-tunes pretrained translation models on PHINC, or evaluates them zero-shot
with `--epochs 0`:

```bash
python src/finetune.py --model mbart --epochs 3          # mBART-50
python src/finetune.py --model mt5 --epochs 5            # mT5
python src/finetune.py --model indictrans2 --epochs 3    # IndicTrans2 (needs IndicTransToolkit)
python src/finetune.py --model mt5 --epochs 0            # zero-shot baseline
python src/finetune.py --model mt5 --max-train 64 --max-test 20 --epochs 1   # CPU smoke test
```

Training uses gradient accumulation and mixed precision on GPU (fp32 for mT5, which is unstable in
fp16). Scores are written to `results/finetune_<model>.json`. Full runs on a GPU are still pending.

## Results

Test set, 1,345 sentences, scored with `evaluate()` (BLEU and chrF, higher is better).
A cell is empty until that model has been run on the current split.

| Model | Trained by | BLEU | chrF |
| --- | --- | --- | --- |
| LSTM + Bahdanau attention (beam 5) | Pinky | 8.97 | 27.71 |
| LSTM + Bahdanau attention (greedy) | Pinky | 7.64 | 27.17 |
| Transformer from scratch | Ujjval | | |
| mBART-50 fine-tuned | Vinay | | |
| mT5 fine-tuned | Vinay | | |
| IndicTrans2 fine-tuned | Vinay | | |
| Zero-shot pretrained (no fine-tuning) | Vinay | | |

**What we know so far (LSTM only).** Beam search beats greedy decoding by about 1.3 BLEU.
Error analysis of the 50 worst translations, tagged by hand:

| Error category | Count |
| --- | --- |
| Wrong meaning | 17 |
| Junk output (learned from noisy data) | 14 |
| Noisy reference | 9 |
| Repetition | 4 |
| English not copied | 3 |
| Dropped words | 2 |
| Wrong word choice | 1 |

The question "does pretraining help?" can only be answered once the Transformer and the
pretrained models have been run on the same split.

## Limitations and next steps

- **Noisy references.** PHINC translations are sometimes wrong or only loosely aligned (about 9 of
  the 50 worst LSTM errors were tagged "noisy reference"), which lowers every model's BLEU.
- **Small data.** About 10.8k training pairs; from-scratch models overfit quickly.
- **Emoticon-only references.** About 34 training references are still only emoticons such as
  `) :p` because `p` counts as a letter. The LSTM sometimes outputs `) :p` because of them.
  Fixing this changes the split again, so it has to be decided once for all models.
- **Single run per model.** Scores come from one seed; differences of a few tenths of BLEU are
  not meaningful.

Ideas for after the three model families are done (planned, not started): train on 10%, 25%, 50%
and 100% of the data to see where pretraining helps most; repeat runs with several seeds or use a
paired bootstrap test; break scores down by sentence length and amount of code-mixing.

## How we work

- One GitHub issue per task, one branch and one pull request per issue
  (for example `feat/issue-3-transformer`). The PR description says `Closes #<n>`.
- Nobody pushes directly to `main`. Every PR needs a green `tests` check and a review from a
  teammate (at least 15 words, saying what was checked) before it is merged.
- Do not commit data, checkpoints or large files (`data/` and `checkpoints/` are gitignored).
- Everyone uses the shared split and `evaluate()`, so results stay comparable.
- Each member commits their own work under their own GitHub account.

## Repo layout

```
src/download_data.py       download PHINC from Zenodo
src/preprocess.py          cleaning + fixed train/val/test split
src/vocab.py               word-level vocabulary and SentencePiece helper
src/evaluate.py            BLEU + chrF (sacreBLEU) and error csv, used by all models
src/eda.py                 dataset statistics
src/models/lstm_attn.py    LSTM + Bahdanau attention model (#2)
src/train_lstm.py          training / evaluation script for #2
src/finetune.py            fine-tuning of pretrained models (#4)
configs/                   hyperparameters per model
notebooks/                 Colab notebooks
tests/                     pytest tests
results/                   score json files, plots and error csvs (small files, committed)
.github/workflows/         CI (pytest)
```

## References

1. Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.*
   ICLR 2015. <https://arxiv.org/abs/1409.0473>
2. Srivastava, Singh. *PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for Machine
   Translation.* W-NUT 2020. <https://arxiv.org/abs/2004.09447>
