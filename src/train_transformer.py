"""Train and evaluate the from-scratch Transformer (issue #3).

Usage, from the repo root after download_data.py and preprocess.py:
    python src/train_transformer.py --config configs/transformer_small.json
    python src/train_transformer.py --config configs/transformer_deep.json
    python src/train_transformer.py --config configs/transformer_small.json --eval-only
    python src/train_transformer.py --epochs 1 --limit 300    # quick smoke test (writes to checkpoints/smoke/)

PHINC is small (about 11k training pairs), so a big model can overfit. The issue asks
for at least two settings; the two config files are that comparison. Each run writes
its files under the "name" in its config.

Outputs (NAME = "name" in the config):
    checkpoints/NAME.pt              best model by validation loss (gitignored)
    checkpoints/spm_<vocab_size>.*   joint BPE tokenizer (gitignored)
    results/NAME.json                test BLEU / chrF (beam search, plus greedy for comparison)
    results/NAME_history.json        loss, val BLEU and learning rate per epoch
    results/NAME_curves.png          training curves
    results/NAME_errors.csv          100 worst test translations for the error analysis
The best setting is copied to results/transformer.json and results/transformer_errors.csv.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from evaluate import evaluate, save_errors, save_result
from models.transformer import TransformerMT
from preprocess import read_tsv
from vocab import train_sentencepiece

SPLIT_DIR = Path("data/processed")
PAD, UNK, BOS, EOS = 0, 1, 2, 3  # ids fixed by vocab.train_sentencepiece


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(train_pairs, vocab_size, prefix):
    """Train (or load) one BPE model on the TRAIN sentences of both languages.

    Subwords matter here: Hinglish spelling is inconsistent ("kya", "kyaa", "kyaaa"),
    so a word-level vocabulary has many unknown words (the LSTM loses about 12% of the
    English training words to <unk>). BPE splits a rare word into pieces the model has
    seen. The vocabulary is small (4000) because there is little data: with a large one
    most pieces would have too few examples to learn from.
    """
    import sentencepiece as spm

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    model_file = f"{prefix}.model"
    if not Path(model_file).exists():
        train_sentencepiece([s for s, _ in train_pairs] + [t for _, t in train_pairs], prefix, vocab_size)
    return spm.SentencePieceProcessor(model_file=model_file)


def src_ids(text, sp, max_len):
    return sp.encode(text)[: max_len - 1] + [EOS]


def pad(seqs):
    out = torch.full((len(seqs), max(len(s) for s in seqs)), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def make_batches(pairs, sp, cfg, shuffle, seed=0):
    """Padded tensors: src, tgt_in (<bos> ...) and tgt_out (... <eos>).

    Teacher forcing: the decoder reads the correct English so far (tgt_in) and must
    predict the same sentence shifted one step left (tgt_out). For "i am fine":
        tgt_in  = <bos> i  am   fine
        tgt_out = i     am fine <eos>
    """
    data = []
    for s, t in pairs:
        tgt = sp.encode(t)[: cfg["max_len"] - 1]  # English sentence -> subword ids
        data.append((src_ids(s, sp, cfg["max_len"]), [BOS] + tgt, tgt + [EOS]))
    order = list(range(len(data)))
    if shuffle:
        # Batches of similar-length sentences need much less padding, which roughly halves
        # the time per epoch on a CPU. The random tie-break and the shuffled batch order keep
        # the batches different (but reproducible) on every epoch.
        rng = random.Random(seed)
        order.sort(key=lambda j: (len(data[j][0]), rng.random()))
    batches = [order[i:i + cfg["batch_size"]] for i in range(0, len(order), cfg["batch_size"])]
    if shuffle:
        rng.shuffle(batches)
    for batch in batches:
        chunk = [data[j] for j in batch]
        yield tuple(pad([c[k] for c in chunk]) for k in range(3))


class NoamScheduler:
    """Learning rate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).

    The rate rises linearly for `warmup` steps and then decays with 1/sqrt(step).
    Warmup matters for Transformers: with a high rate from the first step the
    attention weights are still random and training often diverges.
    """

    def __init__(self, optimizer, d_model, warmup):
        self.opt, self.d_model, self.warmup, self.step_num = optimizer, d_model, warmup, 0

    def rate(self):
        n = max(self.step_num, 1)
        return self.d_model ** -0.5 * min(n ** -0.5, n * self.warmup ** -1.5)

    def step(self):
        self.step_num += 1
        for g in self.opt.param_groups:
            g["lr"] = self.rate()
        self.opt.step()


def run_epoch(model, batches, criterion, device, scheduler=None, clip=1.0):
    """One pass over `batches`. Trains if a scheduler is given. Returns loss per target token."""
    training = scheduler is not None
    model.train(training)
    total, n_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for src, tgt_in, tgt_out in batches:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)                       # (B, T, V) scores for every position
            # Flatten to (B*T, V) vs (B*T,) so every target token is one classification;
            # <pad> positions are skipped by ignore_index.
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if training:
                scheduler.opt.zero_grad()                     # clear the old gradients
                loss.backward()                               # backpropagation
                nn.utils.clip_grad_norm_(model.parameters(), clip)  # stop exploding gradients
                scheduler.step()                              # set the warmup lr, then update weights
            n = (tgt_out != PAD).sum().item()                 # real (non-pad) tokens in this batch
            total += loss.item() * n
            n_tokens += n
    return total / n_tokens


def ids_to_text(ids, sp):
    ids = [i for i in ids if i not in (PAD, BOS)]
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    return sp.decode(ids)


@torch.no_grad()
def translate(model, sources, sp, cfg, device, beam_size=1):
    """Translate a list of Hinglish sentences. beam_size=1 means greedy decoding."""
    model.eval()
    hyps = []
    if beam_size == 1:
        for i in range(0, len(sources), 128):
            src = pad([src_ids(s, sp, cfg["max_len"]) for s in sources[i:i + 128]]).to(device)
            hyps += [ids_to_text(o, sp) for o in model.greedy_decode(src, cfg["max_decode_len"])]
    else:
        for s in sources:
            src = torch.tensor([src_ids(s, sp, cfg["max_len"])], device=device)
            out = model.beam_search(src, beam_size, cfg["max_decode_len"], cfg["length_penalty"])
            hyps.append(ids_to_text(out, sp))
    return hyps


def train(model, train_pairs, val_pairs, sp, cfg, device, ckpt_path):
    # Label smoothing only while training; validation loss is plain cross-entropy.
    train_crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg["label_smoothing"])
    val_crit = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup"])

    history, best, bad_epochs = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, make_batches(train_pairs, sp, cfg, True, cfg["seed"] + epoch),
                               train_crit, device, scheduler, cfg["clip"])
        val_loss = run_epoch(model, make_batches(val_pairs, sp, cfg, False), val_crit, device)
        # Greedy decoding of the whole val set is the slow part on a CPU, so val BLEU is
        # only computed every few epochs. Checkpointing uses the val loss either way.
        val_bleu = None
        if epoch % cfg["val_bleu_every"] == 0:
            val_hyps = translate(model, [s for s, _ in val_pairs], sp, cfg, device)
            val_bleu = evaluate(val_hyps, [t for _, t in val_pairs])["bleu"]
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4), "val_bleu": val_bleu,
                        "lr": round(scheduler.rate(), 6)})

        # Checkpointing + early stopping: keep only the epoch with the lowest val loss, and
        # stop when it has not improved for `patience` epochs (the model is overfitting).
        improved = val_loss < best
        if improved:
            best, bad_epochs = val_loss, 0
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
        else:
            bad_epochs += 1
        bleu_text = f"val BLEU {val_bleu:5.2f}  " if val_bleu is not None else ""
        print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  "
              f"{bleu_text}{time.time() - start:4.0f}s{'  *saved' if improved else ''}", flush=True)
        if bad_epochs >= cfg["patience"]:
            print(f"Early stopping: no val-loss improvement for {cfg['patience']} epochs.")
            break
    return history


def plot_curves(history, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train (label-smoothed)")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_xlabel("epoch"), ax1.set_ylabel("loss per token"), ax1.legend()
    scored = [h for h in history if h["val_bleu"] is not None]
    ax2.plot([h["epoch"] for h in scored], [h["val_bleu"] for h in scored], "o-", color="tab:green")
    ax2.set_xlabel("epoch"), ax2.set_ylabel("validation BLEU (greedy)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer from scratch (issue #3).")
    parser.add_argument("--config", default="configs/transformer_small.json")
    parser.add_argument("--epochs", type=int, help="override the number of epochs in the config")
    parser.add_argument("--limit", type=int, help="smoke test: use only this many training pairs")
    parser.add_argument("--eval-only", action="store_true", help="skip training, load the checkpoint")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_pairs = read_tsv(SPLIT_DIR / "train.tsv")
    val_pairs = read_tsv(SPLIT_DIR / "val.tsv")
    test_pairs = read_tsv(SPLIT_DIR / "test.tsv")
    # A smoke test writes to checkpoints/smoke/ so it never overwrites the real results.
    out_dir, ckpt_dir, vocab_size = Path("results"), Path("checkpoints"), cfg["vocab_size"]
    if args.limit:
        train_pairs, val_pairs, test_pairs = (train_pairs[: args.limit],
                                              val_pairs[: args.limit // 5], test_pairs[: args.limit // 5])
        out_dir = ckpt_dir = Path("checkpoints/smoke")
        vocab_size = 400  # a few hundred pairs cannot fill a 4000-piece vocabulary
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    ckpt_path = ckpt_dir / f"{name}.pt"

    # Tokenizer from the TRAIN split only.
    sp = get_tokenizer(train_pairs, vocab_size, str(ckpt_dir / f"spm_{vocab_size}"))
    print(f"vocab: {sp.get_piece_size()} joint BPE pieces | pairs: "
          f"{len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    model = TransformerMT(sp.get_piece_size(), cfg, PAD, BOS, EOS, UNK).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not args.eval_only:
        history = train(model, train_pairs, val_pairs, sp, cfg, device, ckpt_path)
        (out_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, f"Transformer from scratch ({name})", out_dir / f"{name}_curves.png")

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    # Test set: used once, at the end.
    sources = [s for s, _ in test_pairs]
    references = [t for _, t in test_pairs]
    greedy = evaluate(translate(model, sources, sp, cfg, device), references)
    beam_hyps = translate(model, sources, sp, cfg, device, beam_size=cfg["beam_size"])
    beam = evaluate(beam_hyps, references)
    print(f"test greedy: BLEU {greedy['bleu']}  chrF {greedy['chrf']}")
    print(f"test beam {cfg['beam_size']}: BLEU {beam['bleu']}  chrF {beam['chrf']}")

    save_result({**beam, "decoding": f"beam{cfg['beam_size']}", "greedy": greedy, "config": cfg,
                 "parameters": sum(p.numel() for p in model.parameters())},
                out_dir / f"{name}.json", model=name)
    n = save_errors(sources, beam_hyps, references, out_dir / f"{name}_errors.csv")
    print(f"wrote {n} rows to {out_dir / (name + '_errors.csv')}")


if __name__ == "__main__":
    main()
