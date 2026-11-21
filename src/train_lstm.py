"""Train and evaluate the LSTM + Bahdanau attention baseline (issue #2).

Usage, from the repo root after download_data.py and preprocess.py:
    python src/train_lstm.py                          # train, then test with greedy + beam search
    python src/train_lstm.py --eval-only              # reuse checkpoints/lstm_attn.pt, only test
    python src/train_lstm.py --epochs 1 --limit 300   # quick smoke test (writes to checkpoints/smoke/)

Hyperparameters live in configs/lstm_attn.json.

Outputs:
    checkpoints/lstm_attn.pt               best model by validation loss (gitignored)
    results/lstm_attn.json                 test BLEU / chrF (beam search, plus greedy for comparison)
    results/lstm_attn_history.json         loss and val BLEU per epoch
    results/lstm_attn_curves.png           training curves
    results/lstm_attn_attention_{1,2,3}.png  attention heatmaps for 3 test sentences
    results/lstm_attn_errors.csv           100 worst test translations for the error analysis
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import evaluate, save_errors, save_result
from models.lstm_attn import Seq2Seq
from preprocess import read_tsv
from vocab import Vocab

NAME = "lstm_attn"
SPLIT_DIR = Path("data/processed")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_ids(text, vocab, max_len):
    """Sentence -> [<bos>, ..., <eos>], cut to at most max_len ids."""
    return [vocab.bos_id] + vocab.encode(text, add_bos_eos=False)[: max_len - 2] + [vocab.eos_id]


def pad(seqs, pad_id):
    out = torch.full((len(seqs), max(len(s) for s in seqs)), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def make_loader(pairs, vs, vt, cfg, shuffle, seed=0):
    data = [(to_ids(s, vs, cfg["max_len"]), to_ids(t, vt, cfg["max_len"])) for s, t in pairs]

    def collate(batch):
        src, tgt = zip(*batch)
        return pad(src, vs.pad_id), torch.tensor([len(x) for x in src]), pad(tgt, vt.pad_id)

    gen = torch.Generator().manual_seed(seed)  # same batch order on every run
    return DataLoader(data, batch_size=cfg["batch_size"], shuffle=shuffle,
                      collate_fn=collate, generator=gen)


def run_epoch(model, loader, criterion, device, optimizer=None, clip=1.0, teacher_forcing=1.0):
    """One pass over `loader`. Trains if an optimizer is given. Returns loss per target token."""
    training = optimizer is not None
    model.train(training)
    total, n_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for src, src_len, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, src_len, tgt, teacher_forcing if training else 1.0)
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            n = (gold != model.pad_id).sum().item()
            total += loss.item() * n
            n_tokens += n
    return total / n_tokens


@torch.no_grad()
def translate(model, sources, vs, vt, cfg, device, beam_size=1):
    """Translate a list of Hinglish sentences. beam_size=1 means greedy decoding."""
    model.eval()
    hyps = []
    if beam_size == 1:
        for i in range(0, len(sources), 128):
            ids = [to_ids(s, vs, cfg["max_len"]) for s in sources[i:i + 128]]
            src_len = torch.tensor([len(x) for x in ids])
            out, _ = model.greedy_decode(pad(ids, vs.pad_id).to(device), src_len, cfg["max_decode_len"])
            hyps += [vt.decode(o) for o in out]
    else:
        for s in sources:
            ids = to_ids(s, vs, cfg["max_len"])
            src = torch.tensor([ids], device=device)
            out, _ = model.beam_search(src, torch.tensor([len(ids)]), beam_size,
                                       cfg["max_decode_len"], cfg["length_penalty"])
            hyps.append(vt.decode(out))
    return hyps


def train(model, train_pairs, val_pairs, vs, vt, cfg, device, ckpt_path):
    train_loader = make_loader(train_pairs, vs, vt, cfg, shuffle=True, seed=cfg["seed"])
    val_loader = make_loader(val_pairs, vs, vt, cfg, shuffle=False)
    # Label smoothing only while training; validation loss is plain cross-entropy.
    train_crit = nn.CrossEntropyLoss(ignore_index=vt.pad_id, label_smoothing=cfg["label_smoothing"])
    val_crit = nn.CrossEntropyLoss(ignore_index=vt.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    history, best, bad_epochs = [], float("inf"), 0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, train_crit, device, optimizer,
                               cfg["clip"], cfg["teacher_forcing"])
        val_loss = run_epoch(model, val_loader, val_crit, device)
        val_hyps = translate(model, [s for s, _ in val_pairs], vs, vt, cfg, device)
        val_bleu = evaluate(val_hyps, [t for _, t in val_pairs])["bleu"]
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4), "val_bleu": val_bleu})

        improved = val_loss < best
        if improved:
            best, bad_epochs = val_loss, 0
            torch.save({"model": model.state_dict(), "config": cfg,
                        "src_itos": vs.itos, "tgt_itos": vt.itos}, ckpt_path)
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  "
              f"val BLEU {val_bleu:5.2f}  {time.time() - start:4.0f}s{'  *saved' if improved else ''}")
        if bad_epochs >= cfg["patience"]:
            print(f"Early stopping: no val-loss improvement for {cfg['patience']} epochs.")
            break
    return history


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train (label-smoothed)")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_xlabel("epoch"), ax1.set_ylabel("loss per token"), ax1.legend()
    ax2.plot(epochs, [h["val_bleu"] for h in history], color="tab:green")
    ax2.set_xlabel("epoch"), ax2.set_ylabel("validation BLEU (greedy)")
    fig.suptitle("LSTM + Bahdanau attention")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_attention(model, sentence, vs, vt, cfg, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    ids = to_ids(sentence, vs, cfg["max_len"])
    out, attn = model.greedy_decode(torch.tensor([ids], device=device),
                                    torch.tensor([len(ids)]), cfg["max_decode_len"])
    out = out[0]
    weights = attn[0, : len(out)].cpu().numpy()  # (output words, source words)
    fig, ax = plt.subplots(figsize=(0.5 * len(ids) + 2, 0.5 * len(out) + 2))
    ax.imshow(weights, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ids)), [vs.itos[i] for i in ids], rotation=60, ha="right")
    ax.set_yticks(range(len(out)), [vt.itos[i] for i in out])
    ax.set_xlabel("Hinglish source"), ax.set_ylabel("English output")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM + Bahdanau attention (issue #2).")
    parser.add_argument("--config", default="configs/lstm_attn.json")
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
    out_dir, ckpt_dir = Path("results"), Path("checkpoints")
    if args.limit:
        train_pairs, val_pairs, test_pairs = (train_pairs[: args.limit],
                                              val_pairs[: args.limit // 5], test_pairs[: args.limit // 5])
        out_dir = ckpt_dir = Path("checkpoints/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{NAME}.pt"

    # Vocabularies from the TRAIN split only.
    vs = Vocab([s for s, _ in train_pairs], min_freq=cfg["min_freq"])
    vt = Vocab([t for _, t in train_pairs], min_freq=cfg["min_freq"])
    print(f"vocab: {len(vs)} Hinglish, {len(vt)} English | pairs: "
          f"{len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    model = Seq2Seq(len(vs), len(vt), cfg, vs.pad_id, vt.bos_id, vt.eos_id).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not args.eval_only:
        history = train(model, train_pairs, val_pairs, vs, vt, cfg, device, ckpt_path)
        (out_dir / f"{NAME}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plot_curves(history, out_dir / f"{NAME}_curves.png")

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt["src_itos"] != vs.itos or ckpt["tgt_itos"] != vt.itos:
        raise SystemExit("Checkpoint vocabulary does not match the current train split.")
    model.load_state_dict(ckpt["model"])

    # Test set: used once, at the end.
    sources = [s for s, _ in test_pairs]
    references = [t for _, t in test_pairs]
    greedy = evaluate(translate(model, sources, vs, vt, cfg, device), references)
    beam_hyps = translate(model, sources, vs, vt, cfg, device, beam_size=cfg["beam_size"])
    beam = evaluate(beam_hyps, references)
    print(f"test greedy: BLEU {greedy['bleu']}  chrF {greedy['chrf']}")
    print(f"test beam {cfg['beam_size']}: BLEU {beam['bleu']}  chrF {beam['chrf']}")

    save_result({**beam, "decoding": f"beam{cfg['beam_size']}", "greedy": greedy, "config": cfg},
                out_dir / f"{NAME}.json", model=NAME)
    n = save_errors(sources, beam_hyps, references, out_dir / f"{NAME}_errors.csv")
    print(f"wrote {n} rows to {out_dir / (NAME + '_errors.csv')}")

    # Attention heatmaps for 3 short test sentences (short ones are readable).
    examples = [s for s in sources if 5 <= len(s.split()) <= 10][:3]
    for i, s in enumerate(examples, 1):
        plot_attention(model, s, vs, vt, cfg, device, out_dir / f"{NAME}_attention_{i}.png")
    print(f"saved plots to {out_dir}/")


if __name__ == "__main__":
    main()
