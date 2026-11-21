"""Tokenisation helpers shared by the LSTM (#2) and Transformer (#3) tasks.

Two options are provided:
  * `Vocab`  - simple word-level vocabulary (good enough for the LSTM baseline).
  * `train_sentencepiece` - subword (BPE) model, better for the small, noisy,
    code-mixed PHINC corpus where many Hinglish spellings are rare.

The vocabulary must be built from the TRAIN split only. Building it from val or
test would leak information and inflate the scores.
"""

from collections import Counter

# Special tokens always occupy the first ids, in this fixed order.
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


def tokenize(text: str):
    """Very simple whitespace tokenizer (text is already cleaned/lowercased)."""
    return text.split()


class Vocab:
    """Maps tokens <-> integer ids."""

    def __init__(self, sentences, min_freq: int = 2, max_size: int = 30000):
        # Count how often each token occurs in the training sentences.
        counts = Counter(tok for s in sentences for tok in tokenize(s))
        self.itos = list(SPECIALS)  # id -> token
        # most_common() sorts by frequency, so the cut at max_size keeps frequent words.
        for tok, c in counts.most_common():
            if c < min_freq or len(self.itos) >= max_size:
                break
            self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token -> id

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str, add_bos_eos: bool = True):
        """Sentence -> list of ids. Unknown words become <unk>."""
        unk = self.stoi[UNK]
        ids = [self.stoi.get(t, unk) for t in tokenize(text)]
        return [self.bos_id] + ids + [self.eos_id] if add_bos_eos else ids

    def decode(self, ids):
        """List of ids -> sentence. Stops at <eos> and skips special tokens."""
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            tok = self.itos[i]
            if tok not in SPECIALS:
                out.append(tok)
        return " ".join(out)


def train_sentencepiece(sentences, model_prefix: str, vocab_size: int = 8000):
    """Train a joint BPE SentencePiece model on `sentences` (train split only).

    Writes `<model_prefix>.model` and `<model_prefix>.vocab` (both gitignored).
    Requires `pip install sentencepiece`.
    """
    import sentencepiece as spm  # imported here so the rest of the file needs no extra install

    import tempfile
    # SentencePiece reads its training text from a file, so dump the sentences to a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(sentences))
        path = f.name
    spm.SentencePieceTrainer.train(
        input=path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,  # same special ids as the word-level Vocab
    )
    return f"{model_prefix}.model"
