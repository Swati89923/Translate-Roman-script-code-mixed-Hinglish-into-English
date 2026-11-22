"""Tests for the from-scratch Transformer (issue #3). Skipped if torch is missing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
torch = pytest.importorskip("torch")

from models.transformer import TransformerMT  # noqa: E402

CFG = {"d_model": 32, "nhead": 4, "num_layers": 2, "ff_dim": 64, "dropout": 0.0}
PAD, UNK, BOS, EOS = 0, 1, 2, 3
V = 12


def make_batch():
    # Two source sentences of different length (the second is padded).
    src = torch.tensor([[5, 6, 7, EOS], [8, EOS, PAD, PAD]])
    tgt_in = torch.tensor([[BOS, 4, 5], [BOS, 6, PAD]])
    tgt_out = torch.tensor([[4, 5, EOS], [6, EOS, PAD]])
    return src, tgt_in, tgt_out


def test_forward_shape():
    torch.manual_seed(0)
    model = TransformerMT(V, CFG, PAD, BOS, EOS)
    src, tgt_in, _ = make_batch()
    assert model(src, tgt_in).shape == (2, 3, V)


def test_decoder_cannot_see_future_tokens():
    """Changing a later target token must not change the logits at earlier positions."""
    torch.manual_seed(0)
    model = TransformerMT(V, CFG, PAD, BOS, EOS).eval()
    src, tgt_in, _ = make_batch()
    changed = tgt_in.clone()
    changed[0, 2] = 9
    a, b = model(src, tgt_in), model(src, changed)
    assert torch.allclose(a[0, :2], b[0, :2], atol=1e-5)
    assert not torch.allclose(a[0, 2], b[0, 2], atol=1e-5)


def test_source_padding_is_ignored():
    """Extra <pad> at the end of a source sentence must not change the output."""
    torch.manual_seed(0)
    model = TransformerMT(V, CFG, PAD, BOS, EOS).eval()
    src = torch.tensor([[8, EOS]])
    padded = torch.tensor([[8, EOS, PAD, PAD]])
    tgt_in = torch.tensor([[BOS, 6]])
    assert torch.allclose(model(src, tgt_in), model(padded, tgt_in), atol=1e-5)


def test_overfit_one_batch_and_decode():
    """The model must be able to memorise two pairs; greedy and beam then reproduce them."""
    torch.manual_seed(0)
    model = TransformerMT(V, CFG, PAD, BOS, EOS)
    src, tgt_in, tgt_out = make_batch()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    for _ in range(150):
        loss = loss_fn(model(src, tgt_in).reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert model.greedy_decode(src, max_len=6) == [[4, 5, EOS], [6, EOS]]
    assert model.beam_search(src[:1], beam_size=3, max_len=6) == [4, 5, EOS]


def test_block_unk_never_outputs_unk():
    """Even if the model strongly prefers <unk>, block_unk makes decoding pick a real token."""
    torch.manual_seed(0)
    model = TransformerMT(V, {**CFG, "block_unk": True}, PAD, BOS, EOS, UNK).eval()
    with torch.no_grad():
        model.embedding.weight[UNK] = 50.0  # the tied output layer now scores <unk> very high
    src, _, _ = make_batch()
    assert all(UNK not in row for row in model.greedy_decode(src, max_len=5))
    assert UNK not in model.beam_search(src[:1], beam_size=3, max_len=5)
