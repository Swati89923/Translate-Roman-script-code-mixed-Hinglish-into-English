"""Tests for the LSTM + Bahdanau attention model (issue #2). Skipped if torch is missing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
torch = pytest.importorskip("torch")

from models.lstm_attn import Seq2Seq  # noqa: E402

CFG = {"emb_dim": 16, "enc_hid_dim": 16, "dec_hid_dim": 32, "attn_dim": 16, "dropout": 0.0}
PAD, BOS, EOS = 0, 2, 3


def make_batch():
    # Two source sentences of different length (the second is padded).
    src = torch.tensor([[BOS, 5, 6, 7, EOS], [BOS, 8, EOS, PAD, PAD]])
    src_len = torch.tensor([5, 3])
    tgt = torch.tensor([[BOS, 4, 5, EOS], [BOS, 6, EOS, PAD]])
    return src, src_len, tgt


def test_forward_shape():
    torch.manual_seed(0)
    model = Seq2Seq(10, 12, CFG, PAD, BOS, EOS)
    src, src_len, tgt = make_batch()
    logits = model(src, src_len, tgt)
    assert logits.shape == (2, 3, 12)  # predicts tgt[:, 1:]


def test_attention_ignores_padding():
    torch.manual_seed(0)
    model = Seq2Seq(10, 12, CFG, PAD, BOS, EOS).eval()
    src, src_len, _ = make_batch()
    _, attn = model.greedy_decode(src, src_len, max_len=4)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))   # weights sum to 1
    assert attn[1, :, 3:].abs().max() < 1e-6                            # no weight on <pad>


def test_overfit_one_batch_and_decode():
    """The model must be able to memorise two pairs; greedy and beam then reproduce them."""
    torch.manual_seed(0)
    model = Seq2Seq(10, 12, CFG, PAD, BOS, EOS)
    src, src_len, tgt = make_batch()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    for _ in range(200):
        logits = model(src, src_len, tgt)
        loss = loss_fn(logits.reshape(-1, 12), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    ids, _ = model.greedy_decode(src, src_len, max_len=6)
    assert ids == [[4, 5, EOS], [6, EOS]]
    beam_ids, beam_attn = model.beam_search(src[:1], src_len[:1], beam_size=3, max_len=6)
    assert beam_ids == [4, 5, EOS]
    assert beam_attn.shape == (3, 5)


def test_block_unk_never_outputs_unk():
    """Even if the model strongly prefers <unk>, block_unk makes decoding pick a real word."""
    UNK = 1
    torch.manual_seed(0)
    model = Seq2Seq(10, 12, {**CFG, "block_unk": True}, PAD, BOS, EOS, UNK).eval()
    with torch.no_grad():
        model.decoder.out.bias[UNK] = 100.0  # make <unk> by far the most likely word
    src, src_len, _ = make_batch()
    ids, _ = model.greedy_decode(src, src_len, max_len=5)
    assert all(UNK not in row for row in ids)
    beam_ids, _ = model.beam_search(src[:1], src_len[:1], beam_size=3, max_len=5)
    assert UNK not in beam_ids

    model.block_unk = False  # without blocking, the same model does output <unk>
    ids, _ = model.greedy_decode(src, src_len, max_len=5)
    assert ids[0][0] == UNK
