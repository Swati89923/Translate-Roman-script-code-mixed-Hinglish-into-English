"""Seq2seq LSTM with Bahdanau (additive) attention, issue #2.

Follows Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly
Learning to Align and Translate", ICLR 2015 (https://arxiv.org/abs/1409.0473):

  * Encoder: bidirectional LSTM over the Hinglish source. Every source word gets
    an annotation h_j = [forward state; backward state].
  * Attention: e_ij = v^T tanh(W_dec s_{i-1} + W_enc h_j), alpha_ij = softmax_j(e_ij),
    context c_i = sum_j alpha_ij h_j. Padding positions get zero weight.
  * Decoder: LSTM that reads [previous English word; context] and predicts the
    next English word from [decoder state; context; previous word].

The paper used GRU-like units; we use LSTM cells as the issue asks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dec_hid_dim, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        # The decoder's first state is computed from the encoder's last states.
        self.fc_h = nn.Linear(2 * hid_dim, dec_hid_dim)
        self.fc_c = nn.Linear(2 * hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """src: (B, S) ids, src_len: (B,) true lengths -> outputs (B, S, 2H), (h0, c0)."""
        emb = self.dropout(self.embedding(src))
        # Packing makes the LSTM skip the padding, so the backward direction
        # starts at the real last word and not at a <pad>.
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # h, c: (2, B, H) = [forward last, backward last]
        h0 = torch.tanh(self.fc_h(torch.cat([h[0], h[1]], dim=1)))
        c0 = torch.tanh(self.fc_c(torch.cat([c[0], c[1]], dim=1)))
        return out, (h0, c0)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_proj, enc_out, mask):
        """
        dec_hidden: (B, D) previous decoder state s_{i-1}
        enc_proj:   (B, S, A) W_enc h_j, computed once per sentence
        enc_out:    (B, S, E) encoder annotations h_j
        mask:       (B, S) True for real words, False for padding
        returns context (B, E) and attention weights (B, S)
        """
        energy = self.v(torch.tanh(enc_proj + self.W_dec(dec_hidden).unsqueeze(1))).squeeze(2)
        energy = energy.masked_fill(~mask, -1e9)
        weights = F.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, dec_hid_dim, attn_dim, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, dec_hid_dim, attn_dim)
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_hid_dim)
        self.out = nn.Linear(dec_hid_dim + enc_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(self, y_prev, state, enc_out, enc_proj, mask):
        """One decoding step: previous word ids (B,) -> logits (B, V), new state, attention (B, S)."""
        emb = self.dropout(self.embedding(y_prev))
        h, c = state
        context, weights = self.attention(h, enc_proj, enc_out, mask)
        h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
        logits = self.out(self.dropout(torch.cat([h, context, emb], dim=1)))
        return logits, (h, c), weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg, pad_id=0, bos_id=2, eos_id=3, unk_id=1):
        super().__init__()
        enc_dim = 2 * cfg["enc_hid_dim"]
        self.encoder = Encoder(src_vocab_size, cfg["emb_dim"], cfg["enc_hid_dim"],
                               cfg["dec_hid_dim"], cfg["dropout"], pad_id)
        self.decoder = Decoder(tgt_vocab_size, cfg["emb_dim"], enc_dim, cfg["dec_hid_dim"],
                               cfg["attn_dim"], cfg["dropout"], pad_id)
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        # When True, decoding never outputs <unk> (it picks the best real word instead).
        # About 12% of the English training words are rare and become <unk>, so without
        # this the model learns to write <unk> whenever it is unsure. Training is unchanged.
        self.block_unk = cfg.get("block_unk", False)

    def _decode_logits(self, logits):
        """Logits used at decoding time (greedy and beam search)."""
        if self.block_unk:
            logits = logits.clone()
            logits[:, self.unk_id] = float("-inf")
        return logits

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != self.pad_id
        enc_proj = self.decoder.attention.W_enc(enc_out)  # same for every step, so compute once
        return enc_out, state, mask, enc_proj

    def forward(self, src, src_len, tgt, teacher_forcing=1.0):
        """Training pass. tgt = <bos> w1 ... wn <eos> (padded).

        Returns logits (B, T-1, V) that predict tgt[:, 1:]. With probability
        `teacher_forcing` the gold previous word is fed in, otherwise the
        model's own prediction.
        """
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        y = tgt[:, 0]
        all_logits = []
        for t in range(1, tgt.size(1)):
            logits, state, _ = self.decoder.step(y, state, enc_out, enc_proj, mask)
            all_logits.append(logits)
            if teacher_forcing >= 1.0 or torch.rand(1).item() < teacher_forcing:
                y = tgt[:, t]
            else:
                y = logits.argmax(-1)
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=60):
        """Batched greedy decoding.

        Returns (ids, attn): ids is a list (one per sentence) of output ids that
        ends with <eos> if one was produced; attn is (B, T, S).
        """
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        y = torch.full((src.size(0),), self.bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        outs, attns = [], []
        for _ in range(max_len):
            logits, state, weights = self.decoder.step(y, state, enc_out, enc_proj, mask)
            y = self._decode_logits(logits).argmax(-1)
            outs.append(y)
            attns.append(weights)
            finished |= y == self.eos_id
            if finished.all():
                break
        ids = []
        for row in torch.stack(outs, dim=1).tolist():
            ids.append(row[: row.index(self.eos_id) + 1] if self.eos_id in row else row)
        return ids, torch.stack(attns, dim=1)

    @torch.no_grad()
    def beam_search(self, src, src_len, beam_size=5, max_len=60, length_penalty=0.7):
        """Beam search for ONE sentence (src: (1, S)).

        Hypotheses are ranked by log-probability / length**length_penalty, so
        the search does not simply prefer short outputs.
        Returns (ids, attn) with attn of shape (T, S).
        """
        enc_out, (h, c), mask, enc_proj = self.encode(src, src_len)
        # Each beam: (log-prob, tokens, (h, c), list of attention rows)
        beams = [(0.0, [self.bos_id], (h, c), [])]
        finished = []

        def norm(score, tokens):
            return score / (len(tokens) - 1) ** length_penalty  # -1: <bos> is not generated

        for _ in range(max_len):
            k = len(beams)
            y = torch.tensor([b[1][-1] for b in beams], device=src.device)
            h = torch.cat([b[2][0] for b in beams])
            c = torch.cat([b[2][1] for b in beams])
            logits, (h, c), weights = self.decoder.step(
                y, (h, c), enc_out.expand(k, -1, -1), enc_proj.expand(k, -1, -1), mask.expand(k, -1)
            )
            log_probs = F.log_softmax(self._decode_logits(logits), dim=-1)
            scores = torch.tensor([b[0] for b in beams], device=src.device).unsqueeze(1) + log_probs
            vocab_size = log_probs.size(1)
            top_scores, top_idx = scores.view(-1).topk(beam_size)

            new_beams = []
            for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                b, tok = divmod(idx, vocab_size)
                tokens = beams[b][1] + [tok]
                attn = beams[b][3] + [weights[b]]
                if tok == self.eos_id:
                    finished.append((norm(score, tokens), tokens, attn))
                else:
                    new_beams.append((score, tokens, (h[b:b + 1], c[b:b + 1]), attn))
            beams = new_beams
            # Stop once we have beam_size finished hypotheses or nothing is left to extend.
            if len(finished) >= beam_size or not beams:
                break

        if not finished:  # nothing reached <eos> within max_len
            finished = [(norm(s, t), t, a) for s, t, _, a in beams]
        _, tokens, attn = max(finished, key=lambda x: x[0])
        return tokens[1:], torch.stack(attn)
