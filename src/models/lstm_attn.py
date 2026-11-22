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

Shape names used in the comments below:
    B = batch size, S = source length, T = target length, V = vocabulary size,
    E = encoder output size (2 * enc_hid_dim), D = decoder hidden size, A = attention size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dec_hid_dim, dropout, pad_id):
        super().__init__()
        # padding_idx keeps the <pad> embedding at zero and never updates it,
        # so padding carries no information into the LSTM.
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # Bidirectional: one LSTM reads left-to-right, one right-to-left. Each word's
        # annotation then sees the words before AND after it, which helps with
        # Hinglish where word order differs from English.
        # batch_first=True means tensors are (B, S, ...) instead of (S, B, ...).
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        # The decoder's first state is computed from the encoder's last states.
        # The encoder gives 2 * hid_dim (forward + backward) but the decoder has
        # dec_hid_dim units, so a linear layer maps one size to the other.
        self.fc_h = nn.Linear(2 * hid_dim, dec_hid_dim)
        self.fc_c = nn.Linear(2 * hid_dim, dec_hid_dim)
        # Dropout on the embeddings reduces overfitting; the dataset is small (~11k pairs).
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """src: (B, S) ids, src_len: (B,) true lengths -> outputs (B, S, 2H), (h0, c0)."""
        emb = self.dropout(self.embedding(src))
        # Packing makes the LSTM skip the padding, so the backward direction
        # starts at the real last word and not at a <pad>.
        # pack_padded_sequence needs the lengths on the CPU; enforce_sorted=False
        # lets us keep the batch in its original order instead of sorting by length.
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        # Unpack back to a normal padded tensor. total_length keeps the same S as
        # `src`, so the attention mask (built from src) has a matching shape.
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # h, c: (2, B, H) = [forward last, backward last]
        # Concatenate both directions and squash with tanh to get the decoder's
        # initial hidden and cell state (same idea as the paper's s_0 = tanh(W_s h_1)).
        h0 = torch.tanh(self.fc_h(torch.cat([h[0], h[1]], dim=1)))
        c0 = torch.tanh(self.fc_c(torch.cat([c[0], c[1]], dim=1)))
        return out, (h0, c0)


class BahdanauAttention(nn.Module):
    """Additive attention: score(s, h) = v^T tanh(W_dec s + W_enc h)."""

    def __init__(self, enc_dim, dec_hid_dim, attn_dim):
        super().__init__()
        # Encoder and decoder states have different sizes, so both are projected
        # into the same attn_dim space before they are added.
        # W_enc has no bias because W_dec already adds one; two biases would be redundant.
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hid_dim, attn_dim)
        # v turns each attn_dim vector into a single score per source word.
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_proj, enc_out, mask):
        """
        dec_hidden: (B, D) previous decoder state s_{i-1}
        enc_proj:   (B, S, A) W_enc h_j, computed once per sentence
        enc_out:    (B, S, E) encoder annotations h_j
        mask:       (B, S) True for real words, False for padding
        returns context (B, E) and attention weights (B, S)
        """
        # unsqueeze(1) turns (B, A) into (B, 1, A) so it broadcasts over all S source
        # words; squeeze(2) removes the last size-1 dimension -> energy is (B, S).
        energy = self.v(torch.tanh(enc_proj + self.W_dec(dec_hidden).unsqueeze(1))).squeeze(2)
        # Padding must never receive attention. A very large negative score
        # becomes exp(-1e9) = 0 after the softmax. -1e9 instead of -inf avoids NaN
        # if a row were ever fully masked.
        energy = energy.masked_fill(~mask, -1e9)
        # Softmax over the source positions: the weights of one sentence sum to 1.
        weights = F.softmax(energy, dim=1)
        # Context = weighted sum of encoder states. bmm does it for the whole batch:
        # (B, 1, S) x (B, S, E) -> (B, 1, E) -> squeeze -> (B, E).
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_dim, dec_hid_dim, attn_dim, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, dec_hid_dim, attn_dim)
        # LSTMCell (one time step) instead of nn.LSTM (whole sequence), because the
        # input at each step depends on the attention context of that same step,
        # so we have to run the decoder step by step in a loop.
        # Input = previous word embedding + context vector.
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_hid_dim)
        # Output layer sees the new state, the context and the previous word, as in
        # the paper; giving it the context directly makes copying source words easier.
        self.out = nn.Linear(dec_hid_dim + enc_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(self, y_prev, state, enc_out, enc_proj, mask):
        """One decoding step: previous word ids (B,) -> logits (B, V), new state, attention (B, S)."""
        emb = self.dropout(self.embedding(y_prev))
        h, c = state
        # Attention uses the PREVIOUS decoder state s_{i-1} (Bahdanau style), so the
        # model decides where to look before it produces the next word.
        context, weights = self.attention(h, enc_proj, enc_out, mask)
        h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
        # Logits are raw scores (no softmax here): CrossEntropyLoss applies
        # log-softmax itself, and argmax gives the same word with or without softmax.
        logits = self.out(self.dropout(torch.cat([h, context, emb], dim=1)))
        return logits, (h, c), weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg, pad_id=0, bos_id=2, eos_id=3, unk_id=1):
        super().__init__()
        # Bidirectional encoder -> each annotation is forward + backward = 2 * enc_hid_dim.
        enc_dim = 2 * cfg["enc_hid_dim"]
        self.encoder = Encoder(src_vocab_size, cfg["emb_dim"], cfg["enc_hid_dim"],
                               cfg["dec_hid_dim"], cfg["dropout"], pad_id)
        self.decoder = Decoder(tgt_vocab_size, cfg["emb_dim"], enc_dim, cfg["dec_hid_dim"],
                               cfg["attn_dim"], cfg["dropout"], pad_id)
        # Default ids match SPECIALS in vocab.py: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        # When True, decoding never outputs <unk> (it picks the best real word instead).
        # About 12% of the English training words are rare and become <unk>, so without
        # this the model learns to write <unk> whenever it is unsure. Training is unchanged.
        self.block_unk = cfg.get("block_unk", False)

    def _decode_logits(self, logits):
        """Logits used at decoding time (greedy and beam search)."""
        if self.block_unk:
            # clone() so we do not modify the tensor the caller still holds.
            logits = logits.clone()
            # -inf -> probability exactly 0 after softmax, so <unk> can never win.
            logits[:, self.unk_id] = float("-inf")
        return logits

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        # True where there is a real token, False at <pad>; used by the attention.
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
        # The first decoder input is <bos> for every sentence.
        y = tgt[:, 0]
        all_logits = []
        # Step t predicts tgt[:, t]; there is no step for <bos> itself, hence T-1 outputs.
        for t in range(1, tgt.size(1)):
            logits, state, _ = self.decoder.step(y, state, enc_out, enc_proj, mask)
            all_logits.append(logits)
            # Teacher forcing: feed the correct word so early mistakes do not
            # derail the rest of the sentence during training (faster, more stable).
            # The `>= 1.0` check skips drawing a random number when it is always on.
            if teacher_forcing >= 1.0 or torch.rand(1).item() < teacher_forcing:
                y = tgt[:, t]
            else:
                y = logits.argmax(-1)
        # List of T-1 tensors (B, V) -> one tensor (B, T-1, V).
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()  # no gradients needed when translating: saves memory and time
    def greedy_decode(self, src, src_len, max_len=60):
        """Batched greedy decoding.

        Returns (ids, attn): ids is a list (one per sentence) of output ids that
        ends with <eos> if one was produced; attn is (B, T, S).
        """
        enc_out, state, mask, enc_proj = self.encode(src, src_len)
        y = torch.full((src.size(0),), self.bos_id, dtype=torch.long, device=src.device)
        # Tracks which sentences already produced <eos>.
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        outs, attns = [], []
        # max_len is a safety limit in case the model never produces <eos>.
        for _ in range(max_len):
            logits, state, weights = self.decoder.step(y, state, enc_out, enc_proj, mask)
            # Greedy: always take the single most likely next word.
            y = self._decode_logits(logits).argmax(-1)
            outs.append(y)
            attns.append(weights)
            finished |= y == self.eos_id
            # Stop early once every sentence in the batch has ended.
            if finished.all():
                break
        ids = []
        # Sentences that ended early still got words generated after their <eos>
        # (the batch kept running), so cut each one right after its first <eos>.
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

        # Every extra word adds a negative log-probability, so raw scores always
        # favour short outputs. Dividing by length**length_penalty corrects that.
        def norm(score, tokens):
            return score / (len(tokens) - 1) ** length_penalty  # -1: <bos> is not generated

        for _ in range(max_len):
            k = len(beams)
            # Run all live beams as one batch of size k instead of one by one.
            y = torch.tensor([b[1][-1] for b in beams], device=src.device)
            h = torch.cat([b[2][0] for b in beams])
            c = torch.cat([b[2][1] for b in beams])
            # expand() repeats the single encoded sentence k times without copying memory.
            logits, (h, c), weights = self.decoder.step(
                y, (h, c), enc_out.expand(k, -1, -1), enc_proj.expand(k, -1, -1), mask.expand(k, -1)
            )
            # Log-probabilities so the score of a sequence is a SUM (a product of
            # probabilities would quickly underflow to 0).
            log_probs = F.log_softmax(self._decode_logits(logits), dim=-1)
            # Score of every possible extension: (k, V) = old beam score + next word.
            scores = torch.tensor([b[0] for b in beams], device=src.device).unsqueeze(1) + log_probs
            vocab_size = log_probs.size(1)
            # Flatten (k, V) so one topk picks the best beam_size extensions overall.
            top_scores, top_idx = scores.view(-1).topk(beam_size)

            new_beams = []
            for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                # Undo the flattening: which beam it came from and which word it adds.
                b, tok = divmod(idx, vocab_size)
                tokens = beams[b][1] + [tok]
                attn = beams[b][3] + [weights[b]]
                if tok == self.eos_id:
                    # Finished hypotheses are compared with the length-normalised score.
                    finished.append((norm(score, tokens), tokens, attn))
                else:
                    # Keep this beam's own state (slice b:b+1 keeps the batch dimension).
                    new_beams.append((score, tokens, (h[b:b + 1], c[b:b + 1]), attn))
            beams = new_beams
            # Stop once we have beam_size finished hypotheses or nothing is left to extend.
            if len(finished) >= beam_size or not beams:
                break

        if not finished:  # nothing reached <eos> within max_len
            finished = [(norm(s, t), t, a) for s, t, _, a in beams]
        _, tokens, attn = max(finished, key=lambda x: x[0])
        # Drop the leading <bos>; attention rows become one (T, S) tensor.
        return tokens[1:], torch.stack(attn)
