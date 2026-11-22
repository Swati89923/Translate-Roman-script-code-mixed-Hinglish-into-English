"""Transformer encoder-decoder trained from scratch, issue #3.

Follows Vaswani et al., "Attention Is All You Need", NeurIPS 2017
(https://arxiv.org/abs/1706.03762). No pretrained weights are used anywhere.

  * torch.nn building blocks for the layers (multi-head attention, feed-forward,
    layer norm, residuals), as the issue asks.
  * Our own parts: sinusoidal positional encoding, embedding scaling, one shared
    embedding table for source, target and output layer, the masks, and greedy
    and beam search decoding.

Shapes in the comments: B = batch, S = source length, T = target length,
D = d_model, V = joint subword vocabulary size.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Adds sin/cos position signals so the model knows the order of the tokens.

    Attention by itself has no sense of order, so position i gets a fixed vector:
        PE[i, 2k]   = sin(i / 10000^(2k/D))
        PE[i, 2k+1] = cos(i / 10000^(2k/D))
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)                       # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # A buffer moves with the model to the GPU but is not a trained parameter.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, D)

    def forward(self, x):                                              # x: (B, L, D)
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerMT(nn.Module):
    def __init__(self, vocab_size, cfg, pad_id=0, bos_id=2, eos_id=3, unk_id=None):
        super().__init__()
        d_model = cfg["d_model"]
        self.d_model, self.pad_id, self.bos_id, self.eos_id = d_model, pad_id, bos_id, eos_id
        self.unk_id = unk_id
        self.block_unk = cfg.get("block_unk", False) and unk_id is not None
        # ONE embedding table for encoder input, decoder input and the output layer. The
        # subword vocabulary is joint Hinglish + English (names, hashtags and many English
        # words appear on both sides), and sharing saves parameters on a small dataset.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, cfg["dropout"])
        # norm_first=True ("pre-norm") trains more stably than the paper's post-norm
        # when there is little data.
        enc_layer = nn.TransformerEncoderLayer(d_model, cfg["nhead"], cfg["ff_dim"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, cfg["nhead"], cfg["ff_dim"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg["num_layers"],
                                             norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg["num_layers"], norm=nn.LayerNorm(d_model))
        self._init_weights()

    def _init_weights(self):
        # Xavier init for every weight matrix, as in the original Transformer code.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    def _embed(self, ids):
        # Scale by sqrt(D) so the embeddings are not drowned out by the positional signal.
        return self.pos(self.embedding(ids) * math.sqrt(self.d_model))

    def encode(self, src):
        """src: (B, S) -> memory (B, S, D) and the padding mask (B, S), True = <pad>."""
        src_pad = src == self.pad_id
        return self.encoder(self._embed(src), src_key_padding_mask=src_pad), src_pad

    def decode(self, tgt_in, memory, src_pad):
        """tgt_in: (B, T) decoder input -> logits (B, T, V)."""
        T = tgt_in.size(1)
        # Causal mask: position t may only look at positions <= t (True = blocked).
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        out = self.decoder(self._embed(tgt_in), memory, tgt_mask=causal,
                           tgt_key_padding_mask=(tgt_in == self.pad_id),
                           memory_key_padding_mask=src_pad)
        return F.linear(out, self.embedding.weight)  # output layer tied to the embedding

    def forward(self, src, tgt_in):
        """Training pass: encode the Hinglish once, then score every English position at once.

        Thanks to the causal mask the whole target is processed in parallel (unlike the
        LSTM, which has to go word by word).
        """
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    def _next_log_probs(self, ys, memory, src_pad):
        """Log-probabilities of the next token after the prefixes ys: (B, T) -> (B, V)."""
        logits = self.decode(ys, memory, src_pad)[:, -1]
        if self.block_unk:
            logits[:, self.unk_id] = float("-inf")
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=60):
        """Batched greedy decoding. Returns a list of id lists, each ending at <eos> if reached."""
        memory, src_pad = self.encode(src)
        # Every sentence starts with <bos>; at each step we append the single most likely token.
        ys = torch.full((src.size(0), 1), self.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self._next_log_probs(ys, memory, src_pad).argmax(-1)  # best next token per sentence
            nxt = nxt.masked_fill(done, self.pad_id)   # finished sentences just emit <pad>
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == self.eos_id
            if done.all():
                break
        out = []
        for row in ys[:, 1:].tolist():
            row = [i for i in row if i != self.pad_id]
            out.append(row[: row.index(self.eos_id) + 1] if self.eos_id in row else row)
        return out

    @torch.no_grad()
    def beam_search(self, src, beam_size=5, max_len=60, length_penalty=0.7):
        """Beam search for ONE sentence (src: (1, S)). Returns a list of ids.

        The encoder runs once. At each step all live beams go through the decoder
        as one batch. Hypotheses are ranked by log-probability / length**length_penalty,
        so the search does not simply prefer short outputs.
        """
        assert src.size(0) == 1
        memory, src_pad = self.encode(src)
        beams = [(0.0, [self.bos_id])]
        finished = []

        def norm(score, tokens):
            return score / (len(tokens) - 1) ** length_penalty  # -1: <bos> is not generated

        for _ in range(max_len):
            k = len(beams)
            ys = torch.tensor([b[1] for b in beams], device=src.device)
            log_probs = self._next_log_probs(ys, memory.expand(k, -1, -1), src_pad.expand(k, -1))
            # Score of "beam + next token" = beam's log-prob so far + log-prob of the token.
            scores = torch.tensor([b[0] for b in beams], device=src.device).unsqueeze(1) + log_probs
            vocab_size = log_probs.size(1)
            # Keep the beam_size best extensions over ALL beams (k * V candidates).
            top_scores, top_idx = scores.view(-1).topk(beam_size)

            new_beams = []
            for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                b, tok = divmod(idx, vocab_size)
                tokens = beams[b][1] + [tok]
                if tok == self.eos_id:
                    finished.append((norm(score, tokens), tokens))
                else:
                    new_beams.append((score, tokens))
            beams = new_beams
            # Stop once we have beam_size finished hypotheses or nothing is left to extend.
            if len(finished) >= beam_size or not beams:
                break

        if not finished:  # nothing reached <eos> within max_len
            finished = [(norm(s, t), t) for s, t in beams]
        return max(finished, key=lambda x: x[0])[1][1:]
