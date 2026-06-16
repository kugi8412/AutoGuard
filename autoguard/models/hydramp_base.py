#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch HydrAMP base model - ported from marmarmarmar/pytorch-hydramp.

Original: Szymczak, P., Możejko, M., Grzegorzek, T. et al.
"Discovering highly potent antimicrobial peptides with deep generative model HydrAMP."
Nat Commun 14, 1453 (2023). https://doi.org/10.1038/s41467-023-36994-z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HydrAMPGRU(nn.Module):
    """Custom GRU implementation matching HydrAMP's Keras GRU behavior."""

    def __init__(self, units=66, input_units=66, output_len=25):
        super().__init__()
        self.output_len = output_len
        self.units = units
        self.input_units = input_units
        self.kernel = nn.Parameter(torch.empty(input_units, units * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, units * 3))
        self.bias = nn.Parameter(torch.zeros(units * 3))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)

        for i in range(3):
            nn.init.orthogonal_(self.recurrent_kernel[:, i * self.units:(i + 1) * self.units])

        nn.init.zeros_(self.bias)

    def cell_forward(self, inputs, state):
        h_tm1 = state
        matrix_x = torch.matmul(inputs, self.kernel) + self.bias
        x_z, x_r, x_h = torch.split(matrix_x, self.units, dim=-1)

        matrix_inner = torch.matmul(h_tm1, self.recurrent_kernel[:, :self.units * 2])
        recurrent_z, recurrent_r = torch.split(matrix_inner, self.units, dim=-1)

        z = torch.sigmoid(x_z + recurrent_z)
        r = torch.sigmoid(x_r + recurrent_r)

        recurrent_h = torch.matmul(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:]
        )
        hh = torch.tanh(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        return h, h

    def forward(self, input_, state=None):
        if input_ is None:
            input_ = torch.zeros((state.shape[0], self.input_units), device=state.device)
        if state is None:
            state = torch.zeros((input_.shape[0], self.units), device=input_.device)

        current_output = input_
        current_state = state
        outputs = []
        for _ in range(self.output_len):
            current_output, current_state = self.cell_forward(current_output, current_state)
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)

    def forward_on_sequence(self, input_, state=None):
        if state is None:
            state = torch.zeros((input_.shape[0], self.units), device=input_.device)
        current_state = state
        outputs = []
        for i in range(input_.shape[1]):
            current_output, current_state = self.cell_forward(input_[:, i], current_state)
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)


class HydrAMPEncoder(nn.Module):
    """Bidirectional GRU encoder from HydrAMP, ported to PyTorch."""

    def __init__(self, vocab_size=21, embedding_dim=100, hidden_dim=128, latent_dim=64, max_len=25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru1_f = HydrAMPGRU(input_units=embedding_dim, units=hidden_dim, output_len=max_len)
        self.gru1_r = HydrAMPGRU(input_units=embedding_dim, units=hidden_dim, output_len=max_len)
        self.gru2_f = HydrAMPGRU(input_units=hidden_dim * 2, units=hidden_dim, output_len=max_len)
        self.gru2_r = HydrAMPGRU(input_units=hidden_dim * 2, units=hidden_dim, output_len=max_len)
        self.mean_linear = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_linear = nn.Linear(hidden_dim * 2, latent_dim)

    def _encode_backbone(self, x):
        """Run the stacked bi-GRU and return the two directional outputs.

        Returns (gru2_f_output, gru2_r_output) each [batch, max_len,
        hidden_dim]. gru2_r_output is in flipped (reverse-processing)
        time order, exactly as produced by the reverse GRU.
        """
        embeddings = self.embedding(x)
        gru1_f_output = self.gru1_f.forward_on_sequence(embeddings)
        gru1_r_output = self.gru1_r.forward_on_sequence(torch.flip(embeddings, (1,)))
        gru_1_output = torch.cat([gru1_f_output, torch.flip(gru1_r_output, (1,))], dim=-1)

        gru2_f_output = self.gru2_f.forward_on_sequence(gru_1_output)
        gru2_r_output = self.gru2_r.forward_on_sequence(torch.flip(gru_1_output, (1,)))
        return gru2_f_output, gru2_r_output

    def forward(self, x):
        # Pooled (whole-sequence) latent: full forward + full reverse summary.
        gru2_f_output, gru2_r_output = self._encode_backbone(x)
        gru_2_output = torch.cat([gru2_f_output[:, -1], gru2_r_output[:, -1]], dim=-1)
        mean = self.mean_linear(gru_2_output)
        logvar = self.logvar_linear(gru_2_output)
        return mean, logvar

    def encode_sequence(self, x):
        """Per-position latents for VQ over a latent grid.
        Returns (mean_seq, logvar_seq) each [batch, max_len, latent_dim].
        One latent vector per residue position so the vector quantizer assigns
        a sequence of codes instead of a single code for the whole peptide.
        """
        gru2_f_output, gru2_r_output = self._encode_backbone(x)
        # Align the reverse pass back to forward time order so position t holds
        # both directions' context for residue t.
        gru_2_seq = torch.cat(
            [gru2_f_output, torch.flip(gru2_r_output, (1,))], dim=-1)
        mean_seq = self.mean_linear(gru_2_seq)
        logvar_seq = self.logvar_linear(gru_2_seq)
        return mean_seq, logvar_seq


class HydrAMPDecoder(nn.Module):
    """Autoregressive GRU decoder.

    The latent 'z' and the conditioning vector are concatenated to the input
    at EVERY timestep (and again before the output projection), so the latent
    always receives gradient. This prevents the posterior/latent collapse that
    occurs when a decoder can predict token 't' purely from token 't-1'.

    Training (teacher forcing) and inference (free-running) share the exact
    same loop. Only the choice of "next input token" differs. It eliminating the
    train/inference mismatch that previously caused degenerate generation.
    """

    def __init__(self, latent_dim=48, condition_dim=48, embed_dim=64,
                 gru_dim=128, vocab_size=21, max_len=25, pad_idx=0,
                 # kept for backward-compat with old call sites; ignored
                 lstm_dim=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.gru_dim = gru_dim
        self.condition_dim = condition_dim
        cond_total = latent_dim + condition_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Learned start-of-sequence embedding (BOS) fed at the first step.
        self.start_token = nn.Parameter(torch.zeros(embed_dim))
        self.init_h = nn.Linear(cond_total, gru_dim)
        self.cell = nn.GRUCell(embed_dim + cond_total, gru_dim)
        self.out = nn.Linear(gru_dim + cond_total, vocab_size)

    def forward(self, z, condition=None, temperature=1.0, return_logits=True,
                target_tokens=None, teacher_forcing_ratio=0.5, sample=False):
        """Decode latent + conditioning to sequence logits.
        Args:
            z: [batch, latent_dim] (one code for the whole peptide) or
               [batch, max_len, latent_dim] (a latent grid: one quantized
               code per position). With a grid, position 't' of the decoder
               reads its own latent "z[:, t]", giving the model far more
               reconstruction capacity than a single shared vector.
            condition: [batch, condition_dim] - fused conditioning (or None)
            target_tokens: Optional [batch, max_len] - ground truth for teacher forcing
            teacher_forcing_ratio: prob. of using ground truth as next input
            sample: if True, sample next token from softmax (generation); else argmax
            return_logits: return logits if True, else sampled token ids
        """
        batch = z.shape[0]
        seq_latent = (z.dim() == 3)

        if condition is None:
            condition = torch.zeros(batch, self.condition_dim, device=z.device)
 
        z_pool = z.mean(dim=1) if seq_latent else z
        h = torch.tanh(self.init_h(torch.cat([z_pool, condition], dim=-1)))
        prev_emb = self.start_token.unsqueeze(0).expand(batch, -1)
        logits_list, tokens_list = [], []

        for t in range(self.max_len):
            z_t = z[:, t] if seq_latent else z
            zc = torch.cat([z_t, condition], dim=-1)
            inp = torch.cat([prev_emb, zc], dim=-1)
            h = self.cell(inp, h)
            step_logit = self.out(torch.cat([h, zc], dim=-1))
            logits_list.append(step_logit)

            use_tf = (target_tokens is not None and self.training
                      and torch.rand(1).item() < teacher_forcing_ratio)
            if use_tf:
                next_tok = target_tokens[:, t]
            elif sample:
                probs = F.softmax(step_logit / max(temperature, 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_tok = step_logit.argmax(dim=-1)
            tokens_list.append(next_tok)
            prev_emb = self.token_embed(next_tok)

        logits = torch.stack(logits_list, dim=1)  # [batch, max_len, vocab]

        if return_logits:
            return logits

        return torch.stack(tokens_list, dim=1)
