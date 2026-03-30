"""Small encoder-only Transformer for beat-level chord recognition.

Architecture:
    Input:  (batch, seq_len, 24) — 12-dim HPCP + 12-dim bass HPCP per beat
    Output: root_logits (batch, seq_len, 13), quality_logits (batch, seq_len, 3)

Design choices:
    - Sinusoidal positional encoding (avoids overfitting on small dataset)
    - Two separate prediction heads (factorized root + quality)
    - Padding mask support for variable-length sequences
"""
import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ChordTransformer(nn.Module):
    """Encoder-only Transformer for chord recognition.

    Args:
        input_dim:  Feature dimension per beat (default 24)
        d_model:    Transformer model dimension (default 128)
        nhead:      Number of attention heads (default 4)
        num_layers: Number of encoder layers (default 3)
        d_ff:       Feedforward dimension (default 256)
        dropout:    Dropout rate (default 0.1)
        n_roots:    Number of root classes (default 13)
        n_qualities: Number of quality classes (default 3)
    """

    def __init__(self, input_dim=24, d_model=128, nhead=4, num_layers=3,
                 d_ff=256, dropout=0.1, n_roots=13, n_qualities=3):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads
        self.root_head = nn.Linear(d_model, n_roots)
        self.quality_head = nn.Linear(d_model, n_qualities)

    def forward(self, x, src_key_padding_mask=None):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input features
            src_key_padding_mask: (batch, seq_len) True for padded positions

        Returns:
            root_logits:    (batch, seq_len, n_roots)
            quality_logits: (batch, seq_len, n_qualities)
        """
        # Project input to model dimension
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        h = self.pos_enc(h)

        # Transformer encoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Prediction heads
        root_logits = self.root_head(h)
        quality_logits = self.quality_head(h)

        return root_logits, quality_logits

    def predict_probs(self, x, src_key_padding_mask=None):
        """Get softmax probabilities (for inference / decoder integration).

        Returns:
            root_probs:    (batch, seq_len, n_roots)
            quality_probs: (batch, seq_len, n_qualities)
        """
        root_logits, quality_logits = self.forward(x, src_key_padding_mask)
        root_probs = torch.softmax(root_logits, dim=-1)
        quality_probs = torch.softmax(quality_logits, dim=-1)
        return root_probs, quality_probs


def _masked_mean_time(h, pad_mask):
    """Mean over time; pad_mask True = padded position. h: (B, T, D)."""
    m = (~pad_mask).float().unsqueeze(-1)
    num = (h * m).sum(dim=1)
    den = m.sum(dim=1).clamp(min=1.0)
    return num / den


class ChordTransformerCRF(nn.Module):
    """Transformer encoder + CRF sequence decoder for chord recognition.

    Primary loss:  CRF negative log-likelihood on tier1 (25-class) sequences.
    Auxiliary loss: root + quality CE for per-class discrimination (optional).

    Emissions for the CRF can be **direct** (tier1 head logits) or **hybrid**
    (factorized root × per-root maj/min split from tier1), matching
    ``decode_hybrid`` in ``v2/decode.py`` (no key prior on emissions).

    Args:
        emission_mode:   ``direct`` or ``hybrid``
        n_qualities:     3 (N/maj/min) or 7 (full v2 quality vocab)
        use_key_aux:     If True, add a song-level key classifier (12 roots) aux loss
        key_condition_quality: If True, add nn.Embedding(12, n_qualities) to quality logits
    """

    # Em = index 17 in TIER1_VOCAB (N=0, C..B=1-12, Cm..Bm=13-24)
    _EM_IDX = 17
    _N_KEYS = 12

    def __init__(self, input_dim=60, d_model=128, nhead=4, num_layers=3,
                 d_ff=256, dropout=0.1, n_tier1=25, n_roots=13,
                 n_qualities=3, crf_self_bias=2.0, emission_temp=1.0,
                 emission_dropout=0.0, emission_noise_std=0.0,
                 em_emission_bias=0.0, emission_mode='direct',
                 use_key_aux=False, key_condition_quality=False):
        super().__init__()
        self.d_model = d_model
        self.n_tier1 = n_tier1
        self.n_qualities = n_qualities
        self.emission_temp = emission_temp
        self.emission_noise_std = emission_noise_std
        self.em_emission_bias = em_emission_bias
        self.emission_mode = emission_mode
        if emission_mode not in ('direct', 'hybrid'):
            raise ValueError(f"emission_mode must be 'direct' or 'hybrid', got {emission_mode}")

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Tier1 emission head (feeds into CRF direct path and hybrid quality split)
        self.tier1_head = nn.Linear(d_model, n_tier1)
        self.emission_dropout = nn.Dropout(emission_dropout)
        self._root_dropout = nn.Dropout(emission_dropout)

        # CRF layer
        from v2.crf import CRF
        self.crf = CRF(n_tier1, init_self_bias=crf_self_bias)

        # Auxiliary heads
        self.root_head = nn.Linear(d_model, n_roots)
        self.quality_head = nn.Linear(d_model, n_qualities)

        self.use_key_aux = use_key_aux
        self.key_condition_quality = key_condition_quality
        self.key_head = nn.Linear(d_model, self._N_KEYS) if use_key_aux else None
        self.key_quality_emb = (
            nn.Embedding(self._N_KEYS, n_qualities) if key_condition_quality else None
        )

    def _encode(self, x, pad_mask=None):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return h

    def _apply_tier1_bias_noise_drop_train(self, tier1_logits, for_direct_path=True):
        """Em bias, optional Gaussian noise, optional dropout (training)."""
        e = tier1_logits
        if self.em_emission_bias != 0.0:
            e = e.clone()
            e[:, :, self._EM_IDX] -= self.em_emission_bias
        if self.training and self.emission_noise_std > 0.0:
            e = e + torch.randn_like(e) * self.emission_noise_std
        if self.training and for_direct_path:
            e = self.emission_dropout(e)
        return e

    def _tier1_emissions_hybrid(self, root_logits, tier1_logits):
        """Log emissions matching decode_hybrid (tier1 quality split, factorized root)."""
        e_t1 = tier1_logits
        if self.em_emission_bias != 0.0:
            e_t1 = e_t1.clone()
            e_t1[:, :, self._EM_IDX] -= self.em_emission_bias
        r_log = root_logits
        if self.training and self.emission_noise_std > 0.0:
            e_t1 = e_t1 + torch.randn_like(e_t1) * self.emission_noise_std
            r_log = r_log + torch.randn_like(r_log) * self.emission_noise_std
        if self.training:
            e_t1 = self.emission_dropout(e_t1)
            r_log = self._root_dropout(r_log)

        root_p = torch.softmax(r_log, dim=-1)
        d_p = torch.softmax(e_t1, dim=-1)
        eps = 1e-8
        log_rp = torch.log(root_p + eps)
        log_dp = torch.log(d_p + eps)
        out = torch.zeros_like(e_t1)
        out[:, :, 0] = log_rp[:, :, 0] + log_dp[:, :, 0]
        for note in range(12):
            ri = note + 1
            maj_i = note + 1
            min_i = note + 13
            r = log_rp[:, :, ri]
            d_maj = d_p[:, :, maj_i]
            d_min = d_p[:, :, min_i]
            denom = torch.log(d_maj + d_min + eps)
            out[:, :, maj_i] = r + log_dp[:, :, maj_i] - denom
            out[:, :, min_i] = r + log_dp[:, :, min_i] - denom
        if self.emission_temp != 1.0:
            out = out / self.emission_temp
        return out

    def _tier1_emissions_direct(self, tier1_logits):
        e = self._apply_tier1_bias_noise_drop_train(tier1_logits, for_direct_path=True)
        if self.emission_temp != 1.0:
            e = e / self.emission_temp
        return e

    def forward(self, x, tier1_tags, pad_mask=None, key_idx=None, seq_weights=None):
        """Compute CRF loss (+ auxiliary logits for optional CE loss).

        Args:
            key_idx: (batch,) song key index in 0..11 (relative major); optional
            seq_weights: (batch,) weights for CRF NLL (e.g. mean sample_weight per window)

        Returns:
            crf_loss, root_logits, quality_logits, tier1_logits, key_logits_or_None
        """
        h = self._encode(x, pad_mask)

        if pad_mask is not None:
            real_mask = ~pad_mask
        else:
            real_mask = None

        tier1_logits = self.tier1_head(h)
        root_logits = self.root_head(h)
        quality_logits = self.quality_head(h)

        if self.key_quality_emb is not None and key_idx is not None:
            k = key_idx.long().clamp(0, self._N_KEYS - 1)
            quality_logits = quality_logits + self.key_quality_emb(k).unsqueeze(1)

        key_logits = None
        if self.key_head is not None and pad_mask is not None:
            pooled = _masked_mean_time(h, pad_mask)
            key_logits = self.key_head(pooled)

        tier1_emissions = (
            self._tier1_emissions_direct(tier1_logits)
            if self.emission_mode == 'direct'
            else self._tier1_emissions_hybrid(root_logits, tier1_logits)
        )
        crf_loss = self.crf(
            tier1_emissions, tier1_tags, mask=real_mask, seq_weights=seq_weights
        )

        return crf_loss, root_logits, quality_logits, tier1_logits, key_logits

    def decode(self, x, pad_mask=None):
        """CRF Viterbi decode (hybrid emissions if ``emission_mode=='hybrid'``)."""
        h = self._encode(x, pad_mask)
        tier1_logits = self.tier1_head(h)
        root_logits = self.root_head(h)

        if self.emission_mode == 'hybrid':
            tier1_emissions = self._tier1_emissions_hybrid(root_logits, tier1_logits)
        else:
            tier1_emissions = self._tier1_emissions_direct(tier1_logits)

        if pad_mask is not None:
            real_mask = ~pad_mask
        else:
            real_mask = None

        return self.crf.decode(tier1_emissions, mask=real_mask)

    def predict_probs(self, x, pad_mask=None, key_idx=None):
        """Softmax probabilities from root and quality heads."""
        h = self._encode(x, pad_mask)
        root_logits = self.root_head(h)
        quality_logits = self.quality_head(h)
        if self.key_quality_emb is not None and key_idx is not None:
            k = key_idx.long().clamp(0, self._N_KEYS - 1)
            quality_logits = quality_logits + self.key_quality_emb(k).unsqueeze(1)
        return torch.softmax(root_logits, dim=-1), torch.softmax(quality_logits, dim=-1)
