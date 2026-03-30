"""Linear-chain CRF layer for sequence labelling.

Implements:
    - Forward algorithm (log-partition function)
    - Gold-path scoring
    - Viterbi decoding
    - Negative log-likelihood loss

Operates on (batch, seq_len, n_tags) emission scores.
Mask convention: True = real token, False = padding (opposite of Transformer's
src_key_padding_mask where True = padding).
"""
import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain Conditional Random Field.

    Args:
        num_tags: Number of tag classes (e.g. 25 for tier1 chords).
        init_self_bias: If > 0, initialise self-transition scores higher
                        to encourage staying in the same state.
    """

    def __init__(self, num_tags, init_self_bias=2.0):
        super().__init__()
        self.num_tags = num_tags

        # (from_tag, to_tag) transition scores
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

        self._init_params(init_self_bi

    def _init_params(self, self_bias):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        if self_bias > 0:
            # Encourage self-transitions to reduce flipping
            with torch.no_grad():
                self.transitions.fill_diagonal_(self_bias)

    # ------------------------------------------------------------------
    # Loss: negative log-likelihood
    # ------------------------------------------------------------------
    def forward(self, emissions, tags, mask=None, seq_weights=None):
        """Compute average negative log-likelihood over the batch.

        Args:
            emissions: (batch, seq_len, num_tags) — unnormalised scores
            tags:      (batch, seq_len) — gold tag indices
            mask:      (batch, seq_len) bool — True for real tokens
            seq_weights: optional (batch,) non-negative weights for each sequence

        Returns:
            Scalar loss (mean over batch, or weighted mean if seq_weights given).
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)

        numerator = self._score_sentence(emissions, tags, mask)   # (batch,)
        denominator = self._forward_algorithm(emissions, mask)     # (batch,)
        nll = denominator - numerator                              # (batch,)
        if seq_weights is not None:
            w = seq_weights.float()
            return (nll * w).sum() / w.sum().clamp(min=1e-8)
        return nll.mean()

    # ------------------------------------------------------------------
    # Forward algorithm: log Z (partition function)
    # ------------------------------------------------------------------
    def _forward_algorithm(self, emissions, mask):
        """Log-space forward algorithm.

        Returns: (batch,) log-partition values.
        """
        batch, seq_len, n_tags = emissions.shape

        # alpha_0 = start_transitions + emissions[:, 0]
        alpha = self.start_transitions + emissions[:, 0]  # (batch, n_tags)

        for t in range(1, seq_len):
            # alpha:       (batch, n_tags)      — previous scores
            # emissions_t: (batch, n_tags)      — current emission
            # transitions: (n_tags, n_tags)     — transitions[i, j] = score(i -> j)
            emit_t = emissions[:, t]                            # (batch, n_tags)
            # (batch, n_tags, 1) + (n_tags, n_tags) + (batch, 1, n_tags)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t.unsqueeze(1)
            # logsumexp over previous tag dimension
            next_alpha = torch.logsumexp(scores, dim=1)         # (batch, n_tags)

            # Where mask is False (padding), keep old alpha
            m = mask[:, t].unsqueeze(1)                         # (batch, 1)
            alpha = torch.where(m, next_alpha, alpha)

        # Add end transitions
        alpha = alpha + self.end_transitions                    # (batch, n_tags)
        return torch.logsumexp(alpha, dim=1)                    # (batch,)

    # ------------------------------------------------------------------
    # Score of a specific tag sequence
    # ------------------------------------------------------------------
    def _score_sentence(self, emissions, tags, mask):
        """Score of the gold tag sequence (vectorized — no Python loop).

        Returns: (batch,) scores.
        """
        batch, seq_len, _ = emissions.shape

        # Emission scores for gold tags — single gather
        emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # (batch, seq_len)
        emit_scores = (emit_scores * mask.float()).sum(dim=1)            # (batch,)

        # Transition scores — vectorized gather over all consecutive pairs
        from_tags = tags[:, :-1]                                         # (batch, seq_len-1)
        to_tags = tags[:, 1:]                                            # (batch, seq_len-1)
        trans_scores = self.transitions[from_tags, to_tags]              # (batch, seq_len-1)
        trans_mask = mask[:, 1:].float()                                 # (batch, seq_len-1)
        trans_total = (trans_scores * trans_mask).sum(dim=1)             # (batch,)

        # Start transition
        start_scores = self.start_transitions[tags[:, 0]]               # (batch,)

        # End transition: last real token in each sequence
        lengths = mask.long().sum(dim=1)                                 # (batch,)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        end_scores = self.end_transitions[last_tags]                     # (batch,)

        return emit_scores + start_scores + trans_total + end_scores

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------
    def decode(self, emissions, mask=None):
        """Viterbi decoding.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask:      (batch, seq_len) bool — True for real tokens

        Returns:
            best_tags: list of lists, each inner list is the decoded tag sequence
                       (length = number of real tokens in that example)
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)

        batch, seq_len, n_tags = emissions.shape

        # Viterbi forward
        viterbi = self.start_transitions + emissions[:, 0]     # (batch, n_tags)
        backpointers = []

        for t in range(1, seq_len):
            emit_t = emissions[:, t]
            # (batch, n_tags, 1) + (n_tags, n_tags) → (batch, n_tags, n_tags)
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags_t = scores.max(dim=1)       # (batch, n_tags)
            next_viterbi = best_scores + emit_t

            m = mask[:, t].unsqueeze(1)
            viterbi = torch.where(m, next_viterbi, viterbi)
            backpointers.append(best_tags_t)

        # Add end transitions
        viterbi = viterbi + self.end_transitions

        # Backtrack
        results = []
        best_last = viterbi.argmax(dim=1)                      # (batch,)
        lengths = mask.long().sum(dim=1)

        for b in range(batch):
            L = lengths[b].item()
            tags = [0] * L
            tags[L - 1] = best_last[b].item()
            for t in range(L - 2, -1, -1):
                tags[t] = backpointers[t][b, tags[t + 1]].item()
            results.append(tags)

        return results
