#!/usr/bin/env python3
"""Quick sanity check for Transformer+CRF variants."""
import torch
import sys
sys.path.insert(0, '.')

from v2.transformer_model import ChordTransformerCRF

# Direct emissions (legacy-style)
model = ChordTransformerCRF(
    input_dim=60,
    d_model=128,
    nhead=4,
    num_layers=3,
    d_ff=256,
    dropout=0.1,
    crf_self_bias=2.0,
    emission_temp=5.0,
    emission_dropout=0.1,
    emission_noise_std=0.05,
    emission_mode='direct',
)

print("Model (direct emissions):")
X = torch.randn(2, 128, 60)
tags = torch.randint(0, 25, (2, 128))
model.train()
crf_loss, root_logits, quality_logits, tier1_logits, key_logits = model(
    X, tags, pad_mask=None)
assert key_logits is None
print(f"  CRF loss={crf_loss.item():.4f} root{root_logits.shape} qual{quality_logits.shape}")

model.eval()
with torch.no_grad():
    paths = model.decode(X)
print(f"  decode len={len(paths[0])}")

# Hybrid + key conditioning + key aux
model2 = ChordTransformerCRF(
    input_dim=48,
    n_qualities=7,
    emission_mode='hybrid',
    use_key_aux=True,
    key_condition_quality=True,
)
pad = torch.zeros(2, 128, dtype=torch.bool)
pad[:, 100:] = True
key_idx = torch.tensor([3, 5])
model2.train()
out = model2(X[:, :, :48], tags, pad_mask=pad, key_idx=key_idx)
crf_loss, _, _, _, kl = out
assert kl is not None and kl.shape == (2, 12)
print(f"\nModel (hybrid, q7, key aux+cond): CRF={crf_loss.item():.4f} key_logits{kl.shape}")
model2.eval()
with torch.no_grad():
    p2 = model2.decode(X[:, :, :48], pad_mask=pad)
print(f"  decode len={len(p2[0])}")

print("\n✅ All checks passed!")
