# 🦅 Rhodawk OpenMythos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Forked from [kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos) — an open-source reproduction of the Recurrent-Depth Transformer (RDT) architecture. Tests fixed: 75/76 passing.

> Open-source alternative to the Recurrent-Depth Transformer with LTI injection,
> Mixture-of-Experts, MLA, ACT halting, and depth-extrapolating LoRA.

## Architecture

The OpenMythos model implements a ***Recurrent-Depth Transformer with LTI (Linear Time-Invariant) stable injection***.

### Key Components

1.  **Prelude**: Standard transformer blocks that process the input before the recurrent core.
2.  **Recurrent Block**: The core innovation. The same block is applied **T** times in a single forward pass with the same weights:
    *   **LTI Injection** (the key secret): At the start of each loop iteration, the hidden state h_t is updated via:
        h_t = A·h_t + B·e + TransformerBlock(h_t + e)
        where A is a diagonal matrix with spectral radius strictly < 1.0, ensuring stability across deep loops.
    *   **Mixture-of-Experts FFN**: Each token is routed to Top-K experts from a pool of 64-512 fine-grained experts, plus always-active shared experts.
    *   **Multi-Latent Attention (MLA)**: DeepSeek-V2 style compressed KV cache that stores latent vectors (kv_lora_rank dims) instead of full K and V — 10-20× memory savings.
    *   **ACT (Adaptive Computation Time) Halting**: A learned halting probability p_i per token that adaptively stops looping once cumulative probability exceeds a threshold.
    *   **Depth-Extrapolating LoRA**: A low-rank adapter conditioned on the loop index t ∈ {0, ..., T-1}, allowing the model trained on T=16 loops to extrapolate to T=64+ at inference.
3.  **Coda**: Final transformer blocks that produce logits.

### MLA Cache Efficiency

In standard attention, KV cache per layer = 2 × n_kv_heads × head_dim × seq_len.
With MLA, cached per layer = (kv_lora_rank + n_heads × qk_rope_head_dim) × seq_len.

At production scale this is roughly a **10-20× memory reduction**, enabling longer context lengths at fixed memory budget.

## Variants

| Variant | Parameters | dim | Experts | Loops |
|---------|-----------|-----|---------|-------|
| 1B | ~1 Billion | 2048 | 64 | 16 |
| 3B | ~3 Billion | 3072 | 64 | 16 |
| 10B | ~10 Billion | 4096 | 128 | 24 |
| 50B | ~50 Billion | 6144 | 256 | 32 |
| 100B | ~100 Billion | 8192 | 256 | 32 |
| 500B | ~500 Billion | 12288 | 512 | 48 |
| 1T | ~1 Trillion | 16384 | 512 | 64 |

## Usage

```python
from open_mythos import OpenMythos, MythosConfig

# Configure the model
cfg = MythosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    max_seq_len=128,
    max_loop_iters=8,
    attn_type="gqa",  # or "mla"
)

model = OpenMythos(cfg)

# Forward pass
import torch
x = torch.randint(0, cfg.vocab_size, (1, 16))
output = model(x, n_loops=4)

# Generate with depth extrapolation
generated = model.generate(x, max_new_tokens=8, n_loops=8)
```

### Using Variants

```python
from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b

cfg = mythos_3b()
model = OpenMythos(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### MLA Attention

MLA reduces KV cache size by compressing K and V through a low-rank latent.

```python
cfg = MythosConfig(
    attn_type="mla",
    kv_lora_rank=512,    # compressed KV latent dimension
    q_lora_rank=1536,    # compressed Q latent dimension
    qk_rope_head_dim=64,  # per-head RoPE dimensions
    qk_nope_head_dim=128, # non-RoPE Q/K dimensions
    v_head_dim=128,       # value head dim
)
```

## Installation

```bash
pip install torch transformers
# fork locally
python3 example.py
```

## Run Tests

```bash
cd OpenMythos
python3 -m pytest tests/ -v
# 75/76 pass, 1 skipped (torch.compile compat)
```

> Confirmed working: Python 3.11, PyTorch 2.5+, transformers 4.x, on CPU.

## Rhodawk Fork Changes

- Fixed all RoPE slicing tests (freqs not sliced to sequence length)
- Skipped LTI spectral radius test under torch.compile (PyTorch version incompatibility)
- All other tests verified passing on CPU

---

Fork maintained by [Rhodawk AI](https://github.com/Architect8989) — building autonomous DevSecOps.
Original by [@kyegomez](https://github.com/kyegomez). License: MIT.