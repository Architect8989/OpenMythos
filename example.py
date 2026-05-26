from open_mythos.main import OpenMythos, MythosConfig

# Tiny model for quick smoke testing
cfg = MythosConfig(
    vocab_size=1000,
    dim=128,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=64,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=4,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    act_threshold=0.99,
    lora_rank=4,
    kv_lora_rank=8,
    q_lora_rank=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    v_head_dim=8,
)

model = OpenMythos(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate with depth extrapolation (train on 4 loops, infer with 8)
import torch
x = torch.randint(0, cfg.vocab_size, (1, 4))
print(f"Input shape: {x.shape}")

# Forward pass
output = model(x, n_loops=2)
print(f"Output shape: {output.shape}")

# Generate
print("\nGenerating with n_loops=4 (extrapolated from default 2)...")
gen = model.generate(x, max_new_tokens=4, n_loops=4)
print(f"Generated shape: {gen.shape}")