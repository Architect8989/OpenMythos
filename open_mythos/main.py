from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Args: dim -- feature dimension to normalize over, eps -- small constant for numerical stability."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RMS-normalized tensor of the same shape, rescaled by self.weight."""
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Args:
        dim: head dimension (must be even); frequencies computed for dim//2 pairs
        max_len: maximum sequence length to precompute
        theta: RoPE base (higher = slower frequency decay; 500k is LLaMA-3 default)
    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.
    Args: x of shape (B, T, H, head_dim); freqs_cis of shape (T, head_dim//2) already sliced to positions.
    Returns rotated tensor of same shape and dtype.
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023)."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None, cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}
        if _HAS_FLASH_ATTN:
            orig_dtype = q.dtype
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=(mask is not None))
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """Multi-Latent Attention (DeepSeek-V2, 2024). Compresses KV through low-rank latent c_kv for 10-20x cache reduction."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        B, T, _ = x.shape
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]
        k_rope = kv_raw[..., self.kv_lora_rank:]
        k_rope = k_rope.unsqueeze(2).expand(B, T, self.n_heads, self.qk_rope_dim).contiguous()
        k_rope = apply_rope(k_rope, freqs_cis)
        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}
        S = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv)).view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope, v = kv[..., : self.qk_nope_dim], kv[..., self.qk_nope_dim:]
        k = torch.cat([k_nope, k_rope], dim=-1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = self.q_head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """Single SwiGLU feed-forward expert: output = down(silu(gate(x)) * up(x))."""
    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024)."""
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        self.routed_experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)])
        self.shared_experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok) for _ in range(self.n_shared)])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)
        # Shared experts
        shared_out = sum(e(flat) for e in self.shared_experts) if self.shared_experts else torch.zeros_like(flat)
        # Routed experts
        logits = self.router(flat)
        scores = logits.softmax(dim=-1)
        scores_biased = scores + self.router_bias
        topk_weights, topk_idx = scores_biased.topk(self.topk, dim=-1)
        weights = scores.gather(1, topk_idx)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        y = torch.zeros_like(flat)
        for i in range(self.n_experts):
            mask = (topk_idx == i).any(dim=-1)
            if mask.any():
                tok_idx = mask.nonzero(as_tuple=True)[0]
                expert_out = self.routed_experts[i](flat[tok_idx])
                expert_weights = weights[tok_idx][(topk_idx[tok_idx] == i)].reshape(-1, 1)
                y[tok_idx] += expert_out * expert_weights
        return (y + shared_out).view(B, T, D)


# ---------------------------------------------------------------------------
# ACT Halting
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting module."""
    def __init__(self, dim: int, threshold: float = 0.99):
        super().__init__()
        self.threshold = threshold
        self.halt_proj = nn.Linear(dim, 1)

    def forward(self, h):
        p = self.halt_proj(h.detach()).sigmoid()
        return p


# ---------------------------------------------------------------------------
# LoRA Depth Adapter
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """Depth-conditioned LoRA adapter for extrapolating loop depth at inference."""
    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.lora_A = nn.Parameter(torch.empty(max_loops, dim, rank))
        self.lora_B = nn.Parameter(torch.empty(max_loops, rank, dim))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x, loop_idx):
        loop_idx = min(loop_idx, self.lora_A.size(0) - 1)
        return x + (x @ self.lora_A[loop_idx]) @ self.lora_B[loop_idx]


# ---------------------------------------------------------------------------
# LTI Stable Injection
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """Linear Time-Invariant stable injection: h_next = A@h + B@e where A is diagonal with spectral radius < 1."""
    def __init__(self, dim: int, spectral_radius: float = 0.99):
        super().__init__()
        self.dim = dim
        a_diag = torch.rand(dim) * spectral_radius * 0.5
        self.register_buffer("a_diag", a_diag)
        self.B = nn.Linear(dim, dim, bias=False)

    def forward(self, h, e):
        return h * self.a_diag + self.B(e)


# ---------------------------------------------------------------------------
# Loop Index Embedding
# ---------------------------------------------------------------------------


def loop_index_embedding(loop_idx: int, dim: int, max_loops: int = 128) -> torch.Tensor:
    """Return sinusoidal embedding for the loop index, shape (dim,)."""
    idx_t = torch.tensor(loop_idx, dtype=torch.float32)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    emb = torch.zeros(dim)
    emb[0::2] = torch.sin(idx_t * div)
    emb[1::2] = torch.cos(idx_t * div)
    return emb

import math


# ---------------------------------------------------------------------------
# Transformer Block (Prelude/Coda)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm, MoE FFN, and attention."""
    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        if cfg.attn_type == "gqa":
            self.attn = GQAttention(cfg)
        else:
            self.attn = MLAttention(cfg)
        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = Expert(cfg.dim, cfg.dim * 4 // 3)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key="default"):
        x = x + self.attn(self.norm1(x), freqs_cis, mask, kv_cache, cache_key)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Recurrent Block (the core innovation)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """The core recurrent block with LTI injection, TransformerBlock, ACT halting, and LoRA depth adaptation."""
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.lti = LTIInjection(cfg.dim)
        self.transformer = TransformerBlock(cfg, use_moe=True)
        self.act = ACTHalting(cfg.dim, cfg.act_threshold)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

    def forward(self, h, e, freqs_cis, n_loops, mask=None, kv_cache=None, act_enabled=True):
        h = self.lti(h, e)
        total_p = torch.zeros(h.shape[0], h.shape[1], 1, device=h.device)
        active = torch.ones(h.shape[0], h.shape[1], 1, device=h.device, dtype=torch.bool)
        output = torch.zeros_like(h)
        for t in range(n_loops):
            h_in = self.lora(h, t) + loop_index_embedding(t, h.shape[-1], n_loops).to(h.device).unsqueeze(0).unsqueeze(0)
            h = self.transformer(h_in, freqs_cis, mask, kv_cache, f"loop_{t}")
            if act_enabled:
                p = self.act(h)
                total_p = total_p + active.float() * p
                remainder = 1.0 - total_p
                output = output + active.float() * remainder.clamp(min=0.0) * h
                active = active & (total_p < self.act.threshold)
                if not active.any():
                    break
            else:
                output = h
        return output if not act_enabled else output


# ---------------------------------------------------------------------------
# OpenMythos Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """Recurrent-Depth Transformer with LTI stable injection, MoE, MLA/GQA, ACT halting, depth-extrapolating LoRA."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.prelude = nn.ModuleList([TransformerBlock(cfg, use_moe=(i == 0)) for i in range(cfg.prelude_layers)])
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList([TransformerBlock(cfg, use_moe=(i == 0)) for i in range(cfg.coda_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # RoPE frequencies
        if cfg.attn_type == "gqa":
            head_dim = cfg.dim // cfg.n_heads
        else:
            head_dim = cfg.qk_rope_head_dim
        self.register_buffer("freqs_cis", precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta), persistent=False)

    def forward(self, input_ids, n_loops=None, kv_cache=None, act_enabled=True):
        if n_loops is None:
            n_loops = self.cfg.max_loop_iters
        B, T = input_ids.shape
        h = self.tok_embed(input_ids)
        e = h  # external input = embedding for first LTI injection
        freqs = self.freqs_cis[:T]
        for block in self.prelude:
            h = block(h, freqs, kv_cache=kv_cache)
        h = self.recurrent(h, e, freqs, n_loops=n_loops, kv_cache=kv_cache, act_enabled=act_enabled)
        for block in self.coda:
            h = block(h, freqs, kv_cache=kv_cache)
        h = self.norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, n_loops=None, temperature=1.0, top_p=1.0):
        if n_loops is None:
            n_loops = self.cfg.max_loop_iters
        self.eval()
        kv_cache = {}
        generated = list(input_ids[0].tolist())
        for _ in range(max_new_tokens):
            if len(generated) > self.cfg.max_seq_len:
                break
            inp = torch.tensor([generated[-min(len(generated), self.cfg.max_seq_len):]], device=input_ids.device)
            logits = self(inp, n_loops=n_loops, kv_cache=kv_cache)[:, -1:, :]
            if temperature > 0:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    cumprobs = sorted_logits.softmax(-1).cumsum(-1)
                    cutoff = (cumprobs > top_p).float().argmax(dim=-1, keepdim=True)
                    logits = logits.scatter(-1, sorted_idx, torch.where(cumprobs <= top_p.unsqueeze(-1), sorted_logits, float("-inf")))
                probs = logits.softmax(-1)
                next_token = torch.multinomial(probs.squeeze(0).squeeze(0), 1).item()
            else:
                next_token = logits.argmax(-1).item()
            generated.append(next_token)
        return torch.tensor([generated], device=input_ids.device)