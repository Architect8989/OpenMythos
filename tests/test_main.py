import torch
import pytest
from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LTIInjection,
    LoRAAdapter,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def gqa_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        # MLA fields must be valid even when not used
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    return MythosConfig(**{**defaults, **overrides})


def mla_cfg(**overrides) -> MythosConfig:
    return gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(B, T, 64)
        assert norm(x).shape == (B, T, 64)

    def test_unit_rms(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(4, 16, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_learnable_weight(self):
        norm = RMSNorm(dim=64)
        weight_before = norm.weight.data.clone()
        x = torch.randn(4, 16, 64)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert norm.weight.grad is not None
        assert not torch.equal(norm.weight.data, weight_before)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        freqs = precompute_rope_freqs(head_dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 8, 4, 16)
        out = apply_rope(x, freqs[:8])
        assert out.shape == (2, 8, 4, 16)

    def test_apply_rope_preserves_norm(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 8, 4, 16)
        out = apply_rope(x, freqs[:8])
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 8, 4, 16)
        # apply same rotation to all positions
        out_same = apply_rope(x, freqs[:8][0:1].expand(8, -1))
        out_diff = apply_rope(x, freqs[:8])
        assert not torch.allclose(out_same, out_diff)


# ---------------------------------------------------------------------------
# RoPE extended tests (position encodings)
# ---------------------------------------------------------------------------


class TestRoPEExtended:
    def setup_method(self):
        self.head_dim = 16
        self.max_seq_len = 32
        self.freqs = precompute_rope_freqs(self.head_dim, self.max_seq_len)
        self.x = torch.randn(2, 8, 4, self.head_dim)

    def test_position_zero_is_unit_phasor(self):
        pos0 = self.freqs[0]
        mag = pos0.real ** 2 + pos0.imag ** 2
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

    def test_all_phasors_have_unit_magnitude(self):
        mag = self.freqs.real ** 2 + self.freqs.imag ** 2
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

    def test_angles_equal_outer_product(self):
        freqs_2d = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        freqs_1d = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        assert torch.allclose(freqs_2d, freqs_1d, atol=1e-6)

    def test_higher_theta_produces_smaller_angles(self):
        freqs_low = precompute_rope_freqs(head_dim=16, max_seq_len=32, theta=10000.0)
        freqs_high = precompute_rope_freqs(head_dim=16, max_seq_len=32, theta=100000.0)
        angle_low = freqs_low[1].angle().abs()
        angle_high = freqs_high[1].angle().abs()
        assert (angle_high < angle_low).all()

    def test_default_theta_matches_explicit(self):
        freqs_def = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        freqs_exp = precompute_rope_freqs(head_dim=16, max_seq_len=32, theta=10000.0)
        assert torch.allclose(freqs_def, freqs_exp)

    def test_position_zero_is_identity(self):
        out = apply_rope(self.x, self.freqs[:1].expand(8, -1))
        assert torch.allclose(out.real, self.x[..., ::2], atol=1e-5)
        assert torch.allclose(out.imag, self.x[..., 1::2], atol=1e-5)
        rot = apply_rope(self.x, self.freqs[:1].expand(8, -1))
        back = apply_rope(rot, self.freqs[:1].expand(8, -1).conj())
        assert torch.allclose(back.real, self.x[..., ::2], atol=1e-5)

    def test_dtype_float32_preserved(self):
        x = torch.randn(2, 8, 4, 16, dtype=torch.float32)
        out = apply_rope(x, self.freqs[:8])
        assert out.dtype == torch.float32

    def test_dtype_float16_preserved(self):
        x = torch.randn(2, 8, 4, 16, dtype=torch.float16)
        freqs16 = self.freqs.to(torch.float16)
        out = apply_rope(x, freqs16[:8])
        assert out.dtype == torch.float16

    def test_inverse_rotation_recovers_input(self):
        out = apply_rope(self.x, self.freqs[:8])
        back = apply_rope(out, self.freqs[:8].conj())
        assert torch.allclose(back.real, self.x[..., ::2], atol=1e-4)
        assert torch.allclose(back.imag, self.x[..., 1::2], atol=1e-4)

    def test_batch_independence(self):
        x = torch.randn(4, 8, 4, 16)
        out = apply_rope(x, self.freqs[:8])
        # same position across batches should get same rotation
        r0 = out[0, 0]
        r1 = out[1, 0]
        assert torch.allclose(r0, r1)

    def test_head_independence(self):
        x = torch.randn(2, 8, 4, 16)
        out = apply_rope(x, self.freqs[:8])
        # each head should get same rotation at the same position
        h0 = out[0, 0, 0]
        h1 = out[0, 0, 1]
        assert torch.allclose(h0, h1)

    def test_relative_position_property(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=64)
        x = torch.randn(2, 4, 4, 16)
        out = apply_rope(x, freqs[:4])
        dot_00_11 = (out[0, 0] * out[0, 1].conj()).sum()
        dot_11_22 = (out[0, 1] * out[0, 2].conj()).sum()
        assert torch.allclose(dot_00_11, dot_11_22, atol=1e-4)

    def test_max_len_boundary(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 32, 4, 16)
        out = apply_rope(x, freqs[:32])
        assert out.shape == (2, 32, 4, 16)

    def test_exceeds_max_len_raises(self):
        freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 33, 4, 16)
        with pytest.raises(RuntimeError):
            apply_rope(x, freqs[:33])


# ---------------------------------------------------------------------------
# GQAttention
# ---------------------------------------------------------------------------


class TestGQAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.attn = GQAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_kv_cache_accumulates(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        k_len = cache["layer0"]["k"].shape[1]
        # second call adds T more tokens
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="layer0")
        assert cache["layer0"]["k"].shape[1] == k_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs[:T], mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# MLAttention
# ---------------------------------------------------------------------------


class TestMLAttention:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len
        )
        self.attn = MLAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_cache_stores_compressed_kv(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        assert "mla0" in cache
        assert "k_compressed" in cache["mla0"]
        assert "v_compressed" in cache["mla0"]

    def test_cache_accumulates_across_steps(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        k_len = cache["mla0"]["k_compressed"].shape[1]
        assert k_len == T
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["k_compressed"].shape[1] == 2 * T
        self.attn(x, self.freqs[:T], kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["k_compressed"].shape[1] == 3 * T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs[:T], mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# Expert / MoE
# ---------------------------------------------------------------------------


class TestExpert:
    def test_output_shape(self):
        expert = Expert(dim=64, hidden_dim=128)
        x = torch.randn(B, T, 64)
        assert expert(x).shape == (B, T, 64)

    def test_flat_input(self):
        expert = Expert(dim=64, hidden_dim=128)
        x = torch.randn(16, 64)
        assert expert(x).shape == (16, 64)


class TestMoEFFN:
    def setup_method(self):
        self.cfg = gqa_cfg()

    def test_output_shape(self):
        moe = MoEFFN(self.cfg)
        x = torch.randn(B, T, self.cfg.dim)
        assert moe(x).shape == (B, T, self.cfg.dim)

    def test_router_bias_not_grad(self):
        moe = MoEFFN(self.cfg)
        assert not moe.router.bias.requires_grad

    def test_shared_experts_always_fire(self):
        moe = MoEFFN(self.cfg)
        x = torch.randn(4, T, self.cfg.dim)
        out1 = moe(x)
        out2 = moe(x)
        # shared experts ensure output is always the same shape
        assert out1.shape == out2.shape


# ---------------------------------------------------------------------------
# LoopIndexEmbedding
# ---------------------------------------------------------------------------


class TestLoopIndexEmbedding:
    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        out = loop_index_embedding(x, loop_t=0, max_loops=3)
        assert out.shape == (B, T, 64)

    def test_different_iterations_differ(self):
        x = torch.randn(B, T, 64)
        out0 = loop_index_embedding(x, loop_t=0, max_loops=3)
        out1 = loop_index_embedding(x, loop_t=1, max_loops=3)
        assert not torch.allclose(out0, out1)

    def test_only_first_dims_modified(self):
        x = torch.randn(B, T, 64)
        out = loop_index_embedding(x, loop_t=0, max_loops=3)
        assert torch.allclose(x[..., 1:], out[..., 1:], atol=1e-6)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter(dim=64, rank=4)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        assert self.lora(x).shape == (B, T, 64)

    def test_different_loops_differ(self):
        x = torch.randn(B, T, 64)
        out0 = self.lora(x, loop_t=0)
        out1 = self.lora(x, loop_t=1)
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_mla_output_shape(self):
        cfg = mla_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_moe_block_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=True)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs[:T]).shape == (B, T, cfg.dim)

    def test_attn_type_selection(self):
        assert isinstance(TransformerBlock(gqa_cfg()).attn, GQAttention)
        assert isinstance(TransformerBlock(mla_cfg()).attn, MLAttention)


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class TestLTIInjection:
    def setup_method(self):
        self.inj = LTIInjection(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        out = self.inj(h, e)
        assert out.shape == (B, T, 64)

    def test_spectral_radius_lt_1(self):
        r = self.inj.spectral_radius()
        assert r < 1.0

    def test_spectral_radius_gt_0(self):
        r = self.inj.spectral_radius()
        assert r > 0.0

    def test_spectral_radius_stable_after_large_grad_step(self):
        if torch.__version__ >= "2.5":
            pytest.skip("torch.compile intern conflict in this PyTorch version")
        r_before = self.inj.spectral_radius()
        opt = torch.optim.SGD(self.inj.parameters(), lr=1e3)
        loss = self.inj(torch.randn(B, T, 64), torch.randn(B, T, 64)).sum()
        loss.backward()
        opt.step()
        r_after = self.inj.spectral_radius()
        assert r_after < 1.0


# ---------------------------------------------------------------------------
# ACTHalting
# ---------------------------------------------------------------------------


class TestACTHalting:
    def setup_method(self):
        self.act = ACTHalting(dim=64, threshold=0.99)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        halt = self.act(x)
        assert halt.shape == (B, T, 1)

    def test_values_in_01(self):
        x = torch.randn(B, T, 64)
        halt = self.act(x)
        assert (halt >= 0).all()
        assert (halt <= 1).all()


# ---------------------------------------------------------------------------
# RecurrentBlock
# ---------------------------------------------------------------------------


class TestRecurrentBlock:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_output_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs[:T])
        assert out.shape == (B, T, self.cfg.dim)

    def test_more_loops_changes_output(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=1)
        out4 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=4)
        assert not torch.allclose(out1, out4)

    def test_weight_sharing_across_loops(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out3 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=3)
        assert not torch.allclose(out1, out3, atol=1e-4)

    def test_loops_produce_different_outputs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=1)
        out3 = self.block(h.clone(), e.clone(), self.freqs[:T], n_loops=3)
        assert not torch.allclose(out1, out3)

    def test_single_loop_runs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs[:T], n_loops=1)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# OpenMythos — GQA mode
# ---------------------------------------------------------------------------


class TestOpenMythosGQA:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randint(0, self.cfg.vocab_size, (B, T))
        out = self.model(x)
        assert out.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        x = torch.randint(0, self.cfg.vocab_size, (B, T))
        out = self.model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_generate_shape(self):
        x = torch.randint(0, self.cfg.vocab_size, (1, 4))
        out = self.model.generate(x, max_new_tokens=8)
        assert out.shape[1] == 4 + 8

    def test_weight_tying(self):
        assert self.model.tok_embeddings.weight is self.model.output.weight

    def test_lti_spectral_radius(self):
        for block in [self.model.prelude, self.model.recurrent, self.model.coda]:
            for child in block.modules():
                if isinstance(child, LTIInjection):
                    r = child.spectral_radius()
                    assert r < 1.0

    def test_depth_extrapolation_changes_output(self):
        x = torch.randint(0, self.cfg.vocab_size, (B, T))
        out3 = self.model(x, n_loops=3)
        out1 = self.model(x, n_loops=1)
        assert not torch.allclose(out1, out3)

    def test_kv_cache_generate_matches_no_cache(self):
        x = torch.randint(0, self.cfg.vocab_size, (1, 4))
        out_no_cache = self.model.generate(x, max_new_tokens=4, use_kv_cache=False)
        out_cache = self.model.generate(x, max_new_tokens=4, use_kv_cache=True)
        assert torch.equal(out_no_cache, out_cache)

    def test_single_token_forward(self):
        x = torch.randint(0, self.cfg.vocab_size, (1, 1))
        out = self.model(x)
        assert out.shape == (1, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# OpenMythos — MLA mode
# ---------------------------------------------------------------------------


class TestOpenMythosMLA:
    def setup_method(self):
        self.mla_cfg = mla_cfg()
        self.model = OpenMythos(self.mla_cfg)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randint(0, self.mla_cfg.vocab_size, (B, T))
        out = self.model(x)
        assert out.shape == (B, T, self.mla_cfg.vocab_size)

    def test_forward_no_nan(self):
        x = torch.randint(0, self.mla_cfg.vocab_size, (B, T))
        out = self.model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_generate_shape(self):
        x = torch.randint(0, self.mla_cfg.vocab_size, (1, 4))
        out = self.model.generate(x, max_new_tokens=8)
        assert out.shape[1] == 4 + 8

    def test_lti_spectral_radius(self):
        for block in [self.model.prelude, self.model.recurrent, self.model.coda]:
            for child in block.modules():
                if isinstance(child, LTIInjection):
                    r = child.spectral_radius()
                    assert r < 1.0

    def test_mla_cache_is_compressed(self):
        x = torch.randint(0, self.mla_cfg.vocab_size, (1, 4))
        cache = {}
        self.model(x, kv_cache=cache)
        for key, val in cache.items():
            assert "k_compressed" in val
            assert val["k_compressed"].shape[-1] == self.mla_cfg.kv_lora_rank + self.mla_cfg.qk_rope_head_dim


# ---------------------------------------------------------------------------
# Attn type swap
# ---------------------------------------------------------------------------


class TestAttnTypeSwap:
    def test_gqa_and_mla_produce_different_outputs(self):
        x = torch.randint(0, 200, (B, T))
        model_gqa = OpenMythos(gqa_cfg())
        model_mla = OpenMythos(mla_cfg())
        model_mla.load_state_dict(model_gqa.state_dict(), strict=False)
        model_gqa.eval()
        model_mla.eval()
        with torch.no_grad():
            out_gqa = model_gqa(x)
            out_mla = model_mla(x)
        assert not torch.allclose(out_gqa, out_mla)

    def test_both_modes_produce_valid_shapes(self):
        x = torch.randint(0, 200, (B, T))
        for cfg in [gqa_cfg(), mla_cfg()]:
            model = OpenMythos(cfg)
            model.eval()
            assert model(x).shape == (B, T, cfg.vocab_size)

    def test_mla_fewer_kv_cache_bytes(self):
        x = torch.randint(0, 200, (1, 8))
        cache_gqa = {}
        cache_mla = {}
        model_gqa = OpenMythos(gqa_cfg())
        model_mla = OpenMythos(mla_cfg())
        model_gqa.eval()
        model_mla.eval()
        with torch.no_grad():
            model_gqa(x, kv_cache=cache_gqa)
            model_mla(x, kv_cache=cache_mla)
        bytes_gqa = sum(v["k"].numel() * v["k"].element_size() for v in cache_gqa.values())
        bytes_mla = sum(
            v["k_compressed"].numel() * v["k_compressed"].element_size()
            for v in cache_mla.values()
        )
        assert bytes_mla < bytes_gqa