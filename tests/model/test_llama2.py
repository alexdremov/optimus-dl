import torch
import pytest
import torch.nn as nn

from optimus_dl.modules.model.blocks.rope import (
    _reshape_for_broadcast,
    apply_rotary_emb,
)
from optimus_dl.modules.model.llama2 import (
    Llama,
    LlamaBlock,
    LlamaConfig,
    RotarySelfAttention,
    SwiGLUMLP,
    llama_lite,
    precompute_freqs_cis,
)


class TestRoPEFunctions:
    """Tests for Rotary Position Embedding functions"""

    def test_precompute_freqs_cis(self):
        """Test precomputation of complex frequencies for Rotary Position Embeddings."""
        dim = 64
        seq_len = 100

        freqs_cis = precompute_freqs_cis(dim, seq_len)

        # Check shape: (seq_len, dim // 2)
        assert freqs_cis.shape == (seq_len, dim // 2, 2)
        assert freqs_cis.dtype == torch.float32

        # Check that all values have magnitude 1 (unit complex numbers)
        magnitudes = freqs_cis.norm(dim=-1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_precompute_freqs_cis_different_theta(self):
        dim = 32
        seq_len = 50
        theta = 1000.0

        freqs_cis = precompute_freqs_cis(dim, seq_len, theta)
        assert freqs_cis.shape == (seq_len, dim // 2, 2)
        assert freqs_cis.dtype == torch.float32

    def test_reshape_for_broadcast_assertions(self):
        # Test dimension assertion
        freqs_cis = torch.randn(10, 32, dtype=torch.complex64)
        x_1d = torch.randn(32, dtype=torch.complex64)  # Only 1 dimension

        with pytest.raises(AssertionError):
            _reshape_for_broadcast(freqs_cis, x_1d)

        # Test shape mismatch assertion
        x_wrong_shape = torch.randn(
            2, 5, 16, dtype=torch.complex64
        )  # Wrong seq_len and head_dim

        with pytest.raises(AssertionError):
            _reshape_for_broadcast(freqs_cis, x_wrong_shape)

    def test_apply_rotary_emb(self):
        """Test application of rotary embeddings to query and key tensors."""
        batch_size, seq_len, n_head, head_size = 2, 10, 8, 64

        q = torch.randn(batch_size, seq_len, n_head, head_size)
        k = torch.randn(batch_size, seq_len, n_head, head_size)
        freqs_cis = precompute_freqs_cis(head_size, seq_len)

        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        # Check output shapes
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Check data types are preserved
        assert q_rot.dtype == q.dtype
        assert k_rot.dtype == k.dtype

        # Rotary embeddings should change the values
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)

    def test_apply_rotary_emb_even_head_size(self):
        """Test that rotary embeddings work only with even head sizes"""
        batch_size, seq_len, n_head = 2, 10, 8

        # Even head size should work
        head_size = 64
        q = torch.randn(batch_size, seq_len, n_head, head_size)
        k = torch.randn(batch_size, seq_len, n_head, head_size)
        freqs_cis = precompute_freqs_cis(head_size, seq_len)

        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
        assert q_rot.shape == q.shape

    def test_rotary_emb_position_dependence(self):
        """Test that rotary embeddings produce different results for different positions (key property)."""
        batch_size, n_head, head_size = 1, 1, 32

        # Same query at different positions
        q_pos0 = torch.ones(batch_size, 1, n_head, head_size)
        q_pos1 = torch.ones(batch_size, 1, n_head, head_size)

        # Different position embeddings
        freqs_cis_pos0 = precompute_freqs_cis(head_size, 2)[:1]  # Position 0
        freqs_cis_pos1 = precompute_freqs_cis(head_size, 2)[1:2]  # Position 1

        k_dummy = torch.zeros_like(q_pos0)

        q_rot_pos0, _ = apply_rotary_emb(q_pos0, k_dummy, freqs_cis_pos0)
        q_rot_pos1, _ = apply_rotary_emb(q_pos1, k_dummy, freqs_cis_pos1)

        # Different positions should produce different rotations
        assert not torch.allclose(q_rot_pos0, q_rot_pos1, atol=1e-6)


class TestLlamaMLP:
    """Tests for LlamaMLP (SwiGLU implementation)"""

    def test_init(self):
        mlp = SwiGLUMLP(n_embd=768, multiple_of=256)

        # Check that all linear layers exist and have no bias
        assert isinstance(mlp.w1, nn.Linear)
        assert isinstance(mlp.w2, nn.Linear)
        assert isinstance(mlp.c_proj, nn.Linear)

        assert mlp.w1.bias is None
        assert mlp.w2.bias is None
        assert mlp.c_proj.bias is None

    def test_hidden_dimension_calculation(self):
        """Test that hidden dimension is calculated correctly"""
        mlp = SwiGLUMLP(n_embd=768, multiple_of=256)

        # Calculate expected hidden dimension
        expected_hidden_dim = int(2 * 768 * 4 / 3)  # 2048
        expected_hidden_dim = 256 * (
            (expected_hidden_dim + 256 - 1) // 256
        )  # Round up to multiple of 256

        assert mlp.w1.out_features == expected_hidden_dim
        assert mlp.w2.out_features == expected_hidden_dim
        assert mlp.c_proj.in_features == expected_hidden_dim

    def test_forward_swiglu(self):
        """Test SwiGLU activation: SiLU(w1(x)) * w2(x)"""
        mlp = SwiGLUMLP(n_embd=256, multiple_of=64)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)

        output = mlp(x)
        assert output.shape == (batch_size, seq_len, 256)

        # Manually compute SwiGLU to verify
        w1_out = mlp.w1(x)
        w2_out = mlp.w2(x)
        expected_output = mlp.c_proj(torch.nn.functional.silu(w1_out) * w2_out)

        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_different_multiple_of_values(self):
        """Test with different multiple_of values"""
        multiple_of_values = [64, 128, 256, 512]

        for multiple_of in multiple_of_values:
            mlp = SwiGLUMLP(n_embd=512, multiple_of=multiple_of)

            # Hidden dimension should be divisible by multiple_of
            hidden_dim = mlp.w1.out_features
            assert hidden_dim % multiple_of == 0

            # Test forward pass
            x = torch.randn(1, 5, 512)
            output = mlp(x)
            assert output.shape == (1, 5, 512)

    def test_gradient_flow(self):
        mlp = SwiGLUMLP(n_embd=256, multiple_of=64)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert mlp.w1.weight.grad is not None
        assert mlp.w2.weight.grad is not None
        assert mlp.c_proj.weight.grad is not None
        assert x.grad is not None


class TestLlamaAttention:
    """Tests for LlamaAttention with RoPE"""

    def test_init_inheritance(self):
        """Test that LlamaAttention inherits from CausalSelfAttention"""
        attention = RotarySelfAttention(n_embd=768, n_head=12)

        # Should have the new attributes for GQA
        assert hasattr(attention, "wq")
        assert hasattr(attention, "wk")
        assert hasattr(attention, "wv")
        assert hasattr(attention, "wo")
        assert hasattr(attention, "n_head")

    def test_forward_with_rope(self):
        attention = RotarySelfAttention(n_embd=768, n_head=12)
        attention.eval()

        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, 768)

        # Create freqs_cis for RoPE
        head_dim = 768 // 12
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = attention(x, freqs_cis)
        assert output.shape == (batch_size, seq_len, 768)

    def test_different_sequence_lengths(self):
        attention = RotarySelfAttention(n_embd=256, n_head=8)
        attention.eval()

        head_dim = 256 // 8
        sequence_lengths = [1, 5, 20, 100]

        for seq_len in sequence_lengths:
            x = torch.randn(1, seq_len, 256)
            freqs_cis = precompute_freqs_cis(head_dim, seq_len)

            output = attention(x, freqs_cis)
            assert output.shape == (1, seq_len, 256)

    def test_gradient_flow_with_rope(self):
        attention = RotarySelfAttention(n_embd=256, n_head=8)

        batch_size, seq_len = 1, 10
        x = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        freqs_cis = precompute_freqs_cis(256 // 8, seq_len)

        output = attention(x, freqs_cis)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert attention.wq.weight.grad is not None
        assert attention.wk.weight.grad is not None
        assert attention.wv.weight.grad is not None
        assert attention.wo.weight.grad is not None
        assert x.grad is not None


class TestLlamaBlock:
    """Tests for LlamaBlock"""

    def test_init(self):
        config = LlamaConfig(n_embd=768, n_head=12, rmsnorm_eps=1e-5)
        block = LlamaBlock(config)

        # Check components
        from optimus_dl.modules.model.blocks.layer_norms import RMSNorm

        assert isinstance(block.ln_1, RMSNorm)
        assert isinstance(block.ln_2, RMSNorm)
        assert isinstance(block.attn, RotarySelfAttention)
        assert isinstance(block.mlp, SwiGLUMLP)

        # Check RMSNorm epsilon
        assert block.ln_1.eps == 1e-5
        assert block.ln_2.eps == 1e-5

    def test_forward_with_freqs_cis(self):
        config = LlamaConfig(n_embd=256, n_head=8, rmsnorm_eps=1e-6)
        block = LlamaBlock(config)
        block.eval()

        batch_size, seq_len = 2, 15
        x = torch.randn(batch_size, seq_len, 256)
        freqs_cis = precompute_freqs_cis(256 // 8, seq_len)

        output = block(x, freqs_cis)
        assert output.shape == (batch_size, seq_len, 256)

    def test_residual_connections(self):
        """Test that both residual connections work"""
        config = LlamaConfig(n_embd=256, n_head=8)
        block = LlamaBlock(config)
        block.eval()

        batch_size, seq_len = 1, 10
        x = torch.randn(batch_size, seq_len, 256)
        freqs_cis = precompute_freqs_cis(256 // 8, seq_len)

        # The forward should implement:
        # x = x + attn(ln_1(x), freqs_cis)
        # x = x + mlp(ln_2(x))
        output = block(x, freqs_cis)

        # Output should be different from input due to transformations
        assert not torch.allclose(output, x)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        config = LlamaConfig(n_embd=256, n_head=8, rmsnorm_eps=1e-6)
        block = LlamaBlock(config)

        batch_size, seq_len = 1, 10
        x = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        freqs_cis = precompute_freqs_cis(256 // 8, seq_len)

        output = block(x, freqs_cis)
        loss = output.sum()
        loss.backward()

        # Check gradients flow through all components
        assert block.ln_1.weight.grad is not None
        assert block.ln_2.weight.grad is not None
        assert block.attn.wq.weight.grad is not None
        assert block.mlp.w1.weight.grad is not None
        assert x.grad is not None


class TestLlamaConfig:
    """Tests for LlamaConfig"""

    def test_default_config(self):
        config = LlamaConfig()

        # Test LLaMA-specific defaults
        assert config.sequence_length == 16000
        assert config.rmsnorm_eps == 1e-5
        assert config.multiple_of == 256

        # Test inherited GPT defaults
        assert config.vocab_size == 50304
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768

    def test_custom_config(self):
        config = LlamaConfig(
            sequence_length=4096,
            rmsnorm_eps=1e-6,
            multiple_of=128,
            n_layer=24,
            n_head=16,
            n_embd=1024,
        )

        assert config.sequence_length == 4096
        assert config.rmsnorm_eps == 1e-6
        assert config.multiple_of == 128
        assert config.n_layer == 24
        assert config.n_head == 16
        assert config.n_embd == 1024


class TestLlama:
    """Tests for main Llama model"""

    def test_init(self):
        config = LlamaConfig(
            vocab_size=1000, sequence_length=2048, n_layer=2, n_head=4, n_embd=256
        )
        model = Llama(config)

        # Check components
        assert hasattr(model, "freqs_cis")
        assert hasattr(model, "head_dim")
        assert model.head_dim == 256 // 4

        # Check transformer components
        assert isinstance(model.transformer.wte, nn.Embedding)
        assert isinstance(model.transformer.drop, nn.Dropout)
        assert len(model.transformer.h) == 2

        from optimus_dl.modules.model.blocks.layer_norms import RMSNorm

        assert isinstance(model.transformer.ln_f, RMSNorm)

        # Check that all blocks are LlamaBlocks
        for block in model.transformer.h:
            assert isinstance(block, LlamaBlock)

    def test_freqs_cis_precomputation(self):
        config = LlamaConfig(sequence_length=1024, n_head=8, n_embd=512)
        model = Llama(config)

        # Check freqs_cis shape and properties
        head_dim = 512 // 8
        assert model.freqs_cis.shape == (1024, head_dim // 2, 2)
        assert model.freqs_cis.dtype == torch.float32

    def test_forward_no_position_embeddings(self):
        """Test that LLaMA doesn't use position embeddings (uses RoPE instead)"""
        config = LlamaConfig(
            vocab_size=1000, sequence_length=512, n_layer=2, n_head=4, n_embd=256
        )
        model = Llama(config)
        model.eval()

        # Check that there's no position embedding component
        assert not hasattr(model.transformer, "wpe")

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_freqs_cis_slicing(self):
        """Test that freqs_cis is properly sliced for different sequence lengths"""
        config = LlamaConfig(
            vocab_size=1000, sequence_length=1024, n_layer=2, n_head=4, n_embd=256
        )
        model = Llama(config)
        model.eval()

        # Test different sequence lengths
        seq_lengths = [1, 10, 50, 100]

        for seq_len in seq_lengths:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
            output = model(input_ids)

            assert output["logits"].shape == (1, seq_len, config.vocab_size)

    def test_sequence_length_assertion(self):
        """Test sequence length limit enforcement"""
        config = LlamaConfig(sequence_length=100)
        model = Llama(config)

        # Sequence longer than sequence_length should raise an error
        long_input = torch.randint(0, config.vocab_size, (1, 150))

        with pytest.raises(AssertionError):
            model(long_input)

    def test_weight_tying(self):
        """Test weight tying between token embeddings and output head"""
        # Explicitly set tie_word_embeddings=True
        config = LlamaConfig(tie_word_embeddings=True)
        model = Llama(config)

        assert model.transformer.wte.weight is model.lm_head.weight

    def test_gradient_flow(self):
        """Test gradient flow through the entire model"""
        config = LlamaConfig(
            vocab_size=100, sequence_length=512, n_layer=2, n_head=4, n_embd=256
        )
        model = Llama(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        output = model(input_ids)
        loss = output["logits"].sum()
        loss.backward()

        # Check key gradient flows
        assert model.transformer.wte.weight.grad is not None
        assert model.lm_head.weight.grad is not None

        for block in model.transformer.h:
            assert block.attn.wq.weight.grad is not None
            assert block.mlp.w1.weight.grad is not None

    def test_device_handling(self):
        """Test that freqs_cis is moved to the correct device"""
        config = LlamaConfig(
            vocab_size=100, sequence_length=512, n_layer=1, n_head=4, n_embd=256
        )
        model = Llama(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        model(input_ids)

        # freqs_cis should be on the same device as input
        assert model.freqs_cis.device.type == input_ids.device.type


class TestLlamaLiteArch:
    """Tests for llama_lite architecture preset"""

    def test_llama_lite_model_creation(self):
        config = llama_lite()
        model = Llama(config)

        # Test that the model can be created and used
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        output = model(input_ids)

        assert output["logits"].shape == (1, 5, config.vocab_size)


class TestLlamaIntegration:
    """Integration tests for LLaMA model components"""

    def test_rope_consistency_across_layers(self):
        """Test that RoPE is applied consistently across all layers"""
        config = LlamaConfig(
            vocab_size=100, sequence_length=512, n_layer=3, n_head=4, n_embd=256
        )
        model = Llama(config)
        model.eval()

        # All layers should receive the same freqs_cis
        input_ids = torch.randint(0, config.vocab_size, (1, 20))

        # Patch each block to capture the freqs_cis they receive
        received_freqs_cis = []

        def capture_freqs_cis(original_forward):
            def wrapper(x, freqs_cis):
                received_freqs_cis.append(freqs_cis)
                return original_forward(x, freqs_cis)

            return wrapper

        # Temporarily patch the forward methods
        original_forwards = []
        for block in model.transformer.h:
            original_forwards.append(block.forward)
            block.forward = capture_freqs_cis(block.forward)

        try:
            model(input_ids)

            # All blocks should receive the same freqs_cis
            assert len(received_freqs_cis) == config.n_layer
            for i in range(1, len(received_freqs_cis)):
                assert torch.equal(received_freqs_cis[0], received_freqs_cis[i])

        finally:
            # Restore original forward methods
            for i, block in enumerate(model.transformer.h):
                block.forward = original_forwards[i]

    def test_memory_efficiency(self):
        """Test memory usage with different sequence lengths"""
        config = LlamaConfig(
            vocab_size=1000, sequence_length=2048, n_layer=2, n_head=8, n_embd=512
        )
        model = Llama(config)
        model.eval()

        seq_lengths = [10, 50, 100, 500]

        for seq_len in seq_lengths:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output = model(input_ids)
            assert output["logits"].shape == (1, seq_len, config.vocab_size)

            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
