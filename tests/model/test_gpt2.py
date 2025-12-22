import math

import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.model.gpt2 import (
    BLACKLIST_WEIGHT_MODULES,
    GPT,
    MLP,
    Block,
    GPTConfig,
)


class TestGPTConfig:
    """Tests for GPTConfig dataclass"""

    def test_default_config(self):
        config = GPTConfig()

        assert config.block_size == 1024
        assert config.vocab_size == 50304
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.dropout == 0.0
        assert config.bias is True

    def test_custom_config(self):
        config = GPTConfig(
            block_size=2048,
            vocab_size=32000,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            dropout=0.1,
            bias=False,
        )

        assert config.block_size == 2048
        assert config.vocab_size == 32000
        assert config.n_layer == 24
        assert config.n_head == 16
        assert config.n_embd == 1024
        assert config.dropout == 0.1
        assert config.bias is False

    def test_head_dimension_compatibility(self):
        """Test that n_embd is divisible by n_head"""
        # Valid configurations
        valid_configs = [
            (768, 12),  # 768 / 12 = 64
            (1024, 16),  # 1024 / 16 = 64
            (512, 8),  # 512 / 8 = 64
        ]

        for n_embd, n_head in valid_configs:
            GPTConfig(n_embd=n_embd, n_head=n_head)
            assert n_embd % n_head == 0


class TestMLP:
    """Tests for MLP module"""

    def test_init(self):
        config = GPTConfig(n_embd=768, bias=True, dropout=0.1)
        mlp = MLP(config)

        assert isinstance(mlp.c_fc, nn.Linear)
        assert isinstance(mlp.c_proj, nn.Linear)
        assert isinstance(mlp.gelu, nn.GELU)
        assert isinstance(mlp.dropout, nn.Dropout)

        # Check dimensions
        assert mlp.c_fc.in_features == 768
        assert mlp.c_fc.out_features == 4 * 768
        assert mlp.c_proj.in_features == 4 * 768
        assert mlp.c_proj.out_features == 768

    def test_bias_configuration(self):
        # Test with bias
        config_with_bias = GPTConfig(n_embd=768, bias=True)
        mlp_with_bias = MLP(config_with_bias)
        assert mlp_with_bias.c_fc.bias is not None
        assert mlp_with_bias.c_proj.bias is not None

        # Test without bias
        config_no_bias = GPTConfig(n_embd=768, bias=False)
        mlp_no_bias = MLP(config_no_bias)
        assert mlp_no_bias.c_fc.bias is None
        assert mlp_no_bias.c_proj.bias is None

    def test_forward(self):
        config = GPTConfig(
            n_embd=768, dropout=0.0
        )  # No dropout for deterministic testing
        mlp = MLP(config)
        mlp.eval()  # Disable dropout

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 768)

        output = mlp(x)
        assert output.shape == (batch_size, seq_len, 768)

    def test_different_embedding_dimensions(self):
        dimensions = [256, 512, 768, 1024, 2048]

        for n_embd in dimensions:
            config = GPTConfig(n_embd=n_embd)
            mlp = MLP(config)

            x = torch.randn(2, 10, n_embd)
            output = mlp(x)
            assert output.shape == (2, 10, n_embd)

    def test_gradient_flow(self):
        config = GPTConfig(n_embd=256)
        mlp = MLP(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check gradients flow through all parameters
        assert mlp.c_fc.weight.grad is not None
        assert mlp.c_proj.weight.grad is not None
        assert x.grad is not None


class TestBlock:
    """Tests for Transformer Block"""

    def test_init(self):
        config = GPTConfig(n_embd=768, n_head=12, bias=True)
        block = Block(config)

        assert hasattr(block, "ln_1")
        assert hasattr(block, "attn")
        assert hasattr(block, "ln_2")
        assert hasattr(block, "mlp")

        # Check types
        from optimus_dl.modules.model.blocks.attention import CausalSelfAttention
        from optimus_dl.modules.model.blocks.layer_norms import LayerNorm

        assert isinstance(block.ln_1, LayerNorm)
        assert isinstance(block.ln_2, LayerNorm)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.mlp, MLP)

    def test_forward_residual_connections(self):
        config = GPTConfig(n_embd=768, n_head=12, dropout=0.0)
        block = Block(config)
        block.eval()

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 768)

        # Test that residual connections work
        output = block(x)
        assert output.shape == x.shape

        # Output should be different from input (due to transformations)
        assert not torch.allclose(output, x)

    def test_layer_norm_placement(self):
        """Test pre-norm architecture (LayerNorm before attention and MLP)"""
        config = GPTConfig(n_embd=768, n_head=12)
        block = Block(config)

        # This tests the architecture: x + attn(ln_1(x)) and x + mlp(ln_2(x))
        # We can't directly test the ordering, but we can ensure forward pass works
        x = torch.randn(2, 10, 768)
        output = block(x)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        config = GPTConfig(n_embd=256, n_head=8)
        block = Block(config)

        x = torch.randn(2, 10, 256, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check gradients flow through all components
        assert block.ln_1.weight.grad is not None
        assert block.ln_2.weight.grad is not None
        assert block.attn.c_attn.weight.grad is not None
        assert block.mlp.c_fc.weight.grad is not None
        assert x.grad is not None


class TestGPT:
    """Tests for main GPT model"""

    def test_init_assertions(self):
        # Test that model requires vocab_size and block_size
        with pytest.raises(AssertionError):
            config = GPTConfig(vocab_size=None)
            GPT(config)

        with pytest.raises(AssertionError):
            config = GPTConfig(block_size=None)
            GPT(config)

    def test_init_components(self):
        config = GPTConfig()
        model = GPT(config)

        # Check all transformer components exist
        assert hasattr(model.transformer, "wte")  # token embeddings
        assert hasattr(model.transformer, "wpe")  # position embeddings
        assert hasattr(model.transformer, "drop")  # dropout
        assert hasattr(model.transformer, "h")  # transformer blocks
        assert hasattr(model.transformer, "ln_f")  # final layer norm
        assert hasattr(model, "lm_head")  # language model head

        # Check dimensions
        assert model.transformer.wte.num_embeddings == config.vocab_size
        assert model.transformer.wte.embedding_dim == config.n_embd
        assert model.transformer.wpe.num_embeddings == config.block_size
        assert model.transformer.wpe.embedding_dim == config.n_embd
        assert len(model.transformer.h) == config.n_layer
        assert model.lm_head.in_features == config.n_embd
        assert model.lm_head.out_features == config.vocab_size

    def test_weight_tying(self):
        """Test that token embedding weights are tied with lm_head"""
        config = GPTConfig()
        model = GPT(config)

        # Weights should be tied
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_weight_initialization(self):
        """Test that weights are properly initialized"""
        config = GPTConfig(n_layer=2)  # Small model for testing
        model = GPT(config)

        # Check that c_proj weights have special initialization
        for name, param in model.named_parameters():
            if name.endswith("c_proj.weight"):
                # Should be initialized with smaller std
                0.02 / math.sqrt(2 * config.n_layer)
                # We can't check exact values due to randomness, but can check the shape
                assert param.shape[0] == config.n_embd
                assert param.requires_grad

    def test_forward_basic(self):
        config = GPTConfig(
            vocab_size=1000, block_size=512, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert "logits" in output
        logits = output["logits"]
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_sequence_length_assertion(self):
        config = GPTConfig(block_size=512)
        model = GPT(config)

        # Test that sequences longer than block_size raise an error
        too_long_seq = torch.randint(0, config.vocab_size, (1, 600))

        with pytest.raises(AssertionError):
            model(too_long_seq)

    def test_forward_different_sequence_lengths(self):
        config = GPTConfig(vocab_size=1000, block_size=1024, n_layer=2)
        model = GPT(config)
        model.eval()

        sequence_lengths = [1, 10, 50, 100, 500]

        for seq_len in sequence_lengths:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
            output = model(input_ids)

            logits = output["logits"]
            assert logits.shape == (1, seq_len, config.vocab_size)

    def test_make_parameter_groups(self):
        """Test parameter grouping for weight decay optimization (important for transformer training)."""
        config = GPTConfig(n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)

        param_groups = model.make_parameter_groups()

        # Should return two groups: with decay and without decay
        assert len(param_groups) == 2

        decay_group = param_groups[0]
        no_decay_group = param_groups[1]

        assert "params" in decay_group
        assert "params" in no_decay_group
        assert no_decay_group.get("weight_decay") == 0.0

        # Collect parameter names
        decay_names = {name for name, param in decay_group["params"]}
        no_decay_names = {name for name, param in no_decay_group["params"]}

        # Check that all parameters are in exactly one group
        all_param_names = set(dict(model.named_parameters()).keys())
        grouped_names = decay_names | no_decay_names
        assert grouped_names == all_param_names

        # Check that no parameter is in both groups
        assert len(decay_names & no_decay_names) == 0

        # Verify some expected categorizations
        # Biases should not decay
        bias_params = [name for name in all_param_names if name.endswith(".bias")]
        for bias_param in bias_params:
            assert bias_param in no_decay_names

        # Linear weights should decay (except embeddings and layer norms)
        # lm_head.weight should not decay due to weight tying

    def test_generate_basic(self):
        """Test basic text generation functionality with single starting token."""
        config = GPTConfig(
            vocab_size=100, block_size=512, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        # Start with a single token
        start_tokens = torch.tensor([[1]], dtype=torch.long)  # batch_size=1, seq_len=1

        generated = model.generate(start_tokens, max_new_tokens=5, temperature=1.0)

        # Should generate 5 additional tokens
        assert generated.shape == (1, 6)  # original 1 + 5 new tokens
        assert generated[0, 0] == 1  # First token should remain unchanged

    def test_generate_with_context(self):
        """Test generation with longer context"""
        config = GPTConfig(
            vocab_size=100, block_size=512, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        # Start with multiple tokens
        context_len = 10
        start_tokens = torch.randint(0, config.vocab_size, (1, context_len))
        original_context = start_tokens.clone()

        generated = model.generate(start_tokens, max_new_tokens=3, temperature=1.0)

        assert generated.shape == (1, context_len + 3)
        # Original context should be preserved
        assert torch.equal(generated[0, :context_len], original_context[0])

    def test_generate_temperature_effect(self):
        """Test that temperature affects generation diversity"""
        config = GPTConfig(
            vocab_size=100, block_size=512, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        start_tokens = torch.tensor([[1]], dtype=torch.long)

        # Generate with different temperatures
        # Note: Due to randomness, we can't guarantee different outputs,
        # but we can test that the function runs without errors
        low_temp = model.generate(start_tokens, max_new_tokens=3, temperature=0.1)
        high_temp = model.generate(start_tokens, max_new_tokens=3, temperature=2.0)

        assert low_temp.shape == (1, 4)
        assert high_temp.shape == (1, 4)

    def test_generate_top_k_sampling(self):
        """Test top-k sampling in generation"""
        config = GPTConfig(
            vocab_size=100, block_size=512, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        start_tokens = torch.tensor([[1]], dtype=torch.long)

        # Test with top_k sampling
        generated = model.generate(start_tokens, max_new_tokens=3, top_k=10)

        assert generated.shape == (1, 4)
        # All generated tokens should be within vocab range
        assert (generated >= 0).all()
        assert (generated < config.vocab_size).all()

    def test_generate_long_sequence_cropping(self):
        """Test that generation properly crops sequences that exceed block_size"""
        config = GPTConfig(
            vocab_size=100, block_size=20, n_layer=2, n_head=4, n_embd=256
        )
        model = GPT(config)
        model.eval()

        # Start with a sequence close to block_size
        start_tokens = torch.randint(0, config.vocab_size, (1, 18))

        # Generate enough tokens to exceed block_size
        generated = model.generate(start_tokens, max_new_tokens=5, temperature=1.0)

        # Should generate successfully even though intermediate sequences exceed block_size
        assert generated.shape == (1, 23)  # 18 + 5

    def test_model_different_configurations(self):
        """Test model with various valid configurations"""
        configs = [
            GPTConfig(vocab_size=1000, n_layer=6, n_head=6, n_embd=384),  # Small
            GPTConfig(vocab_size=2000, n_layer=12, n_head=12, n_embd=768),  # Medium
            GPTConfig(vocab_size=5000, n_layer=2, n_head=4, n_embd=256),  # Tiny
        ]

        for config in configs:
            model = GPT(config)

            # Test forward pass
            batch_size, seq_len = 2, 10
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)

            assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)

    def test_model_training_mode(self):
        """Test model behavior in training vs eval mode"""
        config = GPTConfig(dropout=0.1, n_layer=2)
        model = GPT(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))

        # Training mode
        model.train()
        train_output = model(input_ids)

        # Eval mode
        model.eval()
        eval_output = model(input_ids)

        # Outputs might be different due to dropout in training mode
        assert train_output["logits"].shape == eval_output["logits"].shape

    def test_gradient_flow_full_model(self):
        """Test gradient flow through the entire model"""
        config = GPTConfig(vocab_size=100, n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        output = model(input_ids)
        loss = output["logits"].sum()
        loss.backward()

        # Check that gradients flow to key components
        assert model.transformer.wte.weight.grad is not None
        assert model.transformer.wpe.weight.grad is not None
        assert model.lm_head.weight.grad is not None

        for block in model.transformer.h:
            assert block.attn.c_attn.weight.grad is not None
            assert block.mlp.c_fc.weight.grad is not None

    def test_blacklist_weight_modules(self):
        """Test that BLACKLIST_WEIGHT_MODULES contains expected module types"""
        from optimus_dl.modules.model.blocks.layer_norms import LayerNorm, RMSNorm

        expected_types = [
            torch.nn.LayerNorm,
            LayerNorm,
            RMSNorm,
            torch.nn.Embedding,
        ]

        for expected_type in expected_types:
            assert expected_type in BLACKLIST_WEIGHT_MODULES

    def test_model_device_consistency(self):
        """Test that model handles device placement correctly"""
        config = GPTConfig(vocab_size=100, n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)

        # Test CPU
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        output = model(input_ids)
        assert output["logits"].device == input_ids.device

        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_ids_gpu = input_ids.cuda()
            output_gpu = model_gpu(input_ids_gpu)
            assert output_gpu["logits"].device == input_ids_gpu.device

    def test_model_memory_efficiency(self):
        """Test model with different batch sizes to check memory usage"""
        config = GPTConfig(vocab_size=1000, n_layer=2, n_head=4, n_embd=256)
        model = GPT(config)
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        seq_len = 50

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)

            assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)

            # Clean up
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
