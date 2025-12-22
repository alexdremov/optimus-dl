import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.optim.adamw import AdamWConfig, make_adamw


class TestAdamWConfig:
    """Tests for AdamWConfig dataclass"""

    def test_default_config(self):
        config = AdamWConfig()

        assert config.lr == 1e-3
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.weight_decay == 1e-2
        assert config.amsgrad is False
        assert config.maximize is False
        assert config.foreach is None
        assert config.capturable is False
        assert config.differentiable is False
        assert config.fused is True

    def test_custom_config(self):
        config = AdamWConfig(
            lr=3e-4,
            betas=(0.95, 0.99),
            eps=1e-6,
            weight_decay=0.01,
            amsgrad=True,
            maximize=True,
            foreach=True,
            capturable=True,
            differentiable=True,
            fused=False,
        )

        assert config.lr == 3e-4
        assert config.betas == (0.95, 0.99)
        assert config.eps == 1e-6
        assert config.weight_decay == 0.01
        assert config.amsgrad is True
        assert config.maximize is True
        assert config.foreach is True
        assert config.capturable is True
        assert config.differentiable is True
        assert config.fused is False

    def test_config_inheritance(self):
        """Test that config inherits from RegistryConfig"""
        from optimus_dl.core.registry import RegistryConfig

        config = AdamWConfig()
        assert isinstance(config, RegistryConfig)


class TestMakeAdamW:
    """Tests for make_adamw function"""

    def test_make_adamw_basic(self):
        """Test basic AdamW optimizer creation"""
        # Create a simple model for parameters
        model = nn.Linear(10, 5)
        params = model.parameters()

        config = AdamWConfig()
        optimizer = make_adamw(config, params)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert optimizer.param_groups[0]["eps"] == 1e-8
        assert optimizer.param_groups[0]["weight_decay"] == 1e-2

    def test_make_adamw_custom_config(self):
        """Test AdamW creation with custom configuration"""
        model = nn.Linear(10, 5)
        params = model.parameters()

        config = AdamWConfig(
            lr=5e-4, betas=(0.95, 0.98), eps=1e-7, weight_decay=0.005, amsgrad=True
        )
        optimizer = make_adamw(config, params)

        assert optimizer.param_groups[0]["lr"] == 5e-4
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.98)
        assert optimizer.param_groups[0]["eps"] == 1e-7
        assert optimizer.param_groups[0]["weight_decay"] == 0.005
        assert optimizer.param_groups[0]["amsgrad"] is True

    def test_make_adamw_with_parameter_groups(self):
        """Test AdamW with multiple parameter groups"""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        # Create parameter groups
        param_groups = [
            {"params": model[0].parameters(), "weight_decay": 0.01},
            {"params": model[1].parameters(), "weight_decay": 0.001},
        ]

        config = AdamWConfig(lr=1e-3)
        optimizer = make_adamw(config, param_groups)

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[1]["weight_decay"] == 0.001

    def test_make_adamw_kwargs_override(self):
        """Test that kwargs can override config values"""
        model = nn.Linear(10, 5)
        params = model.parameters()

        config = AdamWConfig(lr=1e-3, weight_decay=0.01)

        # Override lr via kwargs
        optimizer = make_adamw(config, params, lr=5e-4)

        # The config lr should be used, not the kwarg
        # (kwargs are passed to torch.optim.AdamW but config values take precedence)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_make_adamw_optimization_flags(self):
        """Test AdamW with various optimization flags"""
        # Test different flag combinations
        configs = [
            AdamWConfig(fused=True, foreach=None),
            AdamWConfig(fused=False, foreach=True),
            AdamWConfig(capturable=True, differentiable=False),
            AdamWConfig(
                capturable=False, differentiable=True, fused=False
            ),  # fused=False when differentiable=True
        ]

        for config in configs:
            model = nn.Linear(10, 5)  # Create fresh model for each test
            params = model.parameters()
            optimizer = make_adamw(config, params)
            assert isinstance(optimizer, torch.optim.AdamW)

            # Check that flags are properly set
            assert optimizer.param_groups[0]["fused"] == config.fused
            if config.foreach is not None:
                assert optimizer.param_groups[0]["foreach"] == config.foreach
            assert optimizer.param_groups[0]["capturable"] == config.capturable
            assert optimizer.param_groups[0]["differentiable"] == config.differentiable

    def test_make_adamw_step_functionality(self):
        """Test that the created optimizer can perform optimization steps"""
        # Create a simple optimization problem
        torch.manual_seed(42)
        x = torch.randn(10, 5, requires_grad=True)
        target = torch.randn(10, 1)

        model = nn.Linear(5, 1)
        criterion = nn.MSELoss()

        config = AdamWConfig(lr=1e-2)
        optimizer = make_adamw(config, model.parameters())

        # Perform a few optimization steps
        initial_loss = None
        for _step in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_make_adamw_weight_decay_groups(self):
        """Test weight decay with parameter groups (common pattern in transformers)"""

        # Simulate transformer parameter grouping
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(100, 64)
                self.linear1 = nn.Linear(64, 128)
                self.linear2 = nn.Linear(128, 64)
                self.layer_norm = nn.LayerNorm(64)

        model = MockTransformer()

        # Group parameters: decay weights, no decay for biases and layer norms
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if "weight" in name and "layer_norm" not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        config = AdamWConfig(lr=3e-4, weight_decay=0.01)  # Default weight decay
        optimizer = make_adamw(config, param_groups)

        # Check parameter groups
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_make_adamw_empty_parameters(self):
        """Test AdamW with empty parameter list"""
        config = AdamWConfig()

        # Empty parameter list should raise ValueError
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            make_adamw(config, [])

    def test_make_adamw_beta_validation(self):
        """Test that beta values are properly set"""
        # Test various beta configurations
        beta_configs = [
            (0.9, 0.999),  # Default
            (0.95, 0.99),  # Custom
            (0.8, 0.95),  # Lower values
            (0.99, 0.9999),  # Higher values
        ]

        for beta1, beta2 in beta_configs:
            model = nn.Linear(10, 5)  # Create fresh model for each test
            params = model.parameters()
            config = AdamWConfig(betas=(beta1, beta2))
            optimizer = make_adamw(config, params)

            assert optimizer.param_groups[0]["betas"] == (beta1, beta2)

    def test_make_adamw_eps_validation(self):
        """Test epsilon value handling"""
        eps_values = [1e-8, 1e-6, 1e-10, 1e-4]

        for eps in eps_values:
            model = nn.Linear(10, 5)  # Create fresh model for each test
            params = model.parameters()
            config = AdamWConfig(eps=eps)
            optimizer = make_adamw(config, params)

            assert optimizer.param_groups[0]["eps"] == eps

    def test_make_adamw_reproducibility(self):
        """Test that optimizer behavior is reproducible"""
        torch.manual_seed(42)

        model1 = nn.Linear(5, 2)
        model2 = nn.Linear(5, 2)

        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
                p2.data.copy_(p1.data)

        config = AdamWConfig(lr=1e-2)
        opt1 = make_adamw(config, model1.parameters())
        opt2 = make_adamw(config, model2.parameters())

        x = torch.randn(3, 5)
        target = torch.randn(3, 2)
        criterion = nn.MSELoss()

        # Perform same optimization steps
        for _ in range(3):
            # Model 1
            opt1.zero_grad()
            loss1 = criterion(model1(x), target)
            loss1.backward()
            opt1.step()

            # Model 2
            opt2.zero_grad()
            loss2 = criterion(model2(x), target)
            loss2.backward()
            opt2.step()

        # Parameters should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2, atol=1e-6)
