import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.lr_scheduler.linear_warmup import (
    LinearWarmupLR,
    LinearWarmupLRConfig,
)


class TestLinearWarmupLRConfig:
    """Tests for LinearWarmupLRConfig"""

    def test_default_config(self):
        config = LinearWarmupLRConfig()

        assert config.warmup_steps is None
        assert config.warmup_percent == 0.05  # 5%
        assert config.target_lr is None
        assert config.start_lr == 0.0

    def test_custom_config(self):
        config = LinearWarmupLRConfig(
            warmup_steps=1000, warmup_percent=0.1, target_lr=1e-3, start_lr=1e-6
        )

        assert config.warmup_steps == 1000
        assert config.warmup_percent == 0.1
        assert config.target_lr == 1e-3
        assert config.start_lr == 1e-6

    def test_config_inheritance(self):
        """Test that config inherits from BaseLRSchedulerConfig"""
        from optimus_dl.modules.lr_scheduler.base import BaseLRSchedulerConfig

        config = LinearWarmupLRConfig()
        assert isinstance(config, BaseLRSchedulerConfig)


class TestLinearWarmupLR:
    """Tests for LinearWarmupLR scheduler"""

    def create_optimizer(self, lr=1e-3, num_groups=1):
        """Helper to create optimizer"""
        models = [nn.Linear(10, 5) for _ in range(num_groups)]
        if num_groups == 1:
            return torch.optim.SGD(models[0].parameters(), lr=lr)
        else:
            param_groups = [
                {"params": model.parameters(), "lr": lr} for model in models
            ]
            return torch.optim.SGD(param_groups)

    def test_init_with_warmup_steps(self):
        """Test initialization with explicit warmup_steps"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=100, start_lr=1e-5)

        scheduler = LinearWarmupLR(config, optimizer, iterations=1000)

        assert scheduler.warmup_steps == 100
        assert scheduler.start_lr == 1e-5
        assert scheduler.target_lrs == [1e-2]  # Uses base_lr as target
        assert scheduler.base_lrs == [1e-2]

    def test_init_with_warmup_percent(self):
        """Test initialization with warmup_percent"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_percent=0.1, start_lr=1e-5)

        scheduler = LinearWarmupLR(config, optimizer, iterations=1000)

        assert scheduler.warmup_steps == 100  # 10% of 1000
        assert scheduler.start_lr == 1e-5
        assert scheduler.target_lrs == [1e-2]

    def test_init_with_target_lr(self):
        """Test initialization with explicit target_lr"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=50, target_lr=2e-3, start_lr=1e-5)

        scheduler = LinearWarmupLR(config, optimizer, iterations=1000)

        assert scheduler.target_lrs == [2e-3]  # Uses explicit target_lr

    def test_init_validation_error(self):
        """Test that initialization fails without warmup_steps or warmup_percent"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=None, warmup_percent=None)

        with pytest.raises(
            ValueError, match="Either warmup_steps or warmup_percent must be set"
        ):
            LinearWarmupLR(config, optimizer, iterations=1000)

    def test_get_lr_during_warmup(self):
        """Test learning rate calculation during warmup phase"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=10, start_lr=1e-5, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Test various points during warmup
        test_cases = [
            (0, 1e-5),  # Step 0: start_lr
            (1, 1e-5 + (1e-2 - 1e-5) * 0.1),  # Step 1: 10% progress
            (5, 1e-5 + (1e-2 - 1e-5) * 0.5),  # Step 5: 50% progress
            (10, 1e-2),  # Step 10: target_lr (end of warmup)
        ]

        for step, expected_lr in test_cases:
            scheduler._step_count = step
            lrs = scheduler.get_lr()

            assert len(lrs) == 1
            assert abs(lrs[0] - expected_lr) < 1e-8

    def test_get_lr_after_warmup(self):
        """Test learning rate after warmup phase (should maintain target_lr)"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=10, start_lr=1e-5, target_lr=2e-3)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Test points after warmup
        post_warmup_steps = [11, 20, 50, 100]

        for step in post_warmup_steps:
            scheduler._step_count = step
            lrs = scheduler.get_lr()

            assert len(lrs) == 1
            assert lrs[0] == 2e-3  # Should maintain target_lr

    def test_step_integration(self):
        """Test full step() integration"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=5, start_lr=0.0, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Track LR progression - get LR after each step
        lrs = []
        for _ in range(10):
            scheduler.step()  # Step first to update LR
            lrs.append(optimizer.param_groups[0]["lr"])

        # During warmup (steps 1-5), LR should increase
        for i in range(4):  # Compare consecutive LRs in warmup
            assert (
                lrs[i] <= lrs[i + 1]
            ), f"LR should increase during warmup: {lrs[i]} <= {lrs[i+1]}"

        # After warmup (steps 6+), LR should be constant at target
        target_lr = 1e-2
        for i in range(5, 9):  # After warmup
            assert (
                abs(lrs[i] - target_lr) < 1e-8
            ), f"LR should be target_lr after warmup: {lrs[i]} vs {target_lr}"

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups"""
        base_lrs = [1e-2, 2e-2, 3e-2]
        models = [nn.Linear(10, 5) for _ in range(3)]
        param_groups = [
            {"params": model.parameters(), "lr": lr}
            for model, lr in zip(models, base_lrs)
        ]
        optimizer = torch.optim.SGD(param_groups)

        config = LinearWarmupLRConfig(warmup_steps=10, start_lr=0.0)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Target LRs should use base_lrs since target_lr is None
        assert scheduler.target_lrs == base_lrs

        # Test mid-warmup
        scheduler._step_count = 5  # 50% through warmup
        lrs = scheduler.get_lr()

        assert len(lrs) == 3
        expected_lrs = [0.0 + (base_lr - 0.0) * 0.5 for base_lr in base_lrs]

        for lr, expected_lr in zip(lrs, expected_lrs):
            assert abs(lr - expected_lr) < 1e-8

    def test_multiple_param_groups_with_target_lr(self):
        """Test multiple param groups with explicit target_lr"""
        base_lrs = [1e-2, 2e-2, 3e-2]
        models = [nn.Linear(10, 5) for _ in range(3)]
        param_groups = [
            {"params": model.parameters(), "lr": lr}
            for model, lr in zip(models, base_lrs)
        ]
        optimizer = torch.optim.SGD(param_groups)

        target_lr = 5e-3
        config = LinearWarmupLRConfig(
            warmup_steps=10, start_lr=0.0, target_lr=target_lr
        )
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # All param groups should use the same target_lr
        assert scheduler.target_lrs == [target_lr] * 3

    def test_zero_warmup_steps(self):
        """Test with zero warmup steps"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=0, start_lr=1e-5, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Should immediately return target_lr
        lrs = scheduler.get_lr()
        assert lrs == [1e-2]

        scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs == [1e-2]

    def test_single_warmup_step(self):
        """Test with single warmup step"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=1, start_lr=0.0, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Step 0: should be start_lr
        scheduler._step_count = 0
        lrs = scheduler.get_lr()
        assert lrs == [0.0]

        # Step 1: should be target_lr
        scheduler._step_count = 1
        lrs = scheduler.get_lr()
        assert lrs == [1e-2]

    def test_warmup_percent_calculation(self):
        """Test warmup_percent calculation with different iteration counts"""
        test_cases = [
            (1000, 0.05, 50),  # 5% of 1000 = 50
            (200, 0.1, 20),  # 10% of 200 = 20
            (50, 0.2, 10),  # 20% of 50 = 10
            (33, 0.1, 3),  # 10% of 33 = 3.3 -> 3 (int conversion)
        ]

        for total_iterations, warmup_percent, expected_steps in test_cases:
            optimizer = self.create_optimizer(lr=1e-3)
            config = LinearWarmupLRConfig(warmup_percent=warmup_percent)
            scheduler = LinearWarmupLR(config, optimizer, iterations=total_iterations)

            assert scheduler.warmup_steps == expected_steps

    def test_state_dict(self):
        """Test state dictionary functionality"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=20, start_lr=1e-5, target_lr=2e-3)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        scheduler._step_count = 10

        state = scheduler.state_dict()

        assert "step_count" in state
        assert "base_lrs" in state
        assert "warmup_steps" in state
        assert "start_lr" in state
        assert "target_lrs" in state

        assert state["step_count"] == 10
        assert state["warmup_steps"] == 20
        assert state["start_lr"] == 1e-5
        assert state["target_lrs"] == [2e-3]

    def test_load_state_dict(self):
        """Test loading state dictionary"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=10)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        state = {
            "step_count": 5,
            "base_lrs": [2e-2],
            "warmup_steps": 15,
            "start_lr": 1e-6,
            "target_lrs": [3e-3],
        }

        scheduler.load_state_dict(state)

        assert scheduler._step_count == 5
        assert scheduler.base_lrs == [2e-2]
        assert scheduler.warmup_steps == 15
        assert scheduler.start_lr == 1e-6
        assert scheduler.target_lrs == [3e-3]

    def test_start_lr_equals_target_lr(self):
        """Test with start_lr equal to target_lr (constant LR)"""
        lr = 1e-3
        optimizer = self.create_optimizer(lr=lr)
        config = LinearWarmupLRConfig(warmup_steps=10, start_lr=lr, target_lr=lr)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # LR should remain constant throughout
        for step in range(20):
            scheduler._step_count = step
            lrs = scheduler.get_lr()
            assert lrs == [lr]

    def test_negative_start_lr(self):
        """Test with negative start_lr (unusual but should work)"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=10, start_lr=-1e-5, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # At step 0, should return start_lr
        scheduler._step_count = 0
        lrs = scheduler.get_lr()
        assert lrs == [-1e-5]

        # Should still interpolate correctly
        scheduler._step_count = 5  # 50% through warmup
        lrs = scheduler.get_lr()
        expected_lr = -1e-5 + (1e-2 - (-1e-5)) * 0.5
        assert abs(lrs[0] - expected_lr) < 1e-8

    def test_large_warmup_steps(self):
        """Test with warmup_steps larger than total iterations"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=200, start_lr=0.0, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        # Should still work, just won't reach target_lr within total iterations
        scheduler._step_count = 50  # 25% through warmup
        lrs = scheduler.get_lr()
        expected_lr = 0.0 + (1e-2 - 0.0) * (50 / 200)
        assert abs(lrs[0] - expected_lr) < 1e-8

    def test_inheritance(self):
        """Test that LinearWarmupLR inherits from BaseLRScheduler"""
        from optimus_dl.modules.lr_scheduler.base import BaseLRScheduler

        optimizer = self.create_optimizer()
        config = LinearWarmupLRConfig(warmup_steps=10)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        assert isinstance(scheduler, BaseLRScheduler)

    def test_warmup_progression_monotonic(self):
        """Test that LR increases monotonically during warmup"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=20, start_lr=1e-5, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=100)

        previous_lr = None
        for step in range(21):  # Include end of warmup
            scheduler._step_count = step
            lr = scheduler.get_lr()[0]

            if previous_lr is not None:
                if step <= 20:  # During warmup
                    assert (
                        lr >= previous_lr
                    ), f"LR decreased at step {step}: {lr} < {previous_lr}"

            previous_lr = lr

    def test_precision_at_boundaries(self):
        """Test precision at warmup boundaries"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = LinearWarmupLRConfig(warmup_steps=100, start_lr=1e-6, target_lr=1e-2)
        scheduler = LinearWarmupLR(config, optimizer, iterations=1000)

        # Test exact boundaries
        scheduler._step_count = 0
        lr_start = scheduler.get_lr()[0]
        assert abs(lr_start - 1e-6) < 1e-12

        scheduler._step_count = 100
        lr_end = scheduler.get_lr()[0]
        assert abs(lr_end - 1e-2) < 1e-12

        scheduler._step_count = 101
        lr_post = scheduler.get_lr()[0]
        assert abs(lr_post - 1e-2) < 1e-12

    def test_fractional_warmup_percent(self):
        """Test with fractional warmup percentages"""
        optimizer = self.create_optimizer(lr=1e-3)

        # Test with percentage that results in fractional steps
        config = LinearWarmupLRConfig(warmup_percent=0.075)  # 7.5%
        scheduler = LinearWarmupLR(config, optimizer, iterations=1000)

        # 7.5% of 1000 = 75.0 -> 75 steps
        assert scheduler.warmup_steps == 75

        # Test with very small percentage
        config2 = LinearWarmupLRConfig(warmup_percent=0.001)  # 0.1%
        scheduler2 = LinearWarmupLR(config2, optimizer, iterations=1000)

        # 0.1% of 1000 = 1.0 -> 1 step
        assert scheduler2.warmup_steps == 1
