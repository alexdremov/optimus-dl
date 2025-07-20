import math
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.lr_scheduler.cosine_annealing import (
    CosineAnnealingLR,
    CosineAnnealingLRConfig,
)


class TestCosineAnnealingLRConfig:
    """Tests for CosineAnnealingLRConfig"""

    def test_default_config(self):
        config = CosineAnnealingLRConfig()

        assert config.T_max == 1000
        assert config.eta_min == 0.0
        assert config.last_epoch == -1

    def test_custom_config(self):
        config = CosineAnnealingLRConfig(T_max=500, eta_min=1e-6, last_epoch=10)

        assert config.T_max == 500
        assert config.eta_min == 1e-6
        assert config.last_epoch == 10

    def test_config_inheritance(self):
        """Test that config inherits from BaseLRSchedulerConfig"""
        from optimus_dl.modules.lr_scheduler.base import BaseLRSchedulerConfig

        config = CosineAnnealingLRConfig()
        assert isinstance(config, BaseLRSchedulerConfig)


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler"""

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

    def test_init(self):
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=100, eta_min=1e-5)

        scheduler = CosineAnnealingLR(config, optimizer, iterations=100)

        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-5
        assert scheduler._step_count == 0  # last_epoch + 1 = -1 + 1 = 0
        assert scheduler.base_lrs == [1e-2]

    def test_init_with_last_epoch(self):
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=100, last_epoch=5)

        scheduler = CosineAnnealingLR(config, optimizer, iterations=100)

        assert scheduler._step_count == 6  # last_epoch + 1 = 5 + 1 = 6

    def test_get_lr_at_start(self):
        """Test learning rate at step 0"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=100, eta_min=1e-5)

        scheduler = CosineAnnealingLR(config, optimizer, iterations=100)

        # At step 0, should return base learning rates
        lrs = scheduler.get_lr()
        assert lrs == [1e-2]

    def test_get_lr_cosine_formula(self):
        """Test cosine annealing formula at various points"""
        base_lr = 1e-2
        eta_min = 1e-5
        T_max = 100

        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=T_max, eta_min=eta_min)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=T_max)

        test_steps = [0, 25, 50, 75, 100]

        for step in test_steps:
            scheduler._step_count = step
            lrs = scheduler.get_lr()

            if step == 0:
                # At step 0, should return base LR
                expected_lr = base_lr
            else:
                # Cosine annealing formula
                expected_lr = (
                    eta_min
                    + (base_lr - eta_min) * (1 + math.cos(math.pi * step / T_max)) / 2
                )

            assert len(lrs) == 1
            if step == 0:
                assert lrs[0] == expected_lr
            else:
                assert abs(lrs[0] - expected_lr) < 1e-8

    def test_get_lr_at_half_period(self):
        """Test LR at half period (should be eta_min)"""
        base_lr = 1e-2
        eta_min = 1e-5
        T_max = 100

        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=T_max, eta_min=eta_min)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=T_max)

        # At T_max/2, cos(π) = -1, so LR should be eta_min
        scheduler._step_count = T_max // 2
        lrs = scheduler.get_lr()

        expected_lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * 0.5)) / 2
        assert abs(lrs[0] - expected_lr) < 1e-8

    def test_get_lr_at_full_period(self):
        """Test LR at full period (should return to base_lr)"""
        base_lr = 1e-2
        eta_min = 1e-5
        T_max = 100

        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=T_max, eta_min=eta_min)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=T_max)

        # At T_max, cos(π) = -1, so LR should approach base_lr
        scheduler._step_count = T_max
        lrs = scheduler.get_lr()

        expected_lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi)) / 2
        assert abs(lrs[0] - expected_lr) < 1e-8

    def test_step_integration(self):
        """Test full step() integration"""
        base_lr = 1e-2
        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=10, eta_min=1e-5)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=10)

        # Test first few steps
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == base_lr

        scheduler.step()
        step1_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()
        step2_lr = optimizer.param_groups[0]["lr"]

        # Learning rate should decrease initially
        assert step1_lr < base_lr
        assert step2_lr < step1_lr

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups"""
        base_lrs = [1e-2, 2e-2, 3e-2]
        models = [nn.Linear(10, 5) for _ in range(3)]
        param_groups = [
            {"params": model.parameters(), "lr": lr}
            for model, lr in zip(models, base_lrs)
        ]
        optimizer = torch.optim.SGD(param_groups)

        config = CosineAnnealingLRConfig(T_max=20, eta_min=1e-5)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=20)

        # Step and check all groups are updated
        scheduler.step()

        lrs = scheduler.get_last_lr()
        assert len(lrs) == 3

        # All should be less than their base LRs after first step
        for lr, base_lr in zip(lrs, base_lrs):
            assert lr < base_lr

    def test_state_dict(self):
        """Test state dictionary functionality"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=50, eta_min=1e-6)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=50)

        scheduler._step_count = 10

        state = scheduler.state_dict()

        assert "step_count" in state
        assert "base_lrs" in state
        assert "T_max" in state
        assert "eta_min" in state

        assert state["step_count"] == 10
        assert state["T_max"] == 50
        assert state["eta_min"] == 1e-6

    def test_load_state_dict(self):
        """Test loading state dictionary"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=100, eta_min=1e-5)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=100)

        state = {"step_count": 25, "base_lrs": [2e-2], "T_max": 200, "eta_min": 1e-6}

        scheduler.load_state_dict(state)

        assert scheduler._step_count == 25
        assert scheduler.base_lrs == [2e-2]
        assert scheduler.T_max == 200
        assert scheduler.eta_min == 1e-6

    def test_eta_min_zero(self):
        """Test with eta_min = 0"""
        base_lr = 1e-2
        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=10, eta_min=0.0)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=10)

        # At half period, LR should be 0
        scheduler._step_count = 5  # T_max / 2
        lrs = scheduler.get_lr()

        expected_lr = 0.0 + (base_lr - 0.0) * (1 + math.cos(math.pi * 0.5)) / 2
        assert abs(lrs[0] - expected_lr) < 1e-8

    def test_eta_min_equals_base_lr(self):
        """Test with eta_min equal to base_lr (constant LR)"""
        base_lr = 1e-2
        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=10, eta_min=base_lr)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=10)

        # LR should remain constant
        for step in range(1, 11):
            scheduler._step_count = step
            lrs = scheduler.get_lr()
            assert abs(lrs[0] - base_lr) < 1e-8

    def test_small_t_max(self):
        """Test with very small T_max"""
        optimizer = self.create_optimizer(lr=1e-2)
        config = CosineAnnealingLRConfig(T_max=2, eta_min=1e-5)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=2)

        # Should still work correctly
        scheduler.step()
        lr1 = scheduler.get_last_lr()[0]

        scheduler.step()
        lr2 = scheduler.get_last_lr()[0]

        # LR should change
        assert lr1 != lr2

    def test_large_t_max(self):
        """Test with large T_max"""
        base_lr = 1e-2
        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=10000, eta_min=1e-5)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=10000)

        # Early steps should have very small changes
        scheduler.step()
        lr1 = scheduler.get_last_lr()[0]

        scheduler.step()
        lr2 = scheduler.get_last_lr()[0]

        # Changes should be small for large T_max
        change_rate = abs(lr2 - lr1) / lr1
        assert change_rate < 0.01  # Less than 1% change per step

    def test_inheritance(self):
        """Test that CosineAnnealingLR inherits from BaseLRScheduler"""
        from optimus_dl.modules.lr_scheduler.base import BaseLRScheduler

        optimizer = self.create_optimizer()
        config = CosineAnnealingLRConfig()
        scheduler = CosineAnnealingLR(config, optimizer, iterations=100)

        assert isinstance(scheduler, BaseLRScheduler)

    def test_mathematical_properties(self):
        """Test mathematical properties of cosine annealing"""
        base_lr = 1.0
        eta_min = 0.0
        T_max = 100

        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=T_max, eta_min=eta_min)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=T_max)

        # Test monotonic decrease in first half
        prev_lr = base_lr
        for step in range(1, T_max // 2 + 1):
            scheduler._step_count = step
            current_lr = scheduler.get_lr()[0]
            assert (
                current_lr <= prev_lr
            ), f"LR should decrease in first half, step {step}"
            prev_lr = current_lr

        # Test that LR at T_max is eta_min (minimum)
        scheduler._step_count = T_max
        min_lr = scheduler.get_lr()[0]
        assert abs(min_lr - eta_min) < 1e-8, "LR at T_max should be eta_min"

    def test_minimum_maximum_values(self):
        """Test that LR stays within [eta_min, base_lr] bounds"""
        base_lr = 1e-2
        eta_min = 1e-5

        optimizer = self.create_optimizer(lr=base_lr)
        config = CosineAnnealingLRConfig(T_max=50, eta_min=eta_min)
        scheduler = CosineAnnealingLR(config, optimizer, iterations=50)

        # Test many steps
        for step in range(1, 101):  # Test beyond T_max too
            scheduler._step_count = step
            lr = scheduler.get_lr()[0]

            # LR should be within bounds (allowing small numerical errors)
            assert lr >= eta_min - 1e-10
            assert lr <= base_lr + 1e-10
