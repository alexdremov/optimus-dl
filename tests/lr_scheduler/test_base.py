import torch
import pytest
import torch.nn as nn

from optimus_dl.modules.lr_scheduler.base import (
    BaseLRScheduler,
    BaseLRSchedulerConfig,
)


class MockLRScheduler(BaseLRScheduler):
    """Mock scheduler for testing base functionality"""

    def __init__(self, optimizer, fixed_lr=1e-3):
        super().__init__(optimizer)
        self.fixed_lr = fixed_lr
        self.set()

    def get_lr(self):
        return [self.fixed_lr] * len(self.optimizer.param_groups)


class TestBaseLRSchedulerConfig:
    """Tests for BaseLRSchedulerConfig"""

    def test_default_config(self):
        config = BaseLRSchedulerConfig()
        # Base config should be empty
        assert isinstance(config, BaseLRSchedulerConfig)

    def test_config_inheritance(self):
        """Test that config inherits from RegistryConfig"""
        from optimus_dl.core.registry import RegistryConfig

        config = BaseLRSchedulerConfig()
        assert isinstance(config, RegistryConfig)


class TestBaseLRScheduler:
    """Tests for BaseLRScheduler base class"""

    def create_optimizer(self, lr=1e-3, num_groups=1):
        """Helper to create optimizer with specified parameters"""
        models = [nn.Linear(10, 5) for _ in range(num_groups)]
        if num_groups == 1:
            return torch.optim.SGD(models[0].parameters(), lr=lr)
        else:
            param_groups = [
                {"params": model.parameters(), "lr": lr} for model in models
            ]
            return torch.optim.SGD(param_groups)

    def test_init_single_param_group(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer)

        assert scheduler.optimizer is optimizer
        assert scheduler._step_count == 0
        assert scheduler.base_lrs == [1e-3]

    def test_init_multiple_param_groups(self):
        optimizer = self.create_optimizer(lr=2e-3, num_groups=3)
        scheduler = MockLRScheduler(optimizer)

        assert len(scheduler.base_lrs) == 3
        assert all(lr == 2e-3 for lr in scheduler.base_lrs)

    def test_step_functionality(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer, fixed_lr=5e-4)

        # Initial state
        assert scheduler._step_count == 0
        assert optimizer.param_groups[0]["lr"] == 5e-4

        # After step
        scheduler.step()
        assert scheduler._step_count == 1
        assert optimizer.param_groups[0]["lr"] == 5e-4

    def test_step_multiple_groups(self):
        optimizer = self.create_optimizer(lr=1e-3, num_groups=2)
        scheduler = MockLRScheduler(optimizer, fixed_lr=2e-4)

        scheduler.step()

        for group in optimizer.param_groups:
            assert group["lr"] == 2e-4

    def test_get_last_lr(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer, fixed_lr=3e-4)

        # Before step
        last_lrs = scheduler.get_last_lr()
        assert last_lrs == [3e-4]

        # After step
        scheduler.step()
        last_lrs = scheduler.get_last_lr()
        assert last_lrs == [3e-4]

    def test_state_dict(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer)

        scheduler._step_count = 5

        state = scheduler.state_dict()
        assert state["step_count"] == 5
        assert state["base_lrs"] == [1e-3]

    def test_load_state_dict(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer)

        state = {"step_count": 10, "base_lrs": [2e-3]}

        scheduler.load_state_dict(state)
        assert scheduler._step_count == 10
        assert scheduler.base_lrs == [2e-3]

    def test_last_epoch_property(self):
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer)

        assert scheduler.last_epoch == 0

        scheduler.step()
        assert scheduler.last_epoch == 1

        scheduler.step()
        assert scheduler.last_epoch == 2

    def test_abstract_get_lr_method(self):
        """Test that BaseLRScheduler.get_lr is abstract"""
        optimizer = self.create_optimizer()

        # Should not be able to instantiate BaseLRScheduler directly
        with pytest.raises(TypeError):
            BaseLRScheduler(optimizer)

    def test_scheduler_sequence(self):
        """Test multiple step sequence"""
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer, fixed_lr=1e-4)

        expected_step_counts = [0, 1, 2, 3, 4]
        expected_lrs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

        for i, (expected_step, expected_lr) in enumerate(
            zip(expected_step_counts, expected_lrs, strict=True)
        ):
            assert scheduler._step_count == expected_step
            assert optimizer.param_groups[0]["lr"] == expected_lr

            if i < len(expected_step_counts) - 1:  # Don't step after last iteration
                scheduler.step()

    def test_scheduler_with_different_optimizers(self):
        """Test scheduler works with different optimizer types"""
        model = nn.Linear(10, 5)

        optimizers = [
            torch.optim.SGD(model.parameters(), lr=1e-3),
            torch.optim.Adam(model.parameters(), lr=1e-3),
            torch.optim.AdamW(model.parameters(), lr=1e-3),
        ]

        for optimizer in optimizers:
            scheduler = MockLRScheduler(optimizer, fixed_lr=5e-4)

            initial_lr = optimizer.param_groups[0]["lr"]
            assert initial_lr == 5e-4

            scheduler.step()
            updated_lr = optimizer.param_groups[0]["lr"]
            assert updated_lr == 5e-4

    def test_scheduler_state_persistence(self):
        """Test that scheduler state can be saved and restored"""
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler1 = MockLRScheduler(optimizer, fixed_lr=1e-4)

        # Take some steps
        for _ in range(5):
            scheduler1.step()

        # Save state
        state = scheduler1.state_dict()
        lrs_before = scheduler1.get_lr()

        # Create new scheduler and load state
        optimizer2 = self.create_optimizer(lr=1e-3)
        scheduler2 = MockLRScheduler(optimizer2, fixed_lr=1e-4)
        scheduler2.load_state_dict(state)
        lrs_after = scheduler2.get_lr()

        # Should have same state
        assert scheduler2._step_count == scheduler1._step_count
        assert scheduler2.base_lrs == scheduler1.base_lrs
        assert lrs_after == lrs_before
        for i, j in zip(optimizer.param_groups, optimizer2.param_groups, strict=True):
            assert i["lr"] == j["lr"]

    def test_scheduler_with_zero_lr(self):
        """Test scheduler with zero learning rate"""
        optimizer = self.create_optimizer(lr=0.0)
        scheduler = MockLRScheduler(optimizer, fixed_lr=1e-3)

        assert scheduler.base_lrs == [0.0]

        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_scheduler_with_negative_lr(self):
        """Test scheduler behavior with negative learning rates"""
        # PyTorch optimizers don't allow negative learning rates,
        # so this test should verify that the restriction is enforced
        with pytest.raises(ValueError, match="Invalid learning rate"):
            self.create_optimizer(lr=-1e-3)

    def test_scheduler_large_step_count(self):
        """Test scheduler with large step counts"""
        optimizer = self.create_optimizer(lr=1e-3)
        scheduler = MockLRScheduler(optimizer, fixed_lr=5e-4)

        # Simulate many steps
        for _ in range(1000):
            scheduler.step()

        assert scheduler._step_count == 1000
        assert optimizer.param_groups[0]["lr"] == 5e-4
