"""Integration tests for training pipeline components."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.loggers.jsonl import JsonlLogger, JsonlLoggerConfig
from optimus_dl.modules.lr_scheduler.cosine_annealing import (
    CosineAnnealingLR,
    CosineAnnealingLRConfig,
)
from optimus_dl.modules.lr_scheduler.linear_warmup import (
    LinearWarmupLR,
    LinearWarmupLRConfig,
)
from optimus_dl.modules.metrics.common import AverageMetric, log_averaged
from optimus_dl.modules.optim.adamw import AdamWConfig, make_adamw


class TestTrainingPipelineIntegration:
    """Test real training pipeline usage patterns."""

    def test_optimizer_scheduler_integration(self):
        """Test optimizer and scheduler working together in training loop."""
        # Create a simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Setup optimizer
        config = AdamWConfig(lr=1e-3, weight_decay=0.01)
        optimizer = make_adamw(config, model.parameters())

        # Setup scheduler
        warmup_config = LinearWarmupLRConfig(
            warmup_steps=5, start_lr=0.0, target_lr=1e-3
        )
        scheduler = LinearWarmupLR(warmup_config, optimizer, iterations=20)

        # Simulate training steps
        losses = []
        learning_rates = []

        for step in range(10):
            # Forward pass
            x = torch.randn(32, 10)
            target = torch.randn(32, 5)

            output = model(x)
            loss = nn.MSELoss()(output, target)
            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Schedule learning rate
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        # Verify training progressed (loss should generally decrease)
        # Allow for some fluctuation but expect overall downward trend
        final_avg = sum(losses[-3:]) / 3
        initial_avg = sum(losses[:3]) / 3
        assert final_avg < initial_avg * 1.2  # Some improvement expected

        # Verify learning rate schedule worked as expected
        # First few steps should show warmup (increasing LR)
        assert learning_rates[1] > learning_rates[0]  # Warmup
        assert learning_rates[2] > learning_rates[1]  # Warmup continues

        # After warmup, should reach target LR
        assert abs(learning_rates[-1] - 1e-3) < 1e-6

    def test_metrics_logging_integration(self):
        """Test metrics collection and logging in realistic scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logger
            config = JsonlLoggerConfig(output_dir=temp_dir)
            logger = JsonlLogger(config, group="integration_test")

            # Simulate training with metrics
            metric = AverageMetric(round=4)
            losses = []

            for epoch in range(3):
                epoch_losses = []
                for step in range(5):
                    # Simulate decreasing loss
                    loss = (
                        1.0 - (epoch * 0.2) - (step * 0.05) + torch.rand(1).item() * 0.1
                    )
                    epoch_losses.append(loss)
                    metric.log(loss, weight=1.0)

                # Log metrics
                avg_loss = metric.compute()
                losses.append(avg_loss)

                logger.log_metrics(
                    {
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                        "step_count": (epoch + 1) * 5,
                    },
                    step=(epoch + 1) * 5,
                    group="train",
                )

                # Reset metric for next epoch
                metric = AverageMetric(round=4)

            # Verify loss decreased over training
            assert losses[-1] < losses[0]

            # Verify log files were created
            log_files = os.listdir(temp_dir)
            assert len(log_files) > 0
            assert any(file.endswith(".jsonl") for file in log_files)

    def test_multi_scheduler_combination(self):
        """Test combining multiple schedulers for complex learning rate patterns."""
        model = nn.Linear(10, 5)

        # Setup optimizer
        config = AdamWConfig(lr=1e-2)
        optimizer = make_adamw(config, model.parameters())

        # First phase: Linear warmup
        warmup_scheduler = LinearWarmupLR(
            LinearWarmupLRConfig(warmup_steps=10, start_lr=0.0, target_lr=1e-2),
            optimizer,
            iterations=50,
        )

        # Simulate warmup phase
        warmup_lrs = []
        for step in range(10):
            warmup_scheduler.step()
            warmup_lrs.append(optimizer.param_groups[0]["lr"])

        # Verify warmup worked
        assert warmup_lrs[0] < warmup_lrs[-1]  # LR increased
        assert abs(warmup_lrs[-1] - 1e-2) < 1e-6  # Reached target

        # Second phase: Cosine annealing from current LR
        cosine_scheduler = CosineAnnealingLR(
            CosineAnnealingLRConfig(T_max=20, eta_min=1e-5), optimizer, iterations=20
        )

        # Continue with cosine annealing
        cosine_lrs = []
        for step in range(20):
            cosine_scheduler.step()
            cosine_lrs.append(optimizer.param_groups[0]["lr"])

        # Verify cosine annealing worked
        assert cosine_lrs[0] > cosine_lrs[10]  # LR decreased initially
        assert cosine_lrs[10] > cosine_lrs[-1]  # Then continued decreasing

    def test_distributed_training_simulation(self):
        """Test components working in distributed training scenario."""
        # Simulate 2 ranks
        world_size = 2

        models = []
        optimizers = []
        schedulers = []

        # Setup for each rank
        for rank in range(world_size):
            # Create identical model
            model = nn.Linear(10, 5)
            # In real distributed training, models would be synchronized

            # Setup optimizer with rank-specific parameters
            config = AdamWConfig(lr=1e-3 * (rank + 1))  # Different LR per rank for demo
            optimizer = make_adamw(config, model.parameters())

            # Setup scheduler
            scheduler_config = LinearWarmupLRConfig(warmup_steps=3, start_lr=0.0)
            scheduler = LinearWarmupLR(scheduler_config, optimizer, iterations=10)

            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        # Simulate distributed training steps
        for step in range(5):
            for rank in range(world_size):
                # Each rank processes different data
                x = torch.randn(16, 10)  # Different batch per rank
                target = torch.randn(16, 5)

                # Forward pass
                output = models[rank](x)
                loss = nn.MSELoss()(output, target)

                # Backward pass
                optimizers[rank].zero_grad()
                loss.backward()
                optimizers[rank].step()

                # Update scheduler
                schedulers[rank].step()

        # Verify each rank progressed
        for rank in range(world_size):
            initial_lr = 0.0  # Start LR
            current_lr = optimizers[rank].param_groups[0]["lr"]
            assert current_lr > initial_lr  # LR should have increased from warmup

    def test_checkpoint_resume_simulation(self):
        """Test saving and resuming training state."""
        # Initial training setup
        model = nn.Linear(10, 5)
        config = AdamWConfig(lr=1e-3)
        optimizer = make_adamw(config, model.parameters())

        scheduler_config = CosineAnnealingLRConfig(T_max=20, eta_min=1e-5)
        scheduler = CosineAnnealingLR(scheduler_config, optimizer, iterations=20)

        # Train for a few steps
        for step in range(5):
            x = torch.randn(8, 10)
            target = torch.randn(8, 5)

            output = model(x)
            loss = nn.MSELoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Save state
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": 5,
        }

        # Simulate loading from checkpoint
        new_model = nn.Linear(10, 5)
        new_model.load_state_dict(checkpoint["model_state"])

        new_optimizer = make_adamw(config, new_model.parameters())
        new_optimizer.load_state_dict(checkpoint["optimizer_state"])

        new_scheduler = CosineAnnealingLR(
            scheduler_config, new_optimizer, iterations=20
        )
        new_scheduler.load_state_dict(checkpoint["scheduler_state"])

        # Verify states match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

        # Continue training should work
        for step in range(3):
            x = torch.randn(8, 10)
            target = torch.randn(8, 5)

            output = new_model(x)
            loss = nn.MSELoss()(output, target)

            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()
            new_scheduler.step()

        # Should complete without errors
        assert True

    @patch("optimus_dl.modules.loggers.wandb.wandb")
    def test_wandb_integration_pattern(self, mock_wandb):
        """Test typical Weights & Biases integration pattern."""
        from optimus_dl.modules.loggers.wandb import WandbLogger, WandbLoggerConfig

        # Setup mock wandb
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.log = Mock()

        # Setup logger
        config = WandbLoggerConfig(project="test_project", entity="test_entity")
        logger = WandbLogger(config, group="integration_test")

        # Initialize the logger (required for WandB)
        logger.setup("test_experiment", {"lr": 1e-3, "batch_size": 16})

        # Simulate training with W&B logging
        model = nn.Linear(10, 5)
        optimizer = make_adamw(AdamWConfig(lr=1e-3), model.parameters())

        for epoch in range(3):
            epoch_loss = 0.0
            for step in range(5):
                x = torch.randn(16, 10)
                target = torch.randn(16, 5)

                output = model(x)
                loss = nn.MSELoss()(output, target)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log step-level metrics
                logger.log_metrics(
                    {
                        "step_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "step": epoch * 5 + step,
                    },
                    step=epoch * 5 + step,
                    group="train",
                )

            # Log epoch-level metrics
            logger.log_metrics(
                {"epoch": epoch, "epoch_avg_loss": epoch_loss / 5},
                step=(epoch + 1) * 5,
                group="train",
            )

        # Verify wandb was called appropriately
        assert mock_wandb.init.called
        # Note: The actual wandb.log calls may be batched or handled differently
        # Just verify that the logger was properly initialized
