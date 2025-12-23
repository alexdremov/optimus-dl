import importlib.util
import unittest

import pytest
import torch

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)

LIGER_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not LIGER_AVAILABLE, reason="Liger Kernel not installed")
class TestLigerCrossEntropyEquivalence(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")

        # Standard PyTorch implementation
        self.base_cfg = CrossEntropyCriterionConfig(
            _name="cross_entropy", label_smoothing=0.1, use_liger_kernel=False
        )
        self.base_criterion = CrossEntropyCriterion(self.base_cfg)

        # Liger Kernel implementation
        self.liger_cfg = CrossEntropyCriterionConfig(
            _name="cross_entropy", label_smoothing=0.1, use_liger_kernel=True
        )
        self.liger_criterion = CrossEntropyCriterion(self.liger_cfg)

    def test_loss_equivalence(self):
        """Test that Liger CrossEntropy produces same loss as PyTorch implementation."""
        batch_size = 4
        seq_len = 32
        vocab_size = 1000

        # Create dummy inputs
        # Logits: (B, T, V)
        logits = torch.randn(
            batch_size, seq_len, vocab_size, device=self.device, requires_grad=True
        )
        # Targets: (B, T+1) because criterion slices internally
        # Criterion expects input_ids[:, 1:] as targets
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len + 1), device=self.device
        )

        # We need to clone logits for the second pass because backward() might modify gradients
        logits_base = logits.clone().detach().requires_grad_(True)
        logits_liger = logits.clone().detach().requires_grad_(True)

        # Mock model output dict
        class MockModel:
            def __call__(self, **kwargs):
                return {
                    "logits": logits_base if kwargs.get("is_base") else logits_liger
                }

        model = MockModel()

        # 1. Base (PyTorch) Forward
        batch_base = {"input_ids": input_ids.clone(), "is_base": True}
        loss_base = self.base_criterion(model, batch_base)
        loss_base.backward()

        # 2. Liger Forward
        batch_liger = {"input_ids": input_ids.clone(), "is_base": False}
        loss_liger = self.liger_criterion(model, batch_liger)
        loss_liger.backward()

        # Check Loss Equivalence
        torch.testing.assert_close(loss_base, loss_liger, rtol=1e-5, atol=1e-5)

        # Check Gradient Equivalence
        torch.testing.assert_close(
            logits_base.grad, logits_liger.grad, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
