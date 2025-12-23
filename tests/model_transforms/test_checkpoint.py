import unittest

import torch
import torch.nn as nn

from optimus_dl.modules.model_transforms.checkpoint import (
    ActivationCheckpointConfig,
    ActivationCheckpointTransform,
    CheckpointWrapper,
)


class SimpleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = SimpleBlock()
        self.block2 = SimpleBlock()
        self.head = nn.Linear(10, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class TestActivationCheckpoint(unittest.TestCase):
    def test_apply_checkpoint(self):
        model = SimpleModel()
        cfg = ActivationCheckpointConfig(layer_classes=["SimpleBlock"])
        transform = ActivationCheckpointTransform(cfg)

        # Apply transform
        model = transform.apply(model)

        # Verify wrapping
        self.assertIsInstance(model.block1, CheckpointWrapper)
        self.assertIsInstance(model.block2, CheckpointWrapper)
        self.assertNotIsInstance(model.head, CheckpointWrapper)

        # Verify forward/backward
        x = torch.randn(2, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(model.block1.module.linear.weight.grad)
        self.assertIsNotNone(model.block2.module.linear.weight.grad)

    def test_reentrant_config(self):
        model = SimpleModel()
        cfg = ActivationCheckpointConfig(
            layer_classes=["SimpleBlock"], use_reentrant=True
        )
        transform = ActivationCheckpointTransform(cfg)
        model = transform.apply(model)

        self.assertTrue(model.block1.use_reentrant)


if __name__ == "__main__":
    unittest.main()
