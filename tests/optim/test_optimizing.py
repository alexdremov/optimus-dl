import pytest
import torch
import torch.nn as nn

from optimus_dl.modules.optim import build_optimizer

optimizers = [
    {
        "_name": "adamw",
    },
    {
        "_name": "soap",
    },
    {
        "_name": "muon",
    },
]


@pytest.mark.parametrize("optimizer_config", optimizers, ids=lambda x: x["_name"])
def test_default_optimizers_decrease_loss(optimizer_config):
    """Test that the created optimizer can perform optimization steps"""
    # Create a simple optimization problem
    torch.manual_seed(42)
    x = torch.randn(10, 5)
    target = torch.randn(10, 1)

    is_muon = optimizer_config["_name"] == "muon"
    model = nn.Linear(5, 1, bias=not is_muon)
    criterion = nn.MSELoss()

    optimizer = build_optimizer(optimizer_config, params=model.parameters())

    # Perform a few optimization steps
    initial_loss = None
    for _ in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    # Loss should decrease
    final_loss = loss.item()
    print(
        f"{optimizer_config['_name']}: final_loss",
        final_loss,
        "initial_loss",
        initial_loss,
    )
    assert final_loss < initial_loss
