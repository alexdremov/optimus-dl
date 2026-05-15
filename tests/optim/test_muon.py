import torch
import pytest
import torch.nn as nn

from optimus_dl.modules.optim.adamw import AdamWConfig
from optimus_dl.modules.optim.composite import CompositeOptimizer
from optimus_dl.modules.optim.muon import (
    MuonConfig,
    make_muon,
)


class TestMuonOptimizer:
    def test_muon_matrix_split(self):
        # Create a model with matrix (2D) and non-matrix (1D) parameters
        model = nn.Sequential(
            nn.Linear(10, 5),  # weight is 2D, bias is 1D
            nn.LayerNorm(5),  # weight is 1D, bias is 1D
            nn.Embedding(100, 10),  # weight is 2D
        )

        # Support optimizer configuration
        config = MuonConfig(
            _name="muon", lr=1e-3, support_optimizer=AdamWConfig(_name="adamw", lr=1e-2)
        )

        optimizer = make_muon(config, params=model.named_parameters())

        # It should return a CompositeOptimizer because we have both matrix and non-matrix params
        assert isinstance(optimizer, CompositeOptimizer)
        assert "muon" in optimizer.optimizers
        assert "support" in optimizer.optimizers

        muon_opt = optimizer.optimizers["muon"]
        support_opt = optimizer.optimizers["support"]

        # Check that Muon optimizer only has 2D params
        muon_params = []
        for group in muon_opt.param_groups:
            muon_params.extend(group["params"])
        assert len(muon_params) == 2  # Linear weight and Embedding weight
        assert all(p.ndim == 2 for p in muon_params)

        # Check that Support optimizer only has non-2D params
        support_params = []
        for group in support_opt.param_groups:
            support_params.extend(group["params"])
        assert len(support_params) == 3  # Linear bias, LayerNorm weight, LayerNorm bias
        assert all(p.ndim != 2 for p in support_params)

        # Check learning rates were assigned correctly
        assert muon_opt.param_groups[0]["lr"] == 1e-3
        assert support_opt.param_groups[0]["lr"] == 1e-2

    def test_muon_only_matrix(self):
        # A model with only 2D parameters
        class OnlyMatrixModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))

        model = OnlyMatrixModel()

        config = MuonConfig(_name="muon", lr=1e-3)
        optimizer = make_muon(config, params=model.named_parameters())

        # Should return a raw Muon optimizer, not CompositeOptimizer, because there are no 1D params
        assert isinstance(optimizer, torch.optim.Muon)

        muon_params = []
        for group in optimizer.param_groups:
            muon_params.extend(group["params"])
        assert len(muon_params) == 1
        assert muon_params[0].ndim == 2

    def test_muon_only_non_matrix(self):
        # A model with only 1D parameters
        class OnlyNonMatrixModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10))

        model = OnlyNonMatrixModel()

        config = MuonConfig(
            _name="muon", lr=1e-3, support_optimizer=AdamWConfig(_name="adamw", lr=1e-2)
        )
        with pytest.raises(ValueError) as exc_info:
            make_muon(config, params=model.named_parameters())
        assert "Muon optimizer requires at least one 2D parameter" in str(
            exc_info.value
        )
