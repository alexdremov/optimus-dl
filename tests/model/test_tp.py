import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from optimus_dl.modules.model import build_model


def _run_sharding_test(rank, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)

        mesh = init_device_mesh("cpu", (world_size,))
        if hasattr(model, "apply_tp"):
            model.apply_tp(mesh)
        else:
            pytest.skip("Model does not support apply_tp")

        # Check that parameters are DTensors
        # Generic check: verify at least one parameter is sharded if apply_tp was called
        has_dtensor = False
        for _name, param in model.named_parameters():
            if isinstance(param, DTensor):
                has_dtensor = True
                break

        assert has_dtensor, "No parameters were sharded after apply_tp"

        # Specific checks for Llama2 if applicable
        if model_cfg._name == "llama2":
            assert isinstance(model.transformer.wte.weight, DTensor)
            assert isinstance(model.transformer.h[0].attn.wq.weight, DTensor)
            assert isinstance(model.lm_head.weight, DTensor)

    finally:
        dist.destroy_process_group()


def _run_test_full_tensor_parallel(rank, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)

        mesh = init_device_mesh("cpu", (world_size,))
        if hasattr(model, "apply_tp"):
            model.apply_tp(mesh)

        vocab_size = model.config.vocab_size
        n_embd = model.config.n_embd

        input_ids = torch.randint(0, vocab_size, (7, 32))

        # Check embedding output shape (specific to models with 'transformer.wte')
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embedded = model.transformer.wte(input_ids)
            assert embedded.shape == (7, 32, n_embd), embedded.shape

        output = model(input_ids)

        # Updated behavior: lm_head always returns DTensor (Replicated if loss_parallel=False)
        assert isinstance(
            output["logits"], DTensor
        ), "Expected DTensor even with loss_parallel=False"
        assert output["logits"].shape == (7, 32, vocab_size), output["logits"].shape
        # Verify it's replicated (full size locally)
        assert output["logits"].to_local().shape == (7, 32, vocab_size)

        # Test loss_parallel=True
        model = build_model(model_cfg)
        model.apply_tp(mesh, loss_parallel=True)
        output = model(input_ids)

        assert isinstance(output["logits"], DTensor), output["logits"]
        assert output["logits"].shape == (7, 32, vocab_size), output["logits"].shape
        assert output["logits"].to_local().shape == (
            7,
            32,
            vocab_size // world_size,
        ), (
            output["logits"].to_local().shape
        )

    finally:
        dist.destroy_process_group()


def _run_tp_assertion_test(rank, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)
        mesh = init_device_mesh("cpu", (world_size,))
        try:
            model.apply_tp(mesh)
        except AssertionError:
            # Expected
            pass
        else:
            raise RuntimeError("AssertionError not raised for invalid config")
    finally:
        dist.destroy_process_group()


# Configs for parametrization
llama2_cfg = {
    "_name": "llama2",
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 1,
    "vocab_size": 256,
}

llama2_invalid_cfg = {
    "_name": "llama2",
    "n_embd": 48,
    "n_head": 3,  # Not divisible by tp_size=2
    "n_layer": 1,
    "vocab_size": 256,
}


@pytest.mark.parametrize("model_cfg_dict", [llama2_cfg])
class TestTPGeneric:
    def test_tensor_parallel_sharding(self, model_cfg_dict):
        world_size = 2
        mp.spawn(
            _run_sharding_test,
            args=(world_size, model_cfg_dict),
            nprocs=world_size,
            join=True,
        )

    def test_full_tensor_parallel(self, model_cfg_dict):
        world_size = 2
        mp.spawn(
            _run_test_full_tensor_parallel,
            args=(world_size, model_cfg_dict),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize("invalid_model_cfg_dict", [llama2_invalid_cfg])
    def test_tensor_parallel_assertion(
        self, model_cfg_dict, invalid_model_cfg_dict
    ):  # model_cfg_dict arg ignored here
        world_size = 2
        mp.spawn(
            _run_tp_assertion_test,
            args=(world_size, invalid_model_cfg_dict),
            nprocs=world_size,
            join=True,
        )
