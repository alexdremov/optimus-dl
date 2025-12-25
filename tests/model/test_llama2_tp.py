import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from optimus_dl.modules.model.llama2 import Llama, LlamaConfig


def _run_sharding_test(rank, world_size, config: LlamaConfig):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model = Llama(config)

        mesh = init_device_mesh("cpu", (world_size,))
        model.apply_tp(mesh)

        # Check that parameters are DTensors
        assert isinstance(model.transformer.wte.weight, DTensor)
        assert isinstance(model.transformer.h[0].attn.wq.weight, DTensor)
        assert isinstance(model.lm_head.weight, DTensor)

    finally:
        dist.destroy_process_group()


def _run_test_full_tensor_parallel(rank, world_size, config: LlamaConfig):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model = Llama(config)

        mesh = init_device_mesh("cpu", (world_size,))
        model.apply_tp(mesh)

        input_ids = torch.randint(0, config.vocab_size, (7, 32))

        embedded = model.transformer.wte(input_ids)
        assert embedded.shape == (7, 32, config.n_embd), embedded.shape

        output = model(input_ids)

        assert isinstance(output["logits"], torch.Tensor) and not hasattr(
            output["logits"], "device_mesh"
        ), output["logits"]
        assert output["logits"].shape == (7, 32, config.vocab_size), output[
            "logits"
        ].shape

        model = Llama(config)
        model.apply_tp(mesh, loss_parallel=True)
        output = model(input_ids)

        assert isinstance(output["logits"], DTensor), output["logits"]
        assert output["logits"].shape == (7, 32, config.vocab_size), output[
            "logits"
        ].shape
        assert output["logits"].to_local().shape == (
            7,
            32,
            config.vocab_size // world_size,
        ), (
            output["logits"].to_local().shape
        )

    finally:
        dist.destroy_process_group()


class TestLlamaTP:
    def test_tensor_parallel_sharding(self):
        """
        Test that weights are actually sharded.
        """
        world_size = 2
        config = LlamaConfig(
            n_embd=64, n_head=4, n_layer=1, vocab_size=256, force_disable_flash=True
        )
        mp.spawn(
            _run_sharding_test, args=(world_size, config), nprocs=world_size, join=True
        )

    def test_full_tensor_parallel(self):
        """
        Test that lm_head works correctly with Sequence Parallel input (Shard(1)).
        This verifies the fix for "Sharding propagation failed for Op(op=aten.view.default)".
        """
        world_size = 2
        config = LlamaConfig(
            n_embd=64, n_head=4, n_layer=1, vocab_size=256, force_disable_flash=True
        )

        mp.spawn(
            _run_test_full_tensor_parallel,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
