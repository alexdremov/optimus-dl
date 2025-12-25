import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode

from optimus_dl.modules.model import build_model


def _run_collectives_test(rank, world_size, model_cfg_dict, loss_parallel):
    os.environ["MASTER_ADDR"] = "localhost"
    port = 29800 + (1 if loss_parallel else 0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)
        mesh = init_device_mesh("cpu", (world_size,))
        if hasattr(model, "apply_tp"):
            model.apply_tp(mesh, loss_parallel=loss_parallel)
        else:
            if rank == 0:
                print("Skipping collectives check: model does not support apply_tp")
            return

        vocab_size = getattr(model_cfg, "vocab_size", 512)
        seq_len = getattr(model_cfg, "sequence_length", 16)
        n_layer = getattr(model_cfg, "n_layer", 2)

        input_ids = torch.randint(0, vocab_size, (2, seq_len))

        # Warmup (optional, but good practice)
        with torch.no_grad():
            model(input_ids)

        dist.barrier()

        # Capture Collectives
        with CommDebugMode() as tracker:
            with torch.no_grad():
                model(input_ids)

        # Analysis
        # Expected AllReduces:
        # 1 for Embeddings (RowwiseParallel, input replicated -> output allreduce)
        # 2 per Layer:
        #   - Attention Output (RowwiseParallel)
        #   - MLP Output (RowwiseParallel)
        expected_all_reduce = 1 + 2 * n_layer

        # Expected AllGathers:
        # 1 for LM Head if loss_parallel=False (ColwiseParallel output -> Replicated)
        # 0 if loss_parallel=True
        expected_all_gather = 1 if not loss_parallel else 0

        # CommDebugMode tracker keys are usually OpOverload objects or strings depending on version.
        # We aggregate counts.

        all_reduce_count = 0
        all_gather_count = 0

        # tracker.comm_counts is a dict {Op: count}
        for op, count in tracker.comm_counts.items():
            op_name = str(op)
            if "all_reduce" in op_name:
                all_reduce_count += count
            elif "all_gather" in op_name:
                all_gather_count += count

        if rank == 0:
            print(f"Config: loss_parallel={loss_parallel}")
            print(f"Counts: AllReduce={all_reduce_count}, AllGather={all_gather_count}")
            print(f"Operations found: {tracker.comm_counts.keys()}")

            assert (
                all_reduce_count == expected_all_reduce
            ), f"AllReduce count mismatch. Expected {expected_all_reduce}, got {all_reduce_count}"

            assert (
                all_gather_count == expected_all_gather
            ), f"AllGather count mismatch. Expected {expected_all_gather}, got {all_gather_count}"

    finally:
        dist.destroy_process_group()


# Configs for parametrization
llama2_cfg = {
    "_name": "llama2",
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 2,
    "vocab_size": 512,
    "sequence_length": 16,
}


@pytest.mark.parametrize("model_cfg_dict", [llama2_cfg])
class TestTPCollectivesGeneric:
    def test_collectives_loss_parallel_false(self, model_cfg_dict):
        """Test collectives with loss_parallel=False (Expect Gather)"""
        world_size = 2
        mp.spawn(
            _run_collectives_test,
            args=(world_size, model_cfg_dict, False),
            nprocs=world_size,
            join=True,
        )

    def test_collectives_loss_parallel_true(self, model_cfg_dict):
        """Test collectives with loss_parallel=True (No Gather)"""
        world_size = 2
        mp.spawn(
            _run_collectives_test,
            args=(world_size, model_cfg_dict, True),
            nprocs=world_size,
            join=True,
        )
