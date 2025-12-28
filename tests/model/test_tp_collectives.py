import os

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode

from optimus_dl.modules.model import build_model


def _run_collectives_test(
    rank, unique_port, world_size, model_cfg_dict, loss_parallel, sequence_parallel
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)
        mesh = init_device_mesh("cpu", (world_size,))
        if hasattr(model, "apply_tp"):
            model.apply_tp(
                mesh, loss_parallel=loss_parallel, sequence_parallel=sequence_parallel
            )
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
        all_reduce_count = 0
        all_gather_count = 0
        reduce_scatter_count = 0

        # tracker.comm_counts is a dict {Op: count}
        for op, count in tracker.comm_counts.items():
            op_name = str(op)
            if "all_reduce" in op_name:
                all_reduce_count += count
            elif "all_gather" in op_name:
                all_gather_count += count
            elif "reduce_scatter" in op_name:
                reduce_scatter_count += count

        if not sequence_parallel:
            # Standard TP
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

            expected_reduce_scatter = 0
        else:
            # Sequence Parallelism
            # AllReduce replaced by ReduceScatter
            expected_all_reduce = 0

            # ReduceScatter:
            # 1 (Embed) + 2 * n_layer (Attn Output + MLP Output)
            expected_reduce_scatter = 1 + 2 * n_layer

            # AllGather:
            # Attn Input: 3 (wq, wk, wv separate)
            # MLP Input: 2 (w1, w2 separate)
            # LM Head Input: 1
            # LM Head Output gathering: 1 if not loss_parallel

            expected_all_gather = (3 + 2) * n_layer + 1
            if not loss_parallel:
                expected_all_gather += 1

        if rank == 0:
            print(
                f"Config: loss_parallel={loss_parallel}, sequence_parallel={sequence_parallel}"
            )
            print(
                f"Counts: AR={all_reduce_count}, AG={all_gather_count}, RS={reduce_scatter_count}"
            )
            print(f"Operations found: {tracker.comm_counts.keys()}")

            assert (
                all_reduce_count == expected_all_reduce
            ), f"AllReduce count mismatch. Expected {expected_all_reduce}, got {all_reduce_count}"

            assert (
                all_gather_count == expected_all_gather
            ), f"AllGather count mismatch. Expected {expected_all_gather}, got {all_gather_count}"

            assert (
                reduce_scatter_count == expected_reduce_scatter
            ), f"ReduceScatter count mismatch. Expected {expected_reduce_scatter}, got {reduce_scatter_count}"

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
qwen3_test_cfg = {
    "_name": "qwen3",
    "n_embd": 64,
    "n_head": 4,
    "n_kv_head": 2,
    "n_layer": 2,
    "vocab_size": 256,
    "sequence_length": 128,
    "attention_bias": True,
}
models_cfg = [llama2_cfg, qwen3_test_cfg]


@pytest.mark.parametrize("model_cfg_dict", models_cfg)
class TestTPCollectivesGeneric:
    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["NoSP", "SP"])
    def test_collectives_loss_parallel_false(
        self, unique_port, model_cfg_dict, sequence_parallel
    ):
        """Test collectives with loss_parallel=False (Expect Gather)"""
        world_size = 2
        mp.spawn(
            _run_collectives_test,
            args=(unique_port, world_size, model_cfg_dict, False, sequence_parallel),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["NoSP", "SP"])
    def test_collectives_loss_parallel_true(
        self, unique_port, model_cfg_dict, sequence_parallel
    ):
        """Test collectives with loss_parallel=True (No Gather)"""
        world_size = 2
        mp.spawn(
            _run_collectives_test,
            args=(unique_port, world_size, model_cfg_dict, True, sequence_parallel),
            nprocs=world_size,
            join=True,
        )
