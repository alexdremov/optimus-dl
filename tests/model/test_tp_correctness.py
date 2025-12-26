import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.distributed.mesh import MeshCollective
from optimus_dl.modules.model import build_model


def _run_tp_correctness_test(
    rank, unique_port, world_size, model_cfg_dict, loss_parallel, sequence_parallel
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)

    if dist.is_initialized():
        dist.destroy_process_group()

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        # Common setup
        torch.manual_seed(42)
        batch_size = 2

        model_cfg = OmegaConf.create(model_cfg_dict)
        # Assuming config has sequence_length and vocab_size for data generation
        seq_len = getattr(model_cfg, "sequence_length", 32)
        vocab_size = getattr(model_cfg, "vocab_size", 512)

        # Create input: (B, T+1) because criterion slices it
        full_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1))

        # --- 1. Baseline (No TP) ---
        # All ranks run this locally to have the ground truth
        torch.manual_seed(42)  # Reset seed for identical initialization
        baseline_model = build_model(model_cfg)

        # Prepare inputs for manual forward/loss
        input_ids = full_input_ids[:, :-1]
        targets = full_input_ids[:, 1:]

        baseline_out = baseline_model(input_ids)
        baseline_logits = baseline_out["logits"]

        # Standard Cross Entropy
        baseline_loss = F.cross_entropy(
            baseline_logits.view(-1, vocab_size), targets.reshape(-1)
        )
        baseline_loss.backward()

        baseline_grads = {
            n: p.grad.clone()
            for n, p in baseline_model.named_parameters()
            if p.grad is not None
        }

        # --- 2. TP Model ---
        torch.manual_seed(42)  # Reset seed for identical initialization
        tp_model = build_model(model_cfg)

        # Setup Mesh
        collective = MeshCollective(
            rank=rank,
            world_size=world_size,
            local_world_size=world_size,
            local_rank=rank,
            device_type="cpu",
            tp_size=world_size,
        )

        # Apply TP
        if hasattr(tp_model, "apply_tp"):
            tp_model.apply_tp(
                collective.tp_mesh,
                loss_parallel=loss_parallel,
                sequence_parallel=sequence_parallel,
            )
        else:
            if rank == 0:
                print("Skipping TP application: model does not support apply_tp")
            return

        # Criterion
        criterion = CrossEntropyCriterion(CrossEntropyCriterionConfig(), collective)

        # Run Forward + Loss via Criterion
        # Pass a copy of batch to avoid mutation side-effects affecting other checks if any
        batch = {"input_ids": full_input_ids.clone()}

        # Verify the model output type before criterion
        # We manually peek to ensure our fix works as expected
        if loss_parallel:
            # Should be sharded
            with torch.no_grad():
                out_check = tp_model(input_ids)
                logits_check = out_check["logits"]
                assert isinstance(
                    logits_check, DTensor
                ), "Expected DTensor with loss_parallel=True"
        else:
            # Should be replicated DTensor (our fix ensured this)
            with torch.no_grad():
                out_check = tp_model(input_ids)
                logits_check = out_check["logits"]
                assert isinstance(
                    logits_check, DTensor
                ), "Expected DTensor with loss_parallel=False"

        # Compute loss
        tp_loss = criterion(tp_model, batch)

        # Backward
        tp_loss.backward()

        # --- 3. Verification ---

        # A. Loss Correctness
        # tp_loss is a scalar DTensor. Convert to local for comparison.
        if isinstance(tp_loss, DTensor):
            tp_loss_local = tp_loss.to_local()
        else:
            tp_loss_local = tp_loss

        assert torch.allclose(
            tp_loss_local, baseline_loss, atol=1e-5
        ), f"Rank {rank}: Loss mismatch. TP={tp_loss_local.item()}, Base={baseline_loss.item()}"

        # B. Gradient Correctness
        for name, tp_param in tp_model.named_parameters():
            if tp_param.grad is None:
                continue

            assert name in baseline_grads, f"Gradient for {name} missing in baseline"
            base_grad = baseline_grads[name]
            tp_grad = tp_param.grad

            if isinstance(tp_grad, DTensor):
                # Reconstruct full gradient from shards
                full_tp_grad = tp_grad.full_tensor()

                # Check shape
                assert (
                    full_tp_grad.shape == base_grad.shape
                ), f"Shape mismatch for {name}: {full_tp_grad.shape} vs {base_grad.shape}"

                # Check values
                # Relax tolerance slightly for float accumulation differences
                assert torch.allclose(
                    full_tp_grad, base_grad, atol=1e-5, rtol=1e-4
                ), f"Rank {rank}: Grad mismatch for {name}. \nMax Diff: {(full_tp_grad - base_grad).abs().max()}"
            else:
                # Should be identical
                assert torch.allclose(
                    tp_grad, base_grad, atol=1e-5, rtol=1e-4
                ), f"Rank {rank}: Grad mismatch for {name} (Tensor)."

    finally:
        dist.destroy_process_group()


# Configs for parametrization
llama2_cfg = {
    "_name": "llama2",
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 2,
    "vocab_size": 512,
    "sequence_length": 32,
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
class TestTPCorrectnessGeneric:
    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["NoSP", "SP"])
    @pytest.mark.parametrize("loss_parallel", [False, True], ids=["NoLP", "LP"])
    def test_tp_correctness_loss_parallel_false(
        self, unique_port, model_cfg_dict, sequence_parallel, loss_parallel
    ):
        """Test TP=2 with loss_parallel=False, with/without SP"""
        world_size = 2
        mp.spawn(
            _run_tp_correctness_test,
            args=(
                unique_port,
                world_size,
                model_cfg_dict,
                loss_parallel,
                sequence_parallel,
            ),
            nprocs=world_size,
            join=True,
        )
