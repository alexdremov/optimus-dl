import os

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from optimus_dl.modules.model import build_model


def _run_sharding_test(rank, unique_port, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
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


def _run_test_full_tensor_parallel(
    rank, unique_port, world_size, model_cfg_dict, sequence_parallel
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
            model.apply_tp(mesh, sequence_parallel=sequence_parallel)

        vocab_size = model.config.vocab_size
        n_embd = model.config.n_embd

        input_ids = torch.randint(0, vocab_size, (7, 32))

        # Check embedding output shape (specific to models with 'transformer.wte')
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embedded = model.transformer.wte(input_ids)
            # If SP, embedding output is sharded on seq dim (dim 1)
            # Local shape: (B, S/TP, H)
            if sequence_parallel:
                if not isinstance(embedded, DTensor):
                    assert embedded.shape == (
                        7,
                        32 // world_size,
                        n_embd,
                    ), f"SP Embedding shape mismatch: {embedded.shape}"
                else:
                    assert embedded.shape == (
                        7,
                        32,
                        n_embd,
                    ), f"SP Embedding shape mismatch: {embedded.shape}"
            else:
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
        model.apply_tp(mesh, loss_parallel=True, sequence_parallel=sequence_parallel)
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


def _run_tp_assertion_test(rank, unique_port, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
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
olmo3_test_cfg = {
    "_name": "olmo3",
    "n_embd": 64,
    "n_head": 4,
    "n_kv_head": 2,
    "n_layer": 2,
    "layer_types": ["sliding_attention", "full_attention"],
    "vocab_size": 256,
    "sequence_length": 128,
    "intermediate_size": 128,
}
models_cfg = [llama2_cfg, qwen3_test_cfg, olmo3_test_cfg]


def _run_test_sp_internals(rank, unique_port, world_size, model_cfg_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)
        model = build_model(model_cfg)

        mesh = init_device_mesh("cpu", (world_size,))
        # Apply TP with Sequence Parallelism
        model.apply_tp(mesh, sequence_parallel=True)

        # Check Attention Weights
        # WQ: Colwise -> Shard(0) (Heads)
        assert (
            model.transformer.h[0].attn.wq.weight.placements[0].is_shard(0)
        ), f"WQ should be Shard(0), got {model.transformer.h[0].attn.wq.weight.placements}"

        # WO: Rowwise -> Shard(1) (Heads input)
        assert (
            model.transformer.h[0].attn.wo.weight.placements[0].is_shard(1)
        ), f"WO should be Shard(1), got {model.transformer.h[0].attn.wo.weight.placements}"

        # 2. Verify Data Flow and Intermediate Sharding
        vocab_size = model.config.vocab_size
        n_embd = model.config.n_embd

        input_ids = torch.randint(0, vocab_size, (2, 32))
        # WTE
        tok_emb = model.transformer.wte(input_ids)
        if not isinstance(tok_emb, DTensor):
            # Check output is Shard(1) (Sequence dimension)
            assert tok_emb.shape == (
                2,
                16,
                n_embd,
            ), f"WTE output shape mismatch: {tok_emb.shape}"  # Global shape

        x = tok_emb
        # Handle different model attributes for precomputed freqs
        if hasattr(model, "freqs_cis"):
            model.freqs_cis[: x.shape[1]].to(x.device)
        elif hasattr(model, "freqs_cis_sliding"):
            model.freqs_cis_sliding[: x.shape[1]].to(x.device)

        # Block 0
        block = model.transformer.h[0]

        # LN1 (SequenceParallel)
        # Should accept Shard(1) input and produce Shard(1) output
        ln1_out = block.ln_1(x)
        assert isinstance(ln1_out, DTensor)
        assert ln1_out.placements[0].is_shard(
            1
        ), f"LN1 output should be Shard(1), got {ln1_out.placements}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_cfg_dict", models_cfg)
class TestTPGeneric:
    def test_tensor_parallel_sharding(self, unique_port, model_cfg_dict):
        world_size = 2
        mp.spawn(
            _run_sharding_test,
            args=(unique_port, world_size, model_cfg_dict),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["NoSP", "SP"])
    def test_full_tensor_parallel(self, unique_port, model_cfg_dict, sequence_parallel):
        world_size = 2
        mp.spawn(
            _run_test_full_tensor_parallel,
            args=(unique_port, world_size, model_cfg_dict, sequence_parallel),
            nprocs=world_size,
            join=True,
        )

    def test_sp_internals(self, unique_port, model_cfg_dict):
        world_size = 2
        mp.spawn(
            _run_test_sp_internals,
            args=(unique_port, world_size, model_cfg_dict),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize("invalid_model_cfg_dict", [llama2_invalid_cfg])
    def test_tensor_parallel_assertion(
        self, unique_port, model_cfg_dict, invalid_model_cfg_dict
    ):  # model_cfg_dict arg ignored here
        world_size = 2
        mp.spawn(
            _run_tp_assertion_test,
            args=(unique_port, world_size, invalid_model_cfg_dict),
            nprocs=world_size,
            join=True,
        )
