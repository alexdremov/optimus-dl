import os
import copy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor

from optimus_dl.modules.model.llama2 import Llama, LlamaConfig

def _run_lm_head_test(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    try:
        torch.manual_seed(42)
        model = Llama(config)
        model.eval()
        
        mesh = init_device_mesh("cpu", (world_size,))
        model.apply_tp(mesh)
        
        # Verify lm_head weight is sharded (Rowwise logic -> shards dim 1 of weight)
        # Weight shape (V, H). Dim 1 is H.
        assert isinstance(model.lm_head.weight, DTensor)
        # Check placements: should be Shard(1) because it shares with wte which is ColwiseParallel(Embedding) -> Shard(1)
        assert model.lm_head.weight.placements[0].is_shard(dim=1)
        
        # Create input corresponding to Sequence Parallel output: (B, T, H) sharded on T (dim 1)
        B, T, H = 2, 16, config.n_embd
        # Global input
        x_global = torch.randn(B, T, H)
        
        # Distribute: Shard on dim 1 (Sequence)
        x_dt = distribute_tensor(x_global, mesh, [Shard(1)])
        
        # Run lm_head
        # This was failing with "Sharding propagation failed for Op(op=aten.view.default..." 
        # when input was S(1) and lm_head wasn't correctly configured to handle it (or handle redistribution).
        logits = model.lm_head(x_dt)
        
        print(f"DEBUG: logits type: {type(logits)}")
        # Check output
        # With RowwiseParallel(input_layouts=Shard(1)), it should redistribute x to Shard(2) (Hidden),
        # perform matmul, and AllReduce. Output should be Replicated.
        assert isinstance(logits, DTensor)
        assert logits.placements[0].is_replicate()
        assert logits.shape == (B, T, config.vocab_size)
        
    finally:
        dist.destroy_process_group()

def _run_sharding_test(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    try:
        torch.manual_seed(42)
        model = Llama(config)
        
        orig_wte_shape = model.transformer.wte.weight.shape
        
        mesh = init_device_mesh("cpu", (world_size,))
        model.apply_tp(mesh)
        
        # Check that parameters are DTensors
        assert isinstance(model.transformer.wte.weight, DTensor)
        assert isinstance(model.transformer.h[0].attn.wq.weight, DTensor)
        assert isinstance(model.lm_head.weight, DTensor)
        
    finally:
        dist.destroy_process_group()

class TestLlamaTP:
    def test_tensor_parallel_sharding(self):
        """
        Test that weights are actually sharded.
        """
        world_size = 2
        config = LlamaConfig(n_embd=64, n_head=4, n_layer=1, vocab_size=256)
        mp.spawn(
            _run_sharding_test,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )

    def test_lm_head_tensor_parallel(self):
        """
        Test that lm_head works correctly with Sequence Parallel input (Shard(1)).
        This verifies the fix for "Sharding propagation failed for Op(op=aten.view.default)".
        """
        world_size = 2
        config = LlamaConfig(n_embd=64, n_head=4, n_layer=1, vocab_size=256)
        mp.spawn(
            _run_lm_head_test,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
