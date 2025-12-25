import os
import copy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from optimus_dl.modules.model.llama2 import Llama, LlamaConfig

def _disable_flash(model, config):
    for block in model.transformer.h:
        block.attn.flash = False
        if not hasattr(block.attn, 'bias'):
            # block_size is in config (inherited from GPTConfig)
            size = getattr(config, "block_size", 1024)
            # Ensure it covers sequence_length used in test
            if hasattr(config, "sequence_length"):
                size = max(size, config.sequence_length)
            
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones(size, size)).view(
                    1, 1, size, size
                ),
            )

def _run_tp_test(rank, world_size, config, input_ids, expected_logits):
    # Setup process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "gloo" # Use gloo for CPU testing
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    try:
        # Create model with same seed/config
        torch.manual_seed(42)
        model = Llama(config)
        _disable_flash(model, config)
        model.eval()
        
        # Create device mesh
        mesh = init_device_mesh("cpu", (world_size,))
        
        # Apply Tensor Parallelism
        model.apply_tp(mesh)
        
        # Run forward
        with torch.no_grad():
            output = model(input_ids)
            logits = output["logits"]
            
        # Verify output matches expected (baseline)
        # We check on all ranks
        assert torch.allclose(logits, expected_logits, atol=1e-5), \
            f"Rank {rank}: Logits mismatch. Max diff: {(logits - expected_logits).abs().max()}"
            
    finally:
        dist.destroy_process_group()

def _run_sharding_test(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501" # Different port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    try:
        torch.manual_seed(42)
        model = Llama(config)
        
        # Check original shapes
        orig_wte_shape = model.transformer.wte.weight.shape
        orig_wq_shape = model.transformer.h[0].attn.wq.weight.shape
        
        mesh = init_device_mesh("cpu", (world_size,))
        model.apply_tp(mesh)
        
        # Check that parameters are DTensors
        from torch.distributed.tensor import DTensor
        
        assert isinstance(model.transformer.wte.weight, DTensor)
        assert isinstance(model.transformer.h[0].attn.wq.weight, DTensor)
        
    finally:
        dist.destroy_process_group()

class TestLlamaTP:
    def test_tensor_parallel_correctness(self):
        """
        Test that Tensor Parallelism produces the same results as a single model.
        Running on CPU with Gloo backend.
        """
        world_size = 2
        
        # Configuration for a tiny model
        config = LlamaConfig(
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=64,
            intermediate_size=128,
            sequence_length=32,
            dropout=0.0
        )
        
        # Prepare inputs and baseline
        torch.manual_seed(42)
        baseline_model = Llama(config)
        _disable_flash(baseline_model, config)
        baseline_model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        
        with torch.no_grad():
            expected_output = baseline_model(input_ids)
            expected_logits = expected_output["logits"]
            
        # Spawn processes
        mp.spawn(
            _run_tp_test,
            args=(world_size, config, input_ids, expected_logits),
            nprocs=world_size,
            join=True
        )

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