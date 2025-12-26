import gc
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from optimus_dl.modules.model import build_model
from optimus_dl.modules.model.blocks.layer_norms import RMSNorm

# Config for memory test - needs to be large enough to show difference
# scaling up dimensions slightly to make memory usage measurable
llama2_mem_cfg = {
    "_name": "llama2",
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 4,
    "vocab_size": 1024,
    "sequence_length": 512,  # Increased sequence length
}


def _run_memory_test(rank, world_size, model_cfg_dict, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(42)
        model_cfg = OmegaConf.create(model_cfg_dict)

        target_classes = (nn.Linear, nn.Embedding, RMSNorm)

        def run_pass(sp_flag):
            model = build_model(model_cfg)
            mesh = init_device_mesh("cpu", (world_size,))
            model.apply_tp(mesh, sequence_parallel=sp_flag)

            input_ids = torch.randint(
                0, model_cfg.vocab_size, (1, model_cfg.sequence_length)
            )

            total_size = 0

            def hook(module, input, output):
                nonlocal total_size
                # Handle tuple outputs if any (though Linear/Norm usually return single tensor)
                if isinstance(output, tuple):
                    tensors = output
                else:
                    tensors = (output,)

                for t in tensors:
                    if isinstance(t, torch.distributed.tensor.DTensor):
                        # local size
                        size = t.to_local().numel() * t.to_local().element_size()
                    elif isinstance(t, torch.Tensor):
                        size = t.numel() * t.element_size()
                    else:
                        size = 0
                    total_size += size

            handles = []
            for _name, module in model.named_modules():
                if isinstance(module, target_classes):
                    handles.append(module.register_forward_hook(hook))

            with torch.no_grad():
                model(input_ids)

            for h in handles:
                h.remove()

            del model
            gc.collect()
            return total_size

        # --- Run with SP=False ---
        nosp_total = run_pass(sp_flag=False)

        # --- Run with SP=True ---
        sp_total = run_pass(sp_flag=True)

        if rank == 0:
            result_queue.put({"nosp": nosp_total, "sp": sp_total})

    finally:
        dist.destroy_process_group()


class TestTPMemory:
    def test_sp_memory_usage(self):
        world_size = 2
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        mp.spawn(
            _run_memory_test,
            args=(world_size, llama2_mem_cfg, queue),
            nprocs=world_size,
            join=True,
        )

        if not queue.empty():
            sizes = queue.get()
            nosp_total = sizes["nosp"]
            sp_total = sizes["sp"]

            print(
                f"Total activation sizes (Bytes) - NoSP: {nosp_total}, SP: {sp_total}"
            )

            assert (
                sp_total < nosp_total
            ), f"SP memory ({sp_total}) should be less than NoSP memory ({nosp_total})"

            # Check for significant reduction (e.g., at least 10%)
            # The exact ratio depends on how many layers are actually sharded vs replicated/all-gathered
            ratio = sp_total / nosp_total
            print(f"SP/NoSP Ratio: {ratio:.4f}")
            assert (
                ratio < 0.8
            ), f"SP should provide significant memory savings. Ratio: {ratio}"
