import gc
import os
import traceback

import torch
import pytest
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
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 4,
    "vocab_size": 64,
    "sequence_length": 1024,
}
qwen3_test_cfg = {
    "_name": "qwen3",
    "n_embd": 64,
    "n_head": 4,
    "n_kv_head": 2,
    "n_layer": 4,
    "vocab_size": 64,
    "sequence_length": 1024,
}
olmo3_mem_cfg = {
    "_name": "olmo3",
    "n_embd": 64,
    "n_head": 4,
    "n_kv_head": 2,
    "n_layer": 4,
    "vocab_size": 64,
    "sequence_length": 1024,
    "layer_types": [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ],
}
models_cfg = [llama2_mem_cfg, qwen3_test_cfg, olmo3_mem_cfg]


def _run_memory_test(rank, unique_port, world_size, model_cfg_dict, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
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

    except Exception as e:
        error_msg = f"Rank {rank} failed with error: {e}\n{traceback.format_exc()}"
        result_queue.put({"error": error_msg})
        raise
    finally:
        dist.destroy_process_group()


class TestTPMemory:

    @pytest.mark.slow
    @pytest.mark.parametrize("model_cfg_dict", models_cfg)
    def test_sp_memory_usage(self, unique_port, model_cfg_dict):
        world_size = 2
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        mp.spawn(
            _run_memory_test,
            args=(unique_port, world_size, model_cfg_dict, queue),
            nprocs=world_size,
            join=True,
        )

        results = []
        while not queue.empty():
            results.append(queue.get())

        for res in results:
            if "error" in res:
                pytest.fail(res["error"])

        if results:
            sizes = results[0]
            nosp_total = sizes["nosp"]
            sp_total = sizes["sp"]

            print(
                f"Total activation sizes (Bytes) - NoSP: {nosp_total}, SP: {sp_total}"
            )

            assert (
                sp_total < nosp_total
            ), f"SP memory ({sp_total}) should be less than NoSP memory ({nosp_total})"
