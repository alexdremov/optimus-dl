import numpy as np
import torch
from omegaconf import OmegaConf

from optimus_dl.modules.data import build_data_pipeline


def get_pipeline():
    # Exactly matching the pipeline from train_llama.yaml (eval_datasets.slimpajama6b)
    # but using a smaller dataset for speed.
    cfg = OmegaConf.create(
        {
            "source": {
                "_name": "huggingface_dataset",
                "dataset_load_kwargs": {
                    "path": "wikitext",
                    "name": "wikitext-2-raw-v1",
                    "split": "test",
                    "streaming": True,
                },
            },
            "transform": {
                "_name": "compose",
                "transforms": [
                    {
                        "_name": "tokenize",
                        "tokenizer_config": {"_name": "tiktoken", "name": "gpt2"},
                        "worker_cfg": {"snapshot_frequency": 1},
                    },
                    {"_name": "chunk_tokens", "max_seq_len": 128},
                    {"_name": "shuffle", "buffer_size": 100},
                    {"_name": "flat_batcher", "batch_size": 4, "seq_len": 128},
                    {
                        "_name": "prefetch",
                        "prefetch_factor": 2,
                        "snapshot_frequency": 10,
                    },
                    {"_name": "to_device"},
                ],
            },
            "profile": False,
        }
    )

    # DataBuilder._get_rank_seed logic
    rank = 0
    world_size = 1
    base_data_seed = 42

    rng = torch.Generator()
    rng.manual_seed(base_data_seed + world_size * 10000 + rank)
    seed = torch.randint(0, 2**32, (1,), generator=rng).item()

    device = torch.device("cpu")

    pipeline = build_data_pipeline(
        cfg,
        profile_name="test",
        rank=rank,
        world_size=world_size,
        seed=seed,
        device=device,
    )
    return pipeline


def test_real_pipeline_resumption():
    print("=== Step 1: Uninterrupted Run ===")
    pipe1 = get_pipeline()
    it1 = iter(pipe1.dataloader)

    batches_uninterrupted = []
    for i in range(20):
        batch = next(it1)
        batches_uninterrupted.append(batch["input_ids"].cpu().numpy().copy())
        print(f"Batch {i} sum: {batch['input_ids'].sum().item()}")

    print("\n=== Step 2: Fresh Start (should match Step 1) ===")
    pipe2 = get_pipeline()
    it2 = iter(pipe2.dataloader)
    for i in range(20):
        batch = next(it2)
        match = np.array_equal(
            batch["input_ids"].cpu().numpy(), batches_uninterrupted[i]
        )
        print(f"Batch {i} matches Step 1: {match}")
        if not match:
            print(f"  FAILED: Batch {i} changed on fresh start!")
            raise AssertionError(f"Batch {i} changed on fresh start!")

    print("\n=== Step 3: Resumption at Batch 10 ===")
    pipe3 = get_pipeline()
    it3 = iter(pipe3.dataloader)

    # Iterate to 10
    for _ in range(10):
        next(it3)

    # Get state
    state = it3.state_dict()
    print("Captured state at batch 10. Resuming...")

    # New pipeline instance, load state
    pipe4 = get_pipeline()
    it4 = iter(pipe4.dataloader)
    it4.reset(state)

    for i in range(10, 20):
        batch = next(it4)
        match = np.array_equal(
            batch["input_ids"].cpu().numpy(), batches_uninterrupted[i]
        )
        print(f"Batch {i} matches Step 1: {match}")
        if not match:
            print(f"  FAILED: Batch {i} changed after resumption!")
            # Print some details to help debug
            expected_sum = batches_uninterrupted[i].sum()
            actual_sum = batch["input_ids"].sum().item()
            print(f"  Expected sum: {expected_sum}, Got: {actual_sum}")
            raise AssertionError(f"Batch {i} changed after resumption!")
