"""Integration test for checkpoint resumption consistency."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import pytest

from optimus_dl.modules.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective


class TestCheckpointResumption:
    """Test that training resumption is bit-perfectly consistent.

    This test ensures that training for N steps uninterrupted produces the
    exact same model and optimizer state as training for M steps,
    checkpointing, and then resuming for the remaining N-M steps.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_full = self.temp_dir / "full"
        self.output_split = self.temp_dir / "split"
        self.data_path = self.temp_dir / "data.txt"

        # Create dummy data for training
        with open(self.data_path, "w") as f:
            for i in range(1000):
                f.write(f"Line {i}: This is some dummy text for training Llama.\n")

        yield

        # Cleanup temporary directories
        shutil.rmtree(self.temp_dir)

    def run_train(
        self,
        output_path: Path,
        iterations: int,
        save_freq: int,
        extra_args: list[str] = None,
    ):
        """Run the training script via subprocess."""
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config-name=train_llama_shakespeare",
            f"common.output_path={output_path}",
            f"optimization.iterations={iterations}",
            f"common.save_freq={save_freq}",
            "common.log_freq=1",
            "++common.use_gpu=false",
            "~model_transforms",  # Disable distributed wraps and compilation for speed
            "optimization.amp.enabled=false",
            "data.train_datasets.source.inner._name=txt_lines",
            "~data.train_datasets.source.inner.split",
            "~data.eval_datasets.tinyshakespeare.source.split",
            f"+data.train_datasets.source.inner.file_link={self.data_path}",
            "~loggers",
            "common.eval_freq=0",
            "++common.seed=42",
            "++common.data_seed=42",
            "++common.deterministic=true",
            # Small model architecture for fast execution
            "++model.n_layer=1",
            "++model.n_head=1",
            "++model.n_embd=64",
            "++model.intermediate_size=64",
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        # Force some determinism in the environment
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, text=True, env=env, stdout=sys.stdout, stderr=sys.stderr
        )
        if result.returncode != 0:
            pytest.fail(f"Training failed with return code {result.returncode}")
        return result

    def test_checkpoint_consistency(self):
        """Compare uninterrupted training vs resumed training."""
        total_steps = 50
        split_step = 25

        # 1. Run uninterrupted training
        self.run_train(self.output_full, iterations=total_steps, save_freq=total_steps)

        # 2. Run first half of split training
        self.run_train(self.output_split, iterations=split_step, save_freq=split_step)

        # 3. Resume and run second half of split training
        # Resuming is automatic when common.output_path remains the same
        self.run_train(self.output_split, iterations=total_steps, save_freq=total_steps)

        # 4. Compare resulting checkpoints
        mgr = CheckpointManager(CheckpointManagerConfig(_name="base"))

        cp_full = mgr.get_checkpoint(self.output_full)
        cp_split = mgr.get_checkpoint(self.output_split)

        assert cp_full is not None, "Full training checkpoint not found"
        assert cp_split is not None, "Split training checkpoint not found"

        meta_full = torch.load(cp_full.metadata, map_location="cpu", weights_only=False)
        collective = FakeCollective(rank=0, world_size=1)

        def get_model_and_optim(cp_path):
            cfg = meta_full["config"]
            from optimus_dl.core.registry import build

            model = build("model", cfg["model"])
            optimizer = build(
                "optimizer",
                cfg["optimization"]["optimizer"],
                params=model.make_parameter_groups(),
            )

            mgr.load_checkpoint(
                checkpoint_path=cp_path.checkpoint,
                model=model,
                optimizer=optimizer,
                collective=collective,
            )
            return model, optimizer

        model_full, optim_full = get_model_and_optim(cp_full)
        model_split, optim_split = get_model_and_optim(cp_split)

        # Verify all model parameters match exactly
        for (name1, p1), (name2, p2) in zip(
            model_full.named_parameters(), model_split.named_parameters(), strict=True
        ):
            assert name1 == name2
            assert torch.equal(p1, p2), f"Model parameter mismatch: {name1}"

        # Verify optimizer state (moments, step count, etc.) matches exactly
        for s1, s2 in zip(
            optim_full.state.values(), optim_split.state.values(), strict=True
        ):
            for k in s1:
                if torch.is_tensor(s1[k]):
                    assert torch.equal(
                        s1[k], s2[k]
                    ), f"Optimizer state mismatch for key {k}"
                else:
                    assert s1[k] == s2[k], f"Optimizer state mismatch for key {k}"
