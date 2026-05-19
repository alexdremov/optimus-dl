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
        self.output_split_final = self.output_split / "split_final"
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
        extra_args: list[str] | None = None,
        resume_from_iter: int | None = None,
        resume_from_path: str | Path | None = None,
        use_loggers: bool = False,
        eval_freq: int = 0,
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
            f"common.eval_freq={eval_freq}",
            "++common.use_gpu=false",
            "~model_transforms",  # Disable distributed wraps and compilation for speed
            "optimization.amp.enabled=false",
            "data.train_datasets.source.inner._name=txt_lines",
            f"+data.train_datasets.source.inner.file_link={self.data_path}",
            "++common.seed=42",
            "++common.data_seed=42",
            "++common.deterministic=true",
            # Small model architecture for fast execution
            "++model.n_layer=1",
            "++model.n_head=1",
            "++model.n_embd=64",
            "++model.intermediate_size=64",
            "lr_scheduler.warmup_steps=0",
        ]

        if not use_loggers:
            cmd.append("~loggers")
        else:
            cmd.append("loggers.1.enabled=false")

        if extra_args:
            cmd.extend(extra_args)
        if resume_from_iter:
            assert resume_from_path
            cmd.append(
                f"++common.load_checkpoint={resume_from_path}/checkpoint_{resume_from_iter:09d}"
            )

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
        total_steps = 10
        split_step = 5

        # 1. Run uninterrupted training
        self.run_train(self.output_full, iterations=total_steps, save_freq=total_steps)

        # 2. Run first half of split training
        self.run_train(self.output_split, iterations=total_steps, save_freq=split_step)

        # 3. Resume and run second half of split training
        # Resuming is automatic when common.output_path remains the same
        self.run_train(
            self.output_split_final,
            iterations=total_steps,
            save_freq=total_steps,
            resume_from_path=self.output_split,
            resume_from_iter=split_step,
        )

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

    def test_checkpoint_and_eval_consistency(self):
        """Compare uninterrupted training vs resumed training including evaluation metrics."""
        total_steps = 10
        split_step = 5
        eval_freq = 5

        eval_extra_args = [
            "data.eval_datasets.tinyshakespeare.source._name=txt_lines",
            f"+data.eval_datasets.tinyshakespeare.source.file_link={self.data_path}",
            "++common.eval_iterations=2",
            "++data.eval_datasets.tinyshakespeare.transform={_name: compose, transforms: [{_name: tokenize, tokenizer_config: {_name: char_tokenize}}, {_name: chunk_tokens, max_seq_len: 512}, {_name: flat_batcher, batch_size: 32, seq_len: 512}, {_name: to_device}]}",
        ]

        # 1. Run uninterrupted training
        self.run_train(
            self.output_full,
            iterations=total_steps,
            save_freq=total_steps,
            use_loggers=True,
            eval_freq=eval_freq,
            extra_args=eval_extra_args,
        )

        # 2. Run first half of split training
        self.run_train(
            self.output_split,
            iterations=total_steps,
            save_freq=split_step,
            use_loggers=True,
            eval_freq=eval_freq,
            extra_args=eval_extra_args,
        )

        # 3. Resume and run second half of split training
        self.run_train(
            self.output_split_final,
            iterations=total_steps,
            save_freq=total_steps,
            use_loggers=True,
            eval_freq=eval_freq,
            resume_from_path=self.output_split,
            resume_from_iter=split_step,
            extra_args=eval_extra_args,
        )

        # Parse metrics from jsonl files for the final step to check consistency
        import json

        def get_final_eval_loss(output_dir, expected_step):
            found_files = list(
                Path(output_dir).rglob("metrics_eval_tinyshakespeare.jsonl")
            )
            assert (
                len(found_files) > 0
            ), f"No metrics_eval_tinyshakespeare.jsonl found in {output_dir}"
            metrics_file = found_files[0]

            final_loss = None
            with open(metrics_file) as f:
                content = f.read()
                f.seek(0)
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("step") == expected_step and "loss" in data:
                        final_loss = data["loss"]

            if final_loss is None:
                print(f"--- CONTENT OF {metrics_file} ---")
                print(content)
                print("-----------------------------------")
            assert (
                final_loss is not None
            ), f"Eval loss for step {expected_step} not found in {metrics_file}"
            return final_loss

        loss_full = get_final_eval_loss(self.output_full, total_steps)
        loss_split = get_final_eval_loss(self.output_split_final, total_steps)

        assert (
            abs(loss_full - loss_split) < 1e-5
        ), f"Evaluation loss mismatch: {loss_full} vs {loss_split}"
