"""Integration tests for stateful resumable evaluation."""

import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from optimus_dl.core.log import setup_logging
from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.checkpoint.eval_checkpoint_manager import (
    EvaluationCheckpointManager,
)
from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.data import build_data_pipeline
from optimus_dl.modules.data.config import EvalDataPipelineConfig
from optimus_dl.modules.data.datasets import register_dataset
from optimus_dl.modules.data.datasets.base import BaseDataset
from optimus_dl.modules.distributed import build_best_collective
from optimus_dl.modules.distributed.config import DistributedConfig
from optimus_dl.modules.metrics import (
    log_gathered,
    reset_meters,
)
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.recipe.train.mixins.managers.evaluation_manager import (
    Evaluator,
    EvaluatorConfig,
)


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 16)
        self.linear = torch.nn.Linear(16, 100)

    def forward(self, batch=None, input_ids=None, **kwargs):
        if batch is not None and "input_ids" in batch:
            ids = batch["input_ids"]
        elif input_ids is not None:
            ids = input_ids
        else:
            ids = kwargs.get("input_ids")

        # Log the tokens mean in the batch to identify it
        # Use reset=False so it accumulates across the entire evaluation run
        log_gathered("batch_ids", ids.float().mean().item(), reset=False)

        x = self.embedding(ids)
        return {"logits": self.linear(x)}

    def apply_tp(self, mesh, **kwargs):
        pass


class CrashMidEvalException(Exception):
    pass


@dataclass
class DummyDatasetConfig(RegistryConfig):
    size: int = 10
    rank: int = 0
    crash_at: int | None = None


@register_dataset("dummy_dataset_stateful_test", DummyDatasetConfig)
class DummyDataset(BaseDataset):
    def __init__(self, cfg: DummyDatasetConfig, **kwargs):
        super().__init__(cfg)
        self.size = cfg.size
        self.rank = cfg.rank
        self.current = 0
        self.crash_at = cfg.crash_at

    def reset(self, initial_state=None):
        super().reset(initial_state)
        if initial_state is not None:
            self.current = initial_state.get("current", 0)
        else:
            self.current = 0

    def get_state(self):
        return {"current": self.current}

    def __iter__(self):
        return self

    def next(self):
        if self.crash_at is not None and self.current == self.crash_at:
            raise CrashMidEvalException("Crashing as requested")
        if self.current >= self.size:
            raise StopIteration
        # deterministic output based on current
        x = torch.arange(1, 10, dtype=torch.long).broadcast_to((2, -1)) * (
            (self.current + self.rank) % 100
        )
        target = torch.zeros(2, 10, dtype=torch.long)
        batch = {"input_ids": x, "labels": target}
        self.current += 1
        return batch


def _run_stateful_eval_test(rank, unique_port, world_size, temp_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    setup_logging()

    try:
        collective = build_best_collective(DistributedConfig(), torch.device("cpu"))
        model = DummyModel()
        criterion = CrossEntropyCriterion(
            cfg=CrossEntropyCriterionConfig(padding_token_id=-100),
            collective=collective,
        )

        # Setup Evaluator
        eval_cfg = EvaluatorConfig(_name="base")
        eval_freq = 1
        eval_checkpointing = 2  # Save every 2 batches
        evaluator = Evaluator(
            cfg=eval_cfg,
            eval_freq=eval_freq,
            eval_iterations=6,
            eval_guaranteed_same_batches=True,
            eval_checkpointing=eval_checkpointing,
            output_path=temp_dir,
        )

        device = torch.device("cpu")

        def run_eval_with_crash(crash_at_batch=None, chkp_dir=None, iteration=0):
            # Update evaluator's output path if needed for different runs in same test
            if chkp_dir:
                evaluator.output_path = chkp_dir
                evaluator.eval_checkpoint_manager = EvaluationCheckpointManager(
                    chkp_dir
                )

            eval_pipeline_cfg = EvalDataPipelineConfig(
                source=DummyDatasetConfig(
                    _name="dummy_dataset_stateful_test",
                    size=6,
                    rank=rank,
                    crash_at=crash_at_batch,
                )
            )
            eval_data_pipeline = build_data_pipeline(
                cfg=eval_pipeline_cfg, profile_name="dummy"
            )

            if chkp_dir:
                os.makedirs(chkp_dir, exist_ok=True)

            return evaluator.run_evaluation(
                model=model,
                criterion=criterion,
                eval_data_dict={"dummy": eval_data_pipeline},
                device=device,
                max_iterations=6,
                collective=collective,
                all_metrics_configs={},
                metrics_prefix="eval",
                show_progress=False,
                iteration=iteration,
            )

        # 1. Run uninterrupted evaluation to get ground truth metrics
        uninterrupted_dir = os.path.join(temp_dir, "run1")
        uninterrupted_metrics = run_eval_with_crash(
            crash_at_batch=None, chkp_dir=uninterrupted_dir, iteration=0
        )

        # Verify that checkpoint directory exists for uninterrupted run as well
        # (checkpointed at batch 2, 4, 6)
        if rank == 0:
            chkp_path = os.path.join(
                uninterrupted_dir, "eval_checkpoints_iter_0", "dummy_rank_0.pt"
            )
            assert os.path.exists(chkp_path), f"Checkpoint should exist at {chkp_path}"

        # Reset meters for fresh run
        reset_meters("eval/dummy")

        crash_dir = os.path.join(temp_dir, "run2")
        # 2. Run and crash at batch 3 (after checkpoint at batch 2)
        try:
            run_eval_with_crash(crash_at_batch=3, chkp_dir=crash_dir, iteration=0)
        except CrashMidEvalException:
            pass
        else:
            raise AssertionError("Expected evaluation to crash")

        # Verify that checkpoint exists after crash
        if rank == 0:
            chkp_path = os.path.join(
                crash_dir, "eval_checkpoints_iter_0", "dummy_rank_0.pt"
            )
            assert os.path.exists(
                chkp_path
            ), f"Checkpoint should exist at {chkp_path} after crash"

            # Load checkpoint to verify it's from batch 2
            state = torch.load(chkp_path, map_location="cpu")
            assert (
                state["eval_iterations_processed"] == 2
            ), f"Expected 2 iterations processed, got {state['eval_iterations_processed']}"

        # 3. Resume evaluation
        resumed_metrics = run_eval_with_crash(
            crash_at_batch=None, chkp_dir=crash_dir, iteration=0
        )

        if rank == 0:
            # Metrics should match exactly
            loss_uninterrupted = uninterrupted_metrics["eval/dummy"]["loss"]
            loss_resumed = resumed_metrics["eval/dummy"]["loss"]
            batches_uninterrupted = uninterrupted_metrics["eval/dummy"]["num_batches"]
            batches_resumed = resumed_metrics["eval/dummy"]["num_batches"]

            batch_ids_uninterrupted = uninterrupted_metrics["eval/dummy"]["batch_ids"]
            batch_ids_resumed = resumed_metrics["eval/dummy"]["batch_ids"]

            assert (
                batches_uninterrupted == batches_resumed
            ), f"Batch count mismatch: {batches_uninterrupted} vs {batches_resumed}"
            assert (
                batch_ids_uninterrupted == batch_ids_resumed
            ), f"Batch content/order mismatch: {batch_ids_uninterrupted} vs {batch_ids_resumed}"

            assert torch.isclose(
                torch.tensor(loss_uninterrupted), torch.tensor(loss_resumed)
            ), f"Loss mismatch: {loss_uninterrupted} vs {loss_resumed}"

    finally:
        dist.destroy_process_group()


def _run_eval_test_no_checkpoints(rank, unique_port, world_size, temp_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    setup_logging()

    try:
        collective = build_best_collective(DistributedConfig(), torch.device("cpu"))
        model = DummyModel()
        criterion = CrossEntropyCriterion(
            cfg=CrossEntropyCriterionConfig(padding_token_id=-100),
            collective=collective,
        )

        eval_cfg = EvaluatorConfig(_name="base")
        evaluator = Evaluator(
            cfg=eval_cfg,
            eval_freq=1,
            eval_iterations=6,
            eval_guaranteed_same_batches=True,
            eval_checkpointing=None,  # No checkpointing
            output_path=temp_dir,
        )

        device = torch.device("cpu")

        def run_eval_with_crash(crash_at_batch=None, chkp_dir=None):
            if chkp_dir:
                evaluator.output_path = chkp_dir
                evaluator.eval_checkpoint_manager = EvaluationCheckpointManager(
                    chkp_dir
                )

            eval_pipeline_cfg = EvalDataPipelineConfig(
                source=DummyDatasetConfig(
                    _name="dummy_dataset_stateful_test",
                    size=6,
                    rank=rank,
                    crash_at=crash_at_batch,
                )
            )
            eval_data_pipeline = build_data_pipeline(
                cfg=eval_pipeline_cfg, profile_name="dummy"
            )
            if chkp_dir:
                os.makedirs(chkp_dir, exist_ok=True)
            return evaluator.run_evaluation(
                model=model,
                criterion=criterion,
                eval_data_dict={"dummy": eval_data_pipeline},
                device=device,
                max_iterations=6,
                collective=collective,
                all_metrics_configs={},
                metrics_prefix="eval",
                show_progress=False,
            )

        uninterrupted_dir = os.path.join(temp_dir, "run1")
        uninterrupted_metrics = run_eval_with_crash(
            crash_at_batch=None, chkp_dir=uninterrupted_dir
        )

        # Verify that NO checkpoint directory exists
        if rank == 0:
            # We don't pass iteration, but Evaluator uses 0 by default if it were to save?
            # Actually if iteration is None, it won't even try to save if eval_checkpointing is None.
            # But let's check for any eval_checkpoints_iter_*
            import pathlib

            checkpoints = list(
                pathlib.Path(uninterrupted_dir).glob("eval_checkpoints_iter_*")
            )
            assert (
                len(checkpoints) == 0
            ), f"No checkpoint directories should exist, but found: {checkpoints}"

        reset_meters("eval/dummy")

        crash_dir = os.path.join(temp_dir, "run2")
        try:
            run_eval_with_crash(crash_at_batch=3, chkp_dir=crash_dir)
        except CrashMidEvalException:
            pass
        else:
            raise AssertionError("Expected evaluation to crash")

        # Resume - should start from 0 since no checkpoints
        resumed_metrics = run_eval_with_crash(crash_at_batch=None, chkp_dir=crash_dir)

        if rank == 0:
            loss_uninterrupted = uninterrupted_metrics["eval/dummy"]["loss"]
            loss_resumed = resumed_metrics["eval/dummy"]["loss"]

            assert torch.isclose(
                torch.tensor(loss_uninterrupted), torch.tensor(loss_resumed)
            ), f"Loss mismatch: {loss_uninterrupted} vs {loss_resumed}"

    finally:
        dist.destroy_process_group()


def _run_eval_test_unequal_batches(rank, unique_port, world_size, temp_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(unique_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    setup_logging()

    try:
        collective = build_best_collective(DistributedConfig(), torch.device("cpu"))
        model = DummyModel()
        criterion = CrossEntropyCriterion(
            cfg=CrossEntropyCriterionConfig(padding_token_id=-100),
            collective=collective,
        )

        eval_cfg = EvaluatorConfig(_name="base")
        evaluator = Evaluator(
            cfg=eval_cfg,
            eval_freq=1,
            eval_iterations=10,  # Max limit
            eval_guaranteed_same_batches=False,  # Must synchronize exhaustion
            eval_checkpointing=None,
            output_path=temp_dir,
        )

        device = torch.device("cpu")

        # Rank 0 has 4 batches, Rank 1 has 6 batches.
        # Distributed evaluation should stop at 4 batches for both.
        dataset_size = 4 if rank == 0 else 6

        eval_pipeline_cfg = EvalDataPipelineConfig(
            source=DummyDatasetConfig(
                _name="dummy_dataset_stateful_test",
                size=dataset_size,
                rank=rank,
            )
        )
        eval_data_pipeline = build_data_pipeline(
            cfg=eval_pipeline_cfg, profile_name="dummy"
        )

        metrics = evaluator.run_evaluation(
            model=model,
            criterion=criterion,
            eval_data_dict={"dummy": eval_data_pipeline},
            device=device,
            max_iterations=10,
            collective=collective,
            all_metrics_configs={},
            metrics_prefix="eval",
            show_progress=False,
        )

        if rank == 0:
            # We expect exactly 4 batches processed on each rank, so 4 * world_size in total
            expected_batches = 4 * world_size
            assert (
                metrics["eval/dummy"]["num_batches"] == expected_batches
            ), f"Expected {expected_batches} batches, got {metrics['eval/dummy']['num_batches']}"

    finally:
        dist.destroy_process_group()


class TestStatefulEvaluation:
    """Tests for stateful evaluation resumption and consistency."""

    def test_stateful_evaluation_consistency_distributed(self, unique_port):
        """Test that restoring evaluation yields the same final metrics."""
        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            mp.spawn(
                _run_stateful_eval_test,
                args=(unique_port, world_size, temp_dir),
                nprocs=world_size,
                join=True,
            )

    def test_evaluation_no_checkpoints(self, unique_port):
        """Test that running with eval_checkpointing=None does not create checkpoints and resumes from scratch."""
        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            mp.spawn(
                _run_eval_test_no_checkpoints,
                args=(unique_port, world_size, temp_dir),
                nprocs=world_size,
                join=True,
            )

    def test_evaluation_unequal_batches(self, unique_port):
        """Test that exhaustion synchronization works with unequal batches across ranks."""
        world_size = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            mp.spawn(
                _run_eval_test_unequal_batches,
                args=(unique_port, world_size, temp_dir),
                nprocs=world_size,
                join=True,
            )
