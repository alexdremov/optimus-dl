"""Checkpoint mixin for save/load functionality."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dcp_state_dict
from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.optim import Optimizer

from optimus_dl.core.registry import RegistryConfig, make_registry
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.metrics import (
    load_state_dict as metrics_load_state_dict,
    state_dict as metrics_state_dict,
)
from optimus_dl.modules.metrics.common import log_event_end, log_event_start
from optimus_dl.modules.model.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class CheckpointManagerConfig(RegistryConfig):
    pass


class CheckpointManager:
    """Mixin for checkpoint save/load functionality."""

    def __init__(
        self,
        cfg: CheckpointManagerConfig,
        output_path: str,
        save_freq: int = 0,
        **kwargs,
    ):
        """Initialize checkpoint mixin.

        Args:
            output_path: Base path where checkpoints should be saved
            save_freq: Frequency of checkpoint savings
        """
        self.output_path = output_path
        self.save_freq = save_freq

    def load_checkpoint_if_exists(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        collective: Collective,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        **kwargs,
    ) -> tuple[int, dict | None]:
        """Load checkpoint if exists, return start iteration and metadata."""
        latest_checkpoint = self.find_latest_checkpoint()
        if not latest_checkpoint:
            return 1, None

        try:
            metadata = self.load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                data_loaders=data_loaders,
                collective=collective,
                **kwargs,
            )
            start_iteration = metadata["iteration"] + 1
            logger.info(f"Starting with iteration {start_iteration}")
            return start_iteration, metadata
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            raise

    def save_checkpoint_if_needed(
        self,
        iteration: int,
        collective: Collective,
        **kwargs,
    ) -> bool:
        """Save checkpoint if iteration matches save_freq."""
        if self.save_freq <= 0 or iteration % self.save_freq != 0:
            return False

        try:
            self.save_checkpoint(
                checkpoint_dir=self.output_path,
                iteration=iteration,
                collective=collective,
                **kwargs,
            )
            return True
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {e}")
            raise

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        model: BaseModel,
        optimizer: Optimizer,
        collective: Collective,
        full_config: Any,
        lr_scheduler=None,
        iteration: int = 0,
        data_loaders: dict | None = None,
        **kwargs,
    ) -> None:
        """Save training checkpoint using distributed checkpoint API.

        Args:
            checkpoint_dir: Directory to save checkpoint
            model: Model to save
            optimizer: Optimizer to save
            collective: Collective for distributed operations
            full_config: Full configuration object for metadata
            lr_scheduler: Optional LR scheduler to save
            iteration: Current training iteration
            data_loaders: Optional data loaders to save state
            **kwargs: Additional metadata to save
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving state for model and optimizer at iteration {iteration}")
        model_state_dict, optimizer_state_dict = dcp_state_dict.get_model_state_dict(
            model, options=dcp_state_dict.StateDictOptions()
        ), dcp_state_dict.get_optimizer_state_dict(
            model, optimizer, options=dcp_state_dict.StateDictOptions()
        )

        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

        # Add metadata
        kwargs_states = {}
        for key, value in kwargs.items():
            kwargs_states[key] = value
            if hasattr(value, "state_dict"):
                logger.info(f"Saving state for {key}")
                kwargs_states[key] = value.state_dict()
            else:
                logger.warning(
                    f"Could not save state for {key} as no state_dict() method found"
                )
        metadata = {
            "iteration": iteration,
            "config": full_config,
            "world_size": collective.world_size,
        }

        if lr_scheduler is not None:
            logger.info("Saving lr_scheduler")
            metadata["lr_scheduler"] = lr_scheduler.state_dict()

        # Save using distributed checkpoint API
        checkpoint_id = str(checkpoint_path / f"checkpoint_{iteration:09d}")
        dcp_save(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(checkpoint_id),
            process_group=collective.global_process_group,
        )

        metadata_path = None
        if collective.is_master:
            # Save metadata separately
            metadata_path = checkpoint_path / f"metadata_{iteration:09d}.pt"
            torch.save(metadata, metadata_path)
            logger.info(f"Checkpoint saved to {checkpoint_id} / {metadata_path}")

        assert (
            "data_loaders" not in kwargs_states
        ), "Data loaders should be passed separately"
        assert "metrics" not in kwargs_states, "Metrics should be passed separately"
        logger.info("Saving data loaders and metrics")
        per_rank_metadata = {
            "data_loaders": {
                k: v.state_dict() for k, v in (data_loaders or {}).items()
            },
            "metrics": metrics_state_dict(),
            **kwargs_states,
        }

        # Save per-rank metadata
        rank = collective.rank
        per_rank_metadata_path = (
            checkpoint_path / f"per_rank_metadata_{rank}_{iteration:09d}.pt"
        )
        torch.save(per_rank_metadata, per_rank_metadata_path)

        # Create symlink to latest
        if collective.is_master:
            latest_checkpoint = checkpoint_path / "checkpoint_latest"
            latest_metadata = checkpoint_path / "metadata_latest.pt"

            if latest_checkpoint.exists() or latest_checkpoint.is_symlink():
                latest_checkpoint.unlink()
            if latest_metadata.exists():
                latest_metadata.unlink()

            latest_checkpoint.symlink_to(f"checkpoint_{iteration:09d}")
            latest_metadata.symlink_to(f"metadata_{iteration:09d}.pt")

        logger.info(
            f"Checkpoint saved successfully, {checkpoint_id}, {per_rank_metadata_path}, {metadata_path}"
        )
        logger.info(
            f"{per_rank_metadata.keys() = } {metadata.keys() = } {state_dict.keys() = }"
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: BaseModel,
        optimizer: Optimizer,
        collective: Collective,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        **kwargs,
    ) -> dict:
        """Load training checkpoint using distributed checkpoint API."""
        checkpoint_path_obj = Path(checkpoint_path)

        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Get state dicts for loading
        model_state_dict, optimizer_state_dict = dcp_state_dict.get_model_state_dict(
            model, options=dcp_state_dict.StateDictOptions()
        ), dcp_state_dict.get_optimizer_state_dict(
            model, optimizer, options=dcp_state_dict.StateDictOptions()
        )

        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

        # Load using distributed checkpoint API
        dcp_load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_path),
            process_group=collective.global_process_group,
        )

        # Set the loaded state dicts
        dcp_state_dict.set_model_state_dict(
            model, state_dict["model"], options=dcp_state_dict.StateDictOptions()
        )
        dcp_state_dict.set_optimizer_state_dict(
            model,
            optimizer,
            state_dict["optimizer"],
            options=dcp_state_dict.StateDictOptions(),
        )

        # Load metadata
        if collective.is_master:
            metadata_name = (
                checkpoint_path_obj.name.replace("checkpoint_", "metadata_") + ".pt"
            )
            metadata_path = checkpoint_path_obj.parent / metadata_name
            if not metadata_path.exists():
                logger.warning("No metadata found with checkpoint")
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}. "
                    "Checkpoint loaded but metadata is missing."
                )
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
            metadatas = [metadata]
            collective.broadcast_objects(metadatas, source_rank=0)
        else:
            metadatas = [None]
            collective.broadcast_objects(
                metadatas, source_rank=0
            )  # pyright: ignore[reportArgumentType]
            metadata = metadatas[0]
        assert metadata is not None, "Metadata not loaded correctly"

        if lr_scheduler is not None and "lr_scheduler" in metadata:
            lr_scheduler.load_state_dict(metadata["lr_scheduler"])
            logger.info("Restored lr_scheduler")
        else:
            logger.info("Did not restore lr_scheduler")

        iteration = metadata["iteration"]

        rank = collective.rank
        per_rank_metadata_path = (
            checkpoint_path_obj.parent / f"per_rank_metadata_{rank}_{iteration:09d}.pt"
        )
        per_rank_metadata = torch.load(
            per_rank_metadata_path, map_location="cpu", weights_only=False
        )

        data_loaders = data_loaders or {}
        for k, v in per_rank_metadata.get("data_loaders", {}).items():
            if k in data_loaders:
                logger.info(f"Restoring {k}")
                data_loaders[k].load_state_dict(v)
            else:
                logger.warning(f"Data loader {k} not found in current configuration")

        if "metrics" in per_rank_metadata:
            metrics_load_state_dict(per_rank_metadata["metrics"])
            logger.info("Restoring metrics")
        else:
            logger.info("Metrics not restored")

        for key, value in kwargs.items():
            assert hasattr(
                value, "load_state_dict"
            ), f"Do not how to restore {key} = {value}"
            if key not in per_rank_metadata:
                logger.warning(f"Not restoring {key} = {value} as no state found")
            value.load_state_dict(per_rank_metadata[key])

        logger.info(f"Checkpoint has {iteration = }")
        return metadata

    def get_checkpoint_path(self, iteration: int) -> str:
        """Generate checkpoint path for given iteration."""
        output_dir = Path(self.output_path)
        return str(output_dir / f"checkpoint_{iteration:09d}")

    def find_latest_checkpoint(self) -> str | None:
        """Find the latest checkpoint in output directory."""
        output_dir = Path(self.output_path)
        if not output_dir.exists():
            return None

        latest_path = output_dir / "checkpoint_latest"
        if latest_path.exists():
            return str(latest_path)

        return None


_, register_checkpoint_manager, build_checkpoint_manager = make_registry(
    "checkpoint_manager", CheckpointManager
)
register_checkpoint_manager("base", CheckpointManagerConfig)(CheckpointManager)
