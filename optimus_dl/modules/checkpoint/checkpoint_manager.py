"""Checkpoint management system for distributed training.

This module provides the CheckpointManager which handles saving and loading sharded
model and optimizer states using PyTorch's Distributed Checkpoint (DCP) API.
It also manages metadata, learning rate scheduler states, and data loader positions.
"""

import logging
import pathlib
import tempfile
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

from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.lr_scheduler import BaseLRScheduler
from optimus_dl.modules.metrics import (
    load_state_dict as metrics_load_state_dict,
    state_dict as metrics_state_dict,
)
from optimus_dl.modules.model.base import BaseModel

from .load_strategy import LoadStrategy

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPath:
    """Represents comprehensive checkpoint information.
    This class holds paths to both the metadata and the actual checkpoint files.

    Attributes:
        metadata: Path to the metadata file (always single file)
        checkpoint: Path to the actual checkpoint file or directory (can be sharded)
    """

    metadata: str
    checkpoint: str

    def per_rank_metadata(self, rank) -> str:
        return self.metadata.replace("metadata_", f"per_rank_metadata_{rank}_")

    def is_dcp_checkpoint(self):
        return pathlib.Path(self.checkpoint).is_dir()


@dataclass
class CheckpointManagerConfig(RegistryConfig):
    """Configuration for CheckpointManager."""

    pass


class CheckpointManager:
    """Manages saving and loading of distributed checkpoints.

    This class provides high-level orchestration for training checkpoints. It
    integrates with PyTorch DCP for efficient sharded I/O and handles the
    complexity of synchronizing metadata and per-rank states (like dataloaders).
    """

    def __init__(
        self,
        cfg: CheckpointManagerConfig,
        **kwargs: Any,
    ):
        """Initialize CheckpointManager.

        Args:
            cfg: Configuration object.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg

    def is_restart(self, checkpoint_path):
        """Check if a checkpoint exists in the given directory.
        Such case corresponds to the case when run was started, stopped and then restarted again.

        Args:
            checkpoint_path: Path for output checkpoints
        """
        return self.get_checkpoint(checkpoint_path) is not None

    def get_checkpoint(self, path: str | pathlib.Path) -> CheckpointPath | None:
        """Get a checkpoint from a path.
        Path can be a directory, then the latest checkpoint is selected or a specific checkpoint directory or metadata file.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(str(path))

        path = path.expanduser().resolve()

        if not path.exists():
            return None

        if path.name.startswith("checkpoint_"):
            # this is exact checkpoint directory
            metadata_name = path.name.replace("checkpoint_", "metadata_") + ".pt"
            metadata_path = path.parent / metadata_name
            if metadata_path.exists():
                return CheckpointPath(metadata=str(metadata_path), checkpoint=str(path))

        if (
            path.name.startswith("metadata_")
            and path.name.endswith(".pt")
            and path.is_file()
        ):
            # this is exact metadata file
            checkpoint_name = pathlib.Path(
                path.name.replace("metadata_", "checkpoint_")
            ).stem
            checkpoint_path = path.parent / checkpoint_name
            if checkpoint_path.exists():
                return CheckpointPath(
                    metadata=str(path), checkpoint=str(checkpoint_path)
                )

            # maybe not DCP?
            checkpoint_path = path.parent / (checkpoint_name + ".pt")
            if checkpoint_path.exists():
                return CheckpointPath(
                    metadata=str(path), checkpoint=str(checkpoint_path)
                )

        # this is a directory, find latest checkpoint
        if path.is_dir():
            latest_checkpoint = path / "metadata_latest.pt"
            return self.get_checkpoint(latest_checkpoint)
        else:
            return None

    def load_checkpoint_if_exists(
        self,
        checkpoint_path: str,
        model: BaseModel,
        collective: Collective,
        optimizer: Optimizer | None = None,
        lr_scheduler: BaseLRScheduler | None = None,
        data_loaders: dict | None = None,
        load_strategy: LoadStrategy | None = None,
        **kwargs: Any,
    ) -> tuple[int, dict | None]:
        """Attempt to find and load the latest checkpoint from a directory.

        Args:
            checkpoint_path: Directory to search for checkpoints.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            collective: Collective for distributed coordination.
            lr_scheduler: Optional LR scheduler to restore.
            data_loaders: Optional dict of dataloaders to restore state.
            load_strategy: Strategy defining what components to load.
            **kwargs: Passed to load_checkpoint.

        Returns:
            Tuple of (start_iteration, metadata). start_iteration defaults to 1 if no
            checkpoint is found.
        """
        latest_checkpoint = self.get_checkpoint(checkpoint_path)
        if not latest_checkpoint:
            return 1, None

        try:
            metadata = self.load_checkpoint(
                checkpoint_path=latest_checkpoint.checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                data_loaders=data_loaders,
                collective=collective,
                load_strategy=load_strategy,
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
        checkpoint_path: str | Path,
        save_freq: int,
        **kwargs: Any,
    ) -> bool:
        """Save checkpoint if iteration matches save_freq."""
        if save_freq <= 0 or iteration % save_freq != 0:
            return False

        try:
            self.save_checkpoint(
                checkpoint_path=checkpoint_path,
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
        checkpoint_path: str | Path,
        model: BaseModel,
        optimizer: Optimizer | None,
        collective: Collective,
        full_config: Any,
        lr_scheduler=None,
        iteration: int = 0,
        data_loaders: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Save training checkpoint using distributed checkpoint API.

        Args:
            checkpoint_path: Directory to save checkpoint
            model: Model to save
            optimizer: Optimizer to save
            collective: Collective for distributed operations
            full_config: Full configuration object for metadata
            lr_scheduler: Optional LR scheduler to save
            iteration: Current training iteration
            data_loaders: Optional data loaders to save state
            **kwargs: Additional metadata to save
        """
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving state for model and optimizer at iteration {iteration}")
        model_state_dict = dcp_state_dict.get_model_state_dict(
            model, options=dcp_state_dict.StateDictOptions()
        )

        state_dict = {
            "model": model_state_dict,
        }
        if optimizer is not None:
            state_dict["optimizer"] = dcp_state_dict.get_optimizer_state_dict(
                model, optimizer, options=dcp_state_dict.StateDictOptions()
            )

        # Add metadata
        kwargs_states = {}
        for key, value in kwargs.items():
            kwargs_states[key] = value
            if hasattr(value, "state_dict"):
                logger.info(f"Saving state for {key}")
                kwargs_states[key] = value.state_dict()
            else:
                logger.error(
                    f"Could not save state for {key} ({value}) as no state_dict() method found"
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
            process_group=collective.process_group,
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
        checkpoint_path: str | Path,
        model: BaseModel | None,
        optimizer: Optimizer | None,
        collective: Collective,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        data_sources=None,
        load_strategy: LoadStrategy | None = None,
        **kwargs: Any,
    ) -> dict:
        """Load training checkpoint using distributed checkpoint API."""
        load_strategy = load_strategy or LoadStrategy()
        checkpoint = self.get_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint with restore strategy {load_strategy}")

        if not load_strategy.load_model:
            model = None
            if load_strategy.load_optimizer:
                load_strategy.load_optimizer = False
                logger.warning("Not restoring optimizer as model is not loaded")

        if not load_strategy.load_optimizer:
            optimizer = None

        if not load_strategy.load_scheduler:
            lr_scheduler = None

        if load_strategy.load_data_sources and load_strategy.load_dataloaders:
            data_sources = None
            load_strategy.load_data_sources = False
            logger.warning(
                "Not restoring data sources directly as they will be restored with dataloaders restoration"
            )
        elif not load_strategy.load_data_sources:
            data_sources = None
            if load_strategy.load_dataloaders:
                load_strategy.load_dataloaders = False
                logger.warning(
                    "Not restoring dataloaders as data sources are not loaded"
                )

        if not load_strategy.load_dataloaders:
            data_loaders = None

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Get state dicts for loading
        state_dict = {}
        if model is not None:
            state_dict["model"] = dcp_state_dict.get_model_state_dict(
                model, options=dcp_state_dict.StateDictOptions()
            )
        else:
            optimizer = None

        if optimizer is not None:
            state_dict["optimizer"] = dcp_state_dict.get_optimizer_state_dict(
                model, optimizer, options=dcp_state_dict.StateDictOptions()
            )

        # Load using distributed checkpoint API
        if len(state_dict) > 0:
            dcp_load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint.checkpoint),
                process_group=collective.process_group,
            )

        # Set the loaded state dicts
        if model is not None:
            dcp_state_dict.set_model_state_dict(
                model, state_dict["model"], options=dcp_state_dict.StateDictOptions()
            )
        if optimizer is not None:
            dcp_state_dict.set_optimizer_state_dict(
                model,
                optimizer,
                state_dict["optimizer"],
                options=dcp_state_dict.StateDictOptions(),
            )

        # Load metadata
        if collective.is_master:
            metadata = torch.load(
                checkpoint.metadata, map_location="cpu", weights_only=False
            )
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

        if not load_strategy.load_iteration:
            metadata["iteration"] = 0

        iteration = metadata["iteration"]

        rank = collective.rank
        per_rank_metadata_path = checkpoint.per_rank_metadata(rank)
        per_rank_metadata = torch.load(
            per_rank_metadata_path, map_location="cpu", weights_only=False
        )
        for key in load_strategy.extra_ignore_keys or []:
            if key in per_rank_metadata:
                per_rank_metadata.pop(key)

        data_loaders = data_loaders or {}
        data_loaders_states = [per_rank_metadata.get("data_loaders", {})]

        # make sure all tp ranks have the same data_loaders_states
        tp_world = collective.tp_world
        tp_world.broadcast_objects(data_loaders_states, source_rank=0)
        for k, v in data_loaders_states[0].items():
            if k in data_loaders:
                logger.info(f"Restoring {k}")
                data_loaders[k].load_state_dict(v)
            else:
                logger.warning(f"Data loader {k} not found in current configuration")

        if "data_sources" in per_rank_metadata and data_sources is not None:
            sources_states = [per_rank_metadata["data_sources"]]
            tp_world.broadcast_objects(sources_states, source_rank=0)
            data_sources.load_state_dict(sources_states[0])
            logger.info(
                "Restoring data sources indipendently (without the full dataloader pipeline)"
            )

        if "metrics" in per_rank_metadata and load_strategy.load_metrics:
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

    def build_model_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device,
        model_key="model",
        **kwargs: Any,
    ) -> tuple[BaseModel, dict]:
        """Build model and load from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or metadata file
            device: Device to load model on
            **kwargs: Additional arguments passed to model building

        Returns:
            Tuple of (model, config) where config is the training config from checkpoint
        """
        checkpoint = self.get_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise FileNotFoundError(f"Metadata file not found: {checkpoint}")

        # Load metadata
        metadata_path = checkpoint.metadata
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        config = metadata["config"]

        logger.info(f"Loading model with config: {config[model_key]}")

        # Build model using the config
        model = build("model", config[model_key], **kwargs)
        assert isinstance(model, BaseModel)

        self.load_model_state_dict(model, checkpoint.checkpoint)

        # Move model to device
        model = model.to(device)
        logger.info(f"Loaded model from {checkpoint_path} on {device}")
        return model, config

    def load_model_state_dict(self, model: BaseModel, checkpoint_path: str) -> None:
        """Load model state dict from checkpoint, handling both DCP and regular checkpoints."""
        checkpoint = self.get_checkpoint(checkpoint_path)
        if checkpoint is None:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            else:
                checkpoint = Path(checkpoint_path)
                is_dcp = Path(checkpoint_path).is_dir()
        else:
            is_dcp = checkpoint.is_dcp_checkpoint()

        if is_dcp:
            logger.info(f"Detected DCP checkpoint: {checkpoint}")
            self._load_dcp_checkpoint(model, checkpoint)
        else:
            # Try to load regular PyTorch checkpoint
            self._load_regular_checkpoint(model, checkpoint)

    def _load_dcp_checkpoint(
        self, model: BaseModel, checkpoint: CheckpointPath | Path
    ) -> None:
        """Convert and load DCP checkpoint using dcp_to_torch_save."""
        if isinstance(checkpoint, CheckpointPath):
            assert checkpoint.is_dcp_checkpoint()
            checkpoint_path = checkpoint.checkpoint
        else:
            assert isinstance(checkpoint, Path)
            assert checkpoint.exists() and checkpoint.is_dir()
            checkpoint_path = checkpoint

        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        # Create temporary file for converted checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Convert DCP checkpoint to regular torch format
            dcp_to_torch_save(
                dcp_checkpoint_dir=str(checkpoint_path),
                torch_save_path=temp_path,
            )

            # Load the converted checkpoint
            logger.info(f"Loading converted checkpoint from: {temp_path}")
            state_dict = torch.load(temp_path, map_location="cpu", weights_only=False)

            if "model" in state_dict:
                model.load_state_dict(state_dict["model"], strict=True)
                logger.info("Successfully loaded model weights from DCP checkpoint")
            else:
                # Try to load the state dict directly if no "model" key
                model.load_state_dict(state_dict, strict=True)
                logger.info(
                    "Successfully loaded model weights from DCP checkpoint (direct)"
                )

        finally:
            # Clean up temporary file
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def _load_regular_checkpoint(
        self, model: BaseModel, checkpoint: CheckpointPath | Path
    ) -> None:
        """Load regular PyTorch checkpoint files."""
        if isinstance(checkpoint, CheckpointPath):
            assert not checkpoint.is_dcp_checkpoint()
            checkpoint_path = checkpoint.checkpoint
        else:
            assert isinstance(checkpoint, Path)
            assert checkpoint.exists() and checkpoint.is_file()
            checkpoint_path = checkpoint

        # Look for regular .pt files
        logger.info(f"Attempting to load: {checkpoint}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model" in state_dict:
            model.load_state_dict(state_dict["model"], strict=False)
            logger.info(f"Loaded model weights from {checkpoint_path}")
            return
        elif isinstance(state_dict, dict) and any(
            key.startswith(("module.", "model.", "_orig_mod.")) or "." in key
            for key in state_dict.keys()
        ):
            # This looks like a model state dict
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model weights from {checkpoint_path} (direct)")


_, register_checkpoint_manager, build_checkpoint_manager = make_registry(
    "checkpoint_manager", CheckpointManager
)
register_checkpoint_manager("base", CheckpointManagerConfig)(CheckpointManager)
