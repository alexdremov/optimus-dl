"""Checkpoint management system for distributed training.

This module provides the CheckpointManager which handles saving and loading sharded
model and optimizer states using PyTorch's Distributed Checkpoint (DCP) API.
It also manages metadata, learning rate scheduler states, and data loader positions.
"""

import gc
import logging
import os
import pathlib
import shutil
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
from torch.distributed.checkpoint.state_dict_saver import (
    save as dcp_save,
)
from torch.optim import Optimizer

from optimus_dl.core.registry import (
    RegistryConfig,
    build,
    make_registry,
)
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.lr_scheduler import BaseLRScheduler
from optimus_dl.modules.metrics import state_dict as metrics_state_dict
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


class CheckpointManager:
    """Manages saving and loading of distributed checkpoints.

    This class provides high-level orchestration for training checkpoints. It
    integrates with PyTorch DCP for efficient sharded I/O and handles the
    complexity of synchronizing metadata and per-rank states (like dataloaders).
    """

    def __init__(
        self,
        cfg: CheckpointManagerConfig | None = None,
        checkpoint_path: str | Path | None = None,
        full_config: Any | None = None,
        logger_manager: Any | None = None,
        save_freq: int | None = None,
        last_save_freq: int | None = None,
        **kwargs: Any,
    ):
        """Initialize CheckpointManager.

        Args:
            cfg: Configuration object.
            checkpoint_path: Default path to save/load checkpoints.
            full_config: Default full configuration for metadata.
            logger_manager: Default logger manager.
            save_freq: Default persistent save frequency.
            last_save_freq: Default last-iteration save frequency.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.full_config = full_config
        self.logger_manager = logger_manager
        self.save_freq = save_freq
        self.last_save_freq = last_save_freq

    def is_restart(self, checkpoint_path: str | Path | None = None):
        """Check if a checkpoint exists in the given directory.

        This is used to determine if a training run is a fresh start or a
        restart from a previously interrupted run (e.g., due to preemption
        or manual stop). If True, the recipe will attempt to resume state.

        Args:
            checkpoint_path: Path to the output directory or checkpoint. Defaults to self.checkpoint_path.
        """
        checkpoint_path = checkpoint_path or self.checkpoint_path
        if checkpoint_path is None:
            return False
        return self.get_checkpoint(checkpoint_path) is not None

    def get_checkpoint(
        self, path: str | pathlib.Path | None = None
    ) -> CheckpointPath | None:
        """Resolve a generic path into a structured CheckpointPath.

        The path can be:
        1.  A directory: The method searches for the 'latest' symlink or the
            most recent metadata file.
        2.  A metadata file: Direct resolution.
        3.  A checkpoint directory: Direct resolution.

        Args:
            path: The path to resolve. Defaults to self.checkpoint_path.

        Returns:
            A CheckpointPath object if a valid checkpoint is found, else None.
        """
        path = path or self.checkpoint_path
        if path is None:
            return None

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
            checkpoint_path_obj = path.parent / checkpoint_name
            if checkpoint_path_obj.exists():
                return CheckpointPath(
                    metadata=str(path), checkpoint=str(checkpoint_path_obj)
                )

            # maybe not DCP?
            checkpoint_path_obj = path.parent / (checkpoint_name + ".pt")
            if checkpoint_path_obj.exists():
                return CheckpointPath(
                    metadata=str(path), checkpoint=str(checkpoint_path_obj)
                )

        # this is a directory, find latest checkpoint
        if path.is_dir():
            latest_checkpoint = path / "metadata_latest.pt"
            return self.get_checkpoint(latest_checkpoint)
        else:
            return None

    def load_checkpoint_if_exists(
        self,
        model: BaseModel,
        collective: Collective,
        checkpoint_path: str | Path | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: BaseLRScheduler | None = None,
        data_loaders: dict | None = None,
        load_strategy: LoadStrategy | None = None,
        **kwargs: Any,
    ) -> tuple[int, dict | None]:
        """Attempt to find and load the latest checkpoint from a directory.

        Args:
            model: Model to load weights into.
            collective: Collective for distributed coordination.
            checkpoint_path: Directory to search for checkpoints. Defaults to self.checkpoint_path.
            optimizer: Optional optimizer to restore state.
            lr_scheduler: Optional LR scheduler to restore.
            data_loaders: Optional dict of dataloaders to restore state.
            load_strategy: Strategy defining what components to load.
            **kwargs: Passed to load_checkpoint.

        Returns:
            Tuple of (start_iteration, metadata). start_iteration defaults to 1 if no
            checkpoint is found.
        """
        checkpoint_path = checkpoint_path or self.checkpoint_path
        if checkpoint_path is None:
            return 1, None

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
            logger.info(f"Starting with iteration = {start_iteration}")
            return start_iteration, metadata
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            raise

    def save_checkpoint_if_needed(
        self,
        iteration: int,
        collective: Collective,
        checkpoint_path: str | Path | None = None,
        save_freq: int | None = None,
        last_save_freq: int | None = None,
        force_save: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Save checkpoint if iteration matches save_freq."""
        checkpoint_path = checkpoint_path or self.checkpoint_path
        save_freq = save_freq if save_freq is not None else self.save_freq
        last_save_freq = (
            last_save_freq if last_save_freq is not None else self.last_save_freq
        )

        if checkpoint_path is None:
            logger.warning("save_checkpoint_if_needed called without checkpoint_path")
            return False

        is_save_persistent = force_save or (
            save_freq is not None and save_freq > 0 and iteration % save_freq == 0
        )
        if last_save_freq is None:
            is_save_last = is_save_persistent
        else:
            is_save_last = last_save_freq > 0 and iteration % last_save_freq == 0

        if not (is_save_persistent or is_save_last):
            return False

        try:
            self.save_checkpoint(
                checkpoint_path=checkpoint_path,
                iteration=iteration,
                collective=collective,
                is_save_persistent=is_save_persistent,
                is_save_last=is_save_last,
                **kwargs,
            )
            return True
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {e}")
            raise

    def save_checkpoint(
        self,
        model: BaseModel,
        collective: Collective,
        checkpoint_path: str | Path | None = None,
        optimizer: Optimizer | None = None,
        full_config: Any | None = None,
        is_save_persistent: bool = False,
        is_save_last: bool = False,
        iteration: int = 0,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        extra_metadata: dict | None = None,
        metadata_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Save training checkpoint using distributed checkpoint API.

        Args:
            model: Model to save
            collective: Collective for distributed operations
            checkpoint_path: Directory to save checkpoint. Defaults to self.checkpoint_path.
            optimizer: Optimizer to save
            full_config: Full configuration object for metadata. Defaults to self.full_config.
            is_save_persistent: Whether to save with iteration number in filename. Will also create a symlink to 'latest'.
            is_save_last: Whether to save as 'latest'.
            iteration: Current training iteration
            lr_scheduler: Optional LR scheduler to save
            data_loaders: Optional data loaders to save state
            extra_metadata: Extra metadata to save in global metadata file
            metadata_only: If True, only save metadata (skip model/optimizer save)
            **kwargs: Additional metadata to save per-rank
        """
        checkpoint_path = checkpoint_path or self.checkpoint_path
        full_config = full_config if full_config is not None else self.full_config

        logger_manager = kwargs.get("logger_manager", self.logger_manager)
        if logger_manager is not None:
            kwargs["logger_manager"] = logger_manager

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided or set in __init__")

        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists() and collective.is_master:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info("Waiting for all ranks to reach checkpoint saving point...")
        collective.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not metadata_only:
            logger.info(
                f"Saving state for model and optimizer at iteration {iteration}"
            )
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
        else:
            logger.info(f"Saving metadata only at iteration {iteration}")
            state_dict = {}

        # Add metadata
        kwargs_states = {}
        for key, value in kwargs.items():
            kwargs_states[key] = value
            if hasattr(value, "state_dict"):
                logger.info(f"Saving state for {key}")
                kwargs_states[key] = value.state_dict()
            else:
                raise ValueError(
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

        if extra_metadata is not None:
            for key, value in extra_metadata.items():
                assert (
                    key not in metadata
                ), f"Extra metadata key {key} conflicts with existing metadata keys"
                metadata[key] = value

        # Save using distributed checkpoint API
        should_symlink_last = True
        rank = collective.rank
        if is_save_persistent:
            # Symlink to last checkpoint
            checkpoint_id = checkpoint_path / f"checkpoint_{iteration:09d}"
            metadata_path = checkpoint_path / f"metadata_{iteration:09d}.pt"
            per_rank_metadata_path = (
                checkpoint_path / f"per_rank_metadata_{rank}_{iteration:09d}.pt"
            )
        elif is_save_last:
            # Write to the last location directly
            should_symlink_last = False
            checkpoint_id = checkpoint_path / "checkpoint_latest"
            metadata_path = checkpoint_path / "metadata_latest.pt"
            per_rank_metadata_path = (
                checkpoint_path / f"per_rank_metadata_{rank}_latest.pt"
            )
        else:
            raise ValueError(
                f"Calling save checkpoint with both {is_save_persistent = } and {is_save_last = }"
            )

        checkpoint_id_tmp = Path(str(checkpoint_id) + f".{iteration}.tmp")
        metadata_path_tmp = Path(str(metadata_path) + f".{iteration}.tmp")
        per_rank_metadata_tmp = Path(str(per_rank_metadata_path) + f".{iteration}.tmp")

        tmp_paths_to_check = (per_rank_metadata_tmp,)
        if collective.is_master:
            if not metadata_only:
                tmp_paths_to_check = (
                    checkpoint_id_tmp,
                    metadata_path_tmp,
                    per_rank_metadata_tmp,
                )
            else:
                tmp_paths_to_check = (
                    metadata_path_tmp,
                    per_rank_metadata_tmp,
                )

        for path in tmp_paths_to_check:
            if path.exists():
                logger.warning(
                    f"Temporary checkpoint file {path} already exists and will be overwritten"
                )
                if path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        tmp_mappings = [(per_rank_metadata_path, per_rank_metadata_tmp)]
        if collective.is_master:
            # Only master does final renaming of metadata and checkpoint
            tmp_mappings.append((metadata_path, metadata_path_tmp))
            if not metadata_only:
                tmp_mappings.append((checkpoint_id, checkpoint_id_tmp))

        # Ensure all ranks have checked for existing files before any rank starts writing
        logger.info(
            "All ranks are checking for existing checkpoint files before saving..."
        )
        collective.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "All ranks have checked for existing checkpoint files, proceeding with saving..."
        )

        if not metadata_only:
            # All ranks save to temporary paths first
            dcp_save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(checkpoint_id_tmp),
                process_group=collective.process_group,
            )
        else:
            # If metadata only, we must ensure that checkpoint_id directory exists if we are going to symlink it
            # But usually it should already exist from the pre-evaluation save.
            # If it doesn't exist (e.g. force metadata-only save on first step), we might have issues.
            if (
                is_save_persistent
                and not checkpoint_id.exists()
                and collective.is_master
            ):
                checkpoint_id.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    f"Metadata-only save: Created empty checkpoint directory {checkpoint_id}"
                )

        if collective.is_master:
            # Save metadata separately
            torch.save(metadata, metadata_path_tmp)

        assert (
            "data_loaders" not in kwargs_states
        ), "Data loaders should be passed separately"
        assert "metrics" not in kwargs_states, "Metrics should be passed separately"

        per_rank_metadata = {
            "iteration": iteration,
            "data_loaders": {
                k: v.state_dict() for k, v in (data_loaders or {}).items()
            },
            "metrics": metrics_state_dict(),
            **kwargs_states,
        }

        # Save per-rank metadata
        torch.save(per_rank_metadata, per_rank_metadata_tmp)
        logger.info(
            f"Checkpoint saved to {checkpoint_id_tmp if not metadata_only else 'N/A'} / {metadata_path_tmp} / {per_rank_metadata_tmp}"
        )

        # Ensure all ranks have finished saving before renaming
        # This is crucial to prevent corruption if one of the ranks has failed
        collective.barrier()
        logger.info(
            "All ranks have finished saving checkpoint, proceeding with renaming temporary files..."
        )

        final_paths_to_check = (
            per_rank_metadata_path,
        )  # ranks check their own per-rank metadata path to avoid race conditions
        if collective.is_master:
            if metadata_only:
                final_paths_to_check = (
                    metadata_path,
                    per_rank_metadata_path,
                )
            else:
                final_paths_to_check = (
                    checkpoint_id,
                    metadata_path,
                    per_rank_metadata_path,
                )

        for path in final_paths_to_check:
            if path.exists():
                logger.warning(
                    f"Checkpoint file {path} already exists and will be overwritten"
                )
                if path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        # Atomically move temp files to final location
        for final_path, tmp_path in tmp_mappings:
            tmp_path.rename(final_path)
            logger.info(f"Renamed {tmp_path} to {final_path}")

        # Create symlink to latest
        if should_symlink_last:
            latest_checkpoint = checkpoint_path / "checkpoint_latest"
            latest_metadata = checkpoint_path / "metadata_latest.pt"
            latest_per_rank_metadata = (
                checkpoint_path / f"per_rank_metadata_{rank}_latest.pt"
            )

            to_delete = [latest_metadata, latest_per_rank_metadata]
            if not metadata_only:
                to_delete.append(latest_checkpoint)

            if not collective.is_master:
                to_delete = [latest_per_rank_metadata]

            for future_link in to_delete:
                if future_link.is_symlink():
                    future_link.unlink()
                elif future_link.exists():
                    if future_link.is_dir():
                        shutil.rmtree(future_link)
                    else:
                        os.remove(future_link)

            if collective.is_master:
                if not metadata_only:
                    latest_checkpoint.symlink_to(checkpoint_id.name)
                latest_metadata.symlink_to(metadata_path.name)
            latest_per_rank_metadata.symlink_to(per_rank_metadata_path.name)

            logger.info(
                f"Symlinked: {latest_checkpoint if not metadata_only else 'N/A'} -> {checkpoint_id if not metadata_only else 'N/A'}, {latest_metadata} -> {metadata_path}, {latest_per_rank_metadata} -> {per_rank_metadata_path}"
            )

        logger.info(
            f"Checkpoint saved successfully, {checkpoint_id if not metadata_only else 'N/A'}, {per_rank_metadata_path}, {metadata_path}"
        )

        logger.info(
            f"{per_rank_metadata.keys() = } {metadata.keys() = } {state_dict.keys() = }"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        logger_manager = kwargs.get("logger_manager", self.logger_manager)
        if logger_manager is not None:
            kwargs["logger_manager"] = logger_manager

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

        logger.debug(f"Loaded metadata: {metadata}")
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

        logger.debug(f"Loaded per-rank metadata for rank {rank}: {per_rank_metadata}")
        assert iteration == per_rank_metadata.get(
            "iteration", iteration
        ), f"Global iteration {iteration} does not match per-rank iteration {per_rank_metadata.get('iteration', 'unknown')} - checkpoint may be corrupted."

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
            from optimus_dl.modules.metrics import (
                load_state_dict as metrics_load_state_dict,
            )

            metrics_load_state_dict(per_rank_metadata["metrics"])
            logger.info("Restoring metrics")
        else:
            logger.info("Metrics not restored")

        per_rank_metadata_available = set(per_rank_metadata.keys()) - {
            "iteration",
            "metrics",
            "data_loaders",
            "data_sources",
        }
        for key, value in kwargs.items():
            assert hasattr(
                value, "load_state_dict"
            ), f"Do not how to restore {key} = {value}"
            if key not in per_rank_metadata:
                logger.warning(f"Not restoring {key} = {value} as no state found")
                continue
            value.load_state_dict(per_rank_metadata[key])
            per_rank_metadata_available.remove(key)

        if len(per_rank_metadata_available) > 0:
            logger.warning(
                f"Some per-rank metadata keys were not used during loading: {per_rank_metadata_available}"
            )

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
