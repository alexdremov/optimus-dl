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

from .load_strategy import LoadStrategy

logger = logging.getLogger(__name__)


@dataclass
class CheckpointManagerConfig(RegistryConfig):
    pass


class CheckpointManager:
    """Mixin for checkpoint save/load functionality."""

    def __init__(
        self,
        cfg: CheckpointManagerConfig,
        **kwargs,
    ):
        self.cfg = cfg

    def load_checkpoint_if_exists(
        self,
        checkpoint_dir: str,
        model: BaseModel,
        optimizer: Optimizer | None,
        collective: Collective,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        load_strategy: LoadStrategy | None = None,
        **kwargs,
    ) -> tuple[int, dict | None]:
        """Load checkpoint if exists, return start iteration and metadata."""
        latest_checkpoint = self.find_latest_checkpoint(checkpoint_dir)
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
        checkpoint_dir: str,
        save_freq: int,
        **kwargs,
    ) -> bool:
        """Save checkpoint if iteration matches save_freq."""
        if save_freq <= 0 or iteration % save_freq != 0:
            return False

        try:
            self.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
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
        optimizer: Optimizer | None,
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
        model_state_dict = dcp_state_dict.get_model_state_dict(
            model, options=dcp_state_dict.StateDictOptions()
        )

        state_dict = dict(
            model=model_state_dict,
        )
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
        optimizer: Optimizer | None,
        collective: Collective,
        lr_scheduler=None,
        data_loaders: dict | None = None,
        data_sources=None,
        load_strategy: LoadStrategy | None = None,
        **kwargs,
    ) -> dict:
        """Load training checkpoint using distributed checkpoint API."""
        load_strategy = load_strategy or LoadStrategy()
        checkpoint_path_obj = Path(checkpoint_path)

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
            load_strategy.load_data_sources = False
            logger.warning(
                "Not restoring data sources directly as they will be restored with dataloaders restoration"
            )

        if not load_strategy.load_data_sources:
            data_sources = None
            if load_strategy.load_dataloaders:
                load_strategy.load_dataloaders = False
                logger.warning(
                    "Not restoring dataloaders as data sources are not loaded"
                )

        if not load_strategy.load_dataloaders:
            data_loaders = None

        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Get state dicts for loading
        state_dict = dict()
        if model is not None:
            state_dict["model"] = dcp_state_dict.get_model_state_dict(
                model, options=dcp_state_dict.StateDictOptions()
            )

        if optimizer is not None:
            state_dict["optimizer"] = dcp_state_dict.get_optimizer_state_dict(
                model, optimizer, options=dcp_state_dict.StateDictOptions()
            )

        # Load using distributed checkpoint API
        if len(state_dict) > 0:
            dcp_load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_path),
                process_group=collective.global_process_group,
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

        if not load_strategy.load_iteration:
            metadata["iteration"] = 0

        iteration = metadata["iteration"]

        rank = collective.rank
        per_rank_metadata_path = (
            checkpoint_path_obj.parent / f"per_rank_metadata_{rank}_{iteration:09d}.pt"
        )
        per_rank_metadata = torch.load(
            per_rank_metadata_path, map_location="cpu", weights_only=False
        )
        for key in load_strategy.extra_ignore_keys or []:
            if key in per_rank_metadata:
                per_rank_metadata.pop(key)

        data_loaders = data_loaders or {}
        for k, v in per_rank_metadata.get("data_loaders", {}).items():
            if k in data_loaders:
                logger.info(f"Restoring {k}")
                data_loaders[k].load_state_dict(v)
            else:
                logger.warning(f"Data loader {k} not found in current configuration")

        if "data_sources" in per_rank_metadata and data_sources is not None:
            data_sources.load_state_dict(per_rank_metadata["data_sources"])
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

    def get_checkpoint_path(self, output_path, iteration: int) -> str:
        """Generate checkpoint path for given iteration."""
        output_dir = Path(output_path)
        return str(output_dir / f"checkpoint_{iteration:09d}")

    def find_latest_checkpoint(self, output_path) -> str | None:
        """Find the latest checkpoint in output directory."""
        output_dir = Path(output_path)
        if not output_dir.exists():
            return None

        if "checkpoint_" in output_dir.name:
            # Full checkpoint path
            return str(output_dir)

        # Try to find latest checkpoint
        latest_path = output_dir / "checkpoint_latest"
        if latest_path.exists():
            return str(latest_path)

        return None

    def build_model_from_checkpoint(
        self, checkpoint_path: str, device: str | torch.device, **kwargs
    ) -> tuple[BaseModel, dict]:
        """Build model and load from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or metadata file
            device: Device to load model on
            **kwargs: Additional arguments passed to model building

        Returns:
            Tuple of (model, config) where config is the training config from checkpoint
        """
        checkpoint_path_obj = Path(checkpoint_path)

        # Find metadata file
        metadata_path = self._find_metadata_file(checkpoint_path_obj)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        config = metadata["config"]

        logger.info(f"Loading model with config: {config.model}")

        # Build model using the config
        model = build("model", config.model, **kwargs)
        assert isinstance(model, BaseModel)

        # Load model state dict from checkpoint
        checkpoint_dir = self._find_checkpoint_dir(metadata_path)
        if checkpoint_dir.exists():
            self._load_model_state_dict(model, checkpoint_dir)
        else:
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            logger.warning("Model will use random weights")

        # Move model to device
        model = model.to(device)
        logger.info(f"Loaded model from {checkpoint_path} on {device}")
        return model, config

    def _find_metadata_file(self, checkpoint_path: Path) -> Path:
        """Find the metadata file from checkpoint path."""
        if checkpoint_path.is_file() and checkpoint_path.name.startswith("metadata_"):
            return checkpoint_path
        elif checkpoint_path.is_dir():
            # Look for latest metadata file
            metadata_latest = checkpoint_path / "metadata_latest.pt"
            if metadata_latest.exists():
                return metadata_latest
            # Look for any metadata file
            metadata_files = list(checkpoint_path.glob("metadata_*.pt"))
            if metadata_files:
                return sorted(metadata_files)[-1]  # Return latest by name

        raise FileNotFoundError(f"No metadata file found in {checkpoint_path}")

    def _find_checkpoint_dir(self, metadata_path: Path) -> Path:
        """Find checkpoint directory from metadata path."""
        # Extract iteration from metadata filename
        metadata_name = metadata_path.stem
        if metadata_name == "metadata_latest":
            checkpoint_name = "checkpoint_latest"
        else:
            iteration_str = metadata_name.replace("metadata_", "")
            checkpoint_name = f"checkpoint_{iteration_str}"

        return metadata_path.parent / checkpoint_name

    def _load_model_state_dict(self, model: BaseModel, checkpoint_dir: Path) -> None:
        """Load model state dict from checkpoint, handling both DCP and regular checkpoints."""
        try:
            # First check if this is a DCP checkpoint
            if self._is_dcp_checkpoint(checkpoint_dir):
                logger.info(f"Detected DCP checkpoint: {checkpoint_dir}")
                self._load_dcp_checkpoint(model, checkpoint_dir)
            else:
                # Try to load regular PyTorch checkpoint
                self._load_regular_checkpoint(model, checkpoint_dir)

        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.warning("Continuing with random weights")

    def _is_dcp_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Check if checkpoint directory contains DCP checkpoint format."""
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return False

        # Look for DCP-specific files
        metadata_files = list(checkpoint_dir.glob("*.metadata"))
        shard_files = list(checkpoint_dir.glob("__*.pt"))

        return len(metadata_files) > 0 or len(shard_files) > 0

    def _load_dcp_checkpoint(self, model: BaseModel, checkpoint_dir: Path) -> None:
        """Convert and load DCP checkpoint using dcp_to_torch_save."""
        try:
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

            logger.info(
                f"Converting DCP checkpoint to regular format: {checkpoint_dir}"
            )

            # Create temporary file for converted checkpoint
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Convert DCP checkpoint to regular torch format
                dcp_to_torch_save(
                    dcp_checkpoint_dir=str(checkpoint_dir),
                    torch_save_path=temp_path,
                )

                # Load the converted checkpoint
                logger.info(f"Loading converted checkpoint from: {temp_path}")
                state_dict = torch.load(
                    temp_path, map_location="cpu", weights_only=False
                )

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

        except ImportError:
            logger.error("torch.distributed.checkpoint.format_utils not available")
            logger.warning("Falling back to regular checkpoint loading")
            self._load_regular_checkpoint(model, checkpoint_dir)
        except Exception as e:
            logger.error(f"Failed to convert DCP checkpoint: {e}")
            logger.warning("Falling back to regular checkpoint loading")
            self._load_regular_checkpoint(model, checkpoint_dir)

    def _load_regular_checkpoint(self, model: BaseModel, checkpoint_dir: Path) -> None:
        """Load regular PyTorch checkpoint files."""
        # Look for regular .pt files
        state_files = list(checkpoint_dir.glob("*.pt"))

        if state_files:
            # Try to load the first state file (simplified approach)
            for state_file in state_files:
                try:
                    logger.info(f"Attempting to load: {state_file}")
                    state_dict = torch.load(
                        state_file, map_location="cpu", weights_only=True
                    )

                    if "model" in state_dict:
                        model.load_state_dict(state_dict["model"], strict=False)
                        logger.info(f"Loaded model weights from {state_file}")
                        return
                    elif isinstance(state_dict, dict) and any(
                        key.startswith(("module.", "model.", "_orig_mod."))
                        or "." in key
                        for key in state_dict.keys()
                    ):
                        # This looks like a model state dict
                        model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loaded model weights from {state_file} (direct)")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load {state_file}: {e}")
                    continue

            logger.warning("No valid model state found in checkpoint files")
        else:
            logger.warning(f"No state files found in {checkpoint_dir}")


_, register_checkpoint_manager, build_checkpoint_manager = make_registry(
    "checkpoint_manager", CheckpointManager
)
register_checkpoint_manager("base", CheckpointManagerConfig)(CheckpointManager)
