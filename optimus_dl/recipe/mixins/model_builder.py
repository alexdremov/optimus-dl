"""Model builder mixin for building and transforming models with checkpoint loading."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from optimus_dl.core.model_utils import get_num_parameters
from optimus_dl.core.registry import RegistryConfig, build, make_registry
from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.model import ModelConfig
from optimus_dl.modules.model.base import BaseModel
from optimus_dl.modules.model_transforms import (
    ModelTransformConfig,
    build_model_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBuilderConfig(RegistryConfig):
    pass


class ModelBuilder:
    """Mixin for building models and applying transforms, with checkpoint loading support."""

    def __init__(
        self,
        cfg: ModelBuilderConfig,
        model_transforms: list[ModelTransformConfig] | None = None,
        **kwargs,
    ):
        self.model_transforms = model_transforms or []

    def build_model(
        self, model_config: ModelConfig | None, collective: Collective, **kwargs
    ) -> BaseModel:
        """Build and validate the model."""
        if model_config is None:
            raise ValueError(
                "model_config is None. Use build_model_from_checkpoint for evaluation."
            )

        model = build("model", model_config, **kwargs)
        logger.info(
            f"Params num (before model transforms): {get_num_parameters(model):,}"
        )
        assert isinstance(model, BaseModel)

        # Apply model transforms (including distributed setup)
        model = self._apply_model_transforms(
            model, collective=collective, device=collective.default_device, **kwargs
        )
        logger.info(f"Model \n{model}")
        logger.info(
            f"Params num (after model transforms): {get_num_parameters(model):,}"
        )

        return model

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
        model.eval()

        logger.info(f"Loaded model from {checkpoint_path} on {device}")
        return model, config

    def _apply_model_transforms(self, model: BaseModel, **kwargs) -> BaseModel:
        """Apply configured model transforms to the model.

        Args:
            model: The model to transform
            **kwargs: Additional arguments to pass to transforms

        Returns:
            The transformed model
        """
        for transform_cfg in self.model_transforms:
            try:
                transform = build_model_transform(transform_cfg, **kwargs)
                if transform is not None:
                    logger.info(f"Applying model transform: {transform}")
                    model = transform.apply(model, **kwargs)
                else:
                    logger.warning(
                        f"Failed to build model transform from config: {transform_cfg}"
                    )
            except Exception as e:
                logger.error(f"Failed to apply model transform {transform_cfg}: {e}")
                raise

        return model

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


_, register_model_builder, build_model_builder = make_registry(
    "model_builder", ModelBuilder
)
register_model_builder("base", ModelBuilderConfig)(ModelBuilder)
