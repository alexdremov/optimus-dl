"""Training interruption mixin for handling errors and keyboard interrupts."""

import logging
from collections.abc import Callable
from typing import Any

from optimus_dl.modules.distributed import Collective

logger = logging.getLogger(__name__)


class TrainingInterruptionMixin:
    """Mixin for gracefully handling training interruptions.

    Provides a mechanism to catch `KeyboardInterrupt` (Ctrl+C) and trigger a
    safe shutdown sequence, which typically involves saving a final checkpoint
    to ensure progress is not lost.

    Args:
        save_freq: Frequency of regular checkpoints. If 0, saving is disabled.
        output_path: Path where checkpoints are saved.
        checkpoint_callback: Callable to execute for saving the checkpoint.
    """

    def __init__(
        self,
        save_freq: int = 0,
        output_path: str | None = None,
        checkpoint_callback: Callable[..., None] | None = None,
    ):
        self.save_freq = save_freq
        self.output_path = output_path
        self.checkpoint_callback = checkpoint_callback

    def handle_training_interruption(
        self,
        iteration: int,
        collective: Collective | None,
        **kwargs: Any,
    ) -> None:
        """Handle interruption by saving a final checkpoint.

        Args:
            iteration: The current training iteration count.
            collective: The distributed collective instance.
            **kwargs: Additional arguments to pass to the checkpoint callback.
        """
        logger.info("Training interrupted by user")

        # Check if we have checkpoint saving configured and callback available
        if self.save_freq > 0 and self.output_path and self.checkpoint_callback:
            try:
                logger.info("Saving final checkpoint...")

                # Call the checkpoint callback with the required parameters
                self.checkpoint_callback(
                    checkpoint_dir=self.output_path,
                    iteration=iteration,
                    collective=collective,
                    **kwargs,
                )
                logger.info("Final checkpoint saved")

            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")
                raise
        elif self.save_freq > 0:
            logger.warning(
                "Checkpoint saving requested but no callback provided or output_path missing"
            )
