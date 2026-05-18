import logging
import pathlib
from typing import Any

import torch

from optimus_dl.modules.distributed import Collective
from optimus_dl.modules.metrics import (
    load_state_dict as meters_load_state_dict,
    state_dict as meters_state_dict,
)

logger = logging.getLogger(__name__)


class EvaluationCheckpointManager:
    """Manages saving and loading of mid-evaluation state.

    This manager handles the storage of metrics (meters) and dataloader states
    during long-running evaluations, allowing them to be resumed if interrupted.
    These checkpoints are separate from the main training checkpoints.
    """

    def __init__(self, output_path: str | pathlib.Path):
        self.output_path = pathlib.Path(output_path)

    def get_eval_checkpoints_dir(self, iteration: int) -> pathlib.Path:
        """Construct the directory path for evaluation checkpoints at a specific iteration."""
        return self.output_path / f"eval_checkpoints_iter_{iteration}"

    def save_iteration_state(
        self,
        iteration: int,
        eval_name: str,
        dataloader_state: dict[str, Any],
        group_name: str,
        collective: Collective | None = None,
        eval_iterations_processed: int = 0,
    ) -> None:
        """Save the current evaluation state for a specific rank.

        Args:
            iteration: Current training iteration.
            eval_name: Name of the evaluation dataset/task.
            dataloader_state: state_dict of the dataloader.
            group_name: Name of the metrics group (e.g., 'eval/dataset').
            collective: Distributed collective.
            eval_iterations_processed: Number of evaluation batches processed so far.
        """
        eval_dir = self.get_eval_checkpoints_dir(iteration)
        eval_dir.mkdir(parents=True, exist_ok=True)

        rank = collective.rank if collective is not None else 0
        checkpoint_path = eval_dir / f"{eval_name}_rank_{rank}.pt"

        # Get the state of the meters for this group
        all_meters_state = meters_state_dict()
        assert (
            group_name in all_meters_state
        ), f"Group {group_name} not found in meters state"

        state = {
            "iteration": iteration,
            "eval_name": eval_name,
            "group_name": group_name,
            "rank": rank,
            "meters_state": all_meters_state[group_name],
            "dataloader_state": dataloader_state,
            "eval_iterations_processed": eval_iterations_processed,
        }

        try:
            # We use atomic save pattern here as well
            tmp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(state, tmp_path)
            tmp_path.rename(checkpoint_path)
            logger.debug(f"Saved evaluation checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(
                f"Failed to save evaluation checkpoint to {checkpoint_path}: {e}"
            )
            raise

    def load_iteration_state(
        self,
        iteration: int,
        eval_name: str,
        group_name: str,
        dataloader: Any,
        collective: Collective | None = None,
    ) -> int:
        """Load the evaluation state for a specific rank if it exists.

        Args:
            iteration: Current training iteration.
            eval_name: Name of the evaluation dataset/task.
            group_name: Name of the metrics group.
            dataloader: The dataloader to restore state to.
            collective: Distributed collective.

        Returns:
            The number of iterations already processed, or 0 if no checkpoint exists.
        """
        eval_dir = self.get_eval_checkpoints_dir(iteration)
        rank = collective.rank if collective is not None else 0
        checkpoint_path = eval_dir / f"{eval_name}_rank_{rank}.pt"

        if not checkpoint_path.exists():
            logger.debug(f"No evaluation checkpoint found at {checkpoint_path}")
            return 0

        try:
            logger.info(f"Loading evaluation checkpoint from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            assert state["iteration"] == iteration, "Checkpoint iteration mismatch"
            assert state["eval_name"] == eval_name, "Checkpoint eval_name mismatch"
            assert state["rank"] == rank, "Checkpoint rank mismatch"

            # Restore meters state
            meters_load_state_dict({group_name: state["meters_state"]})

            # Restore dataloader state
            assert hasattr(
                dataloader, "load_state_dict"
            ), "Dataloader does not support load_state_dict"
            dataloader.load_state_dict(state["dataloader_state"])

            return state.get("eval_iterations_processed", 0)
        except Exception as e:
            logger.error(
                f"Failed to load evaluation checkpoint from {checkpoint_path}: {e}"
            )
            raise

    def cleanup(
        self, iteration: int | None = None, exclude_iteration: int | None = None
    ) -> None:
        """Remove evaluation checkpoints.

        Args:
            iteration: If provided, only remove checkpoints for this specific iteration.
            exclude_iteration: If provided, remove all checkpoints EXCEPT for this iteration.
                Only used when `iteration` is None.
        """
        assert not (
            iteration is not None and exclude_iteration is not None
        ), "Cannot specify both iteration and exclude_iteration"

        try:
            if iteration is not None:
                eval_dir = self.get_eval_checkpoints_dir(iteration)
                if eval_dir.exists():
                    import shutil

                    shutil.rmtree(eval_dir)
                    logger.info(
                        f"Cleaned up evaluation checkpoints for iteration {iteration}"
                    )
            else:
                for eval_dir in self.output_path.glob("eval_checkpoints_iter_*"):
                    # Extract iteration from directory name
                    try:
                        dir_name = eval_dir.name
                        iter_part = int(dir_name.replace("eval_checkpoints_iter_", ""))
                        if (
                            exclude_iteration is not None
                            and iter_part == exclude_iteration
                        ):
                            logger.debug(
                                f"Skipping cleanup of evaluation checkpoint directory: {eval_dir}"
                            )
                            continue
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not parse iteration from directory name: {eval_dir}"
                        )

                    import shutil

                    shutil.rmtree(eval_dir)
                    logger.info(
                        f"Cleaned up evaluation checkpoint directory: {eval_dir}"
                    )
        except Exception as e:
            logger.warning(f"Failed to cleanup evaluation checkpoints: {e}")
