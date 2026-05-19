from unittest.mock import MagicMock

import pytest

from optimus_dl.recipe.train.base import TrainRecipe


class MockTrainRecipe(TrainRecipe):
    def __init__(self):
        self.cfg = MagicMock()
        self.cfg.optimization.iterations = 10
        self.cfg.common.output_path = "/tmp/mock"
        self.cfg.common.log_freq = 1
        self.cfg.metrics = {}

        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.is_restart.return_value = True

        self.logger_manager = MagicMock()
        self.evaluate_and_log = MagicMock()
        self.save_checkpoint_if_needed = MagicMock()

        self.evaluator = MagicMock()
        self.evaluator.cleanup_all_eval_checkpoints = MagicMock()

        self.build_model = MagicMock()
        self.build_optimizer = MagicMock()
        self.build_lr_scheduler = MagicMock()
        self.build_criterion = MagicMock()
        self.build_train_data = MagicMock()

        # We need a train_data_iter
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = iter([1, 2, 3])
        mock_train_pipeline = MagicMock()
        mock_train_pipeline.dataloader = mock_dataloader
        mock_train_pipeline.datasets = None
        self.build_train_data.return_value = mock_train_pipeline

        self.build_eval_data = MagicMock()
        self.setup_training_context = MagicMock()
        self.setup_training_context.return_value = {"scaler": None}

        # Raise an error to stop training early after the resumption logic
        self.run_training_iteration = MagicMock(
            side_effect=RuntimeError("Stop training loop")
        )

    def load_checkpoint_if_exists(self, **kwargs):
        # Return start_iteration, metadata
        return 5, {"iteration": 4, "eval_finished": False}

    def setup_context(self):
        pass

    def close_loggers(self, **kwargs):
        pass

    def log_metrics_to_loggers(self, *args, **kwargs):
        pass


def test_dirty_checkpoint_resumption():
    recipe = MockTrainRecipe()
    recipe.cfg.optimization.iterations = 10

    with pytest.raises(RuntimeError, match="Stop training loop"):
        recipe.run()

    # verify evaluate_and_log was called for iteration 4
    # During the logic it does:
    # iteration = metadata["iteration"] -> 4
    # eval_path = self.eval_checkpoints_path(iteration)
    # self.evaluate_and_log(...)

    assert recipe.evaluate_and_log.called
    kwargs = recipe.evaluate_and_log.call_args_list[
        0
    ].kwargs  # first call should be for resumption
    assert kwargs["iteration"] == 4

    assert recipe.save_checkpoint_if_needed.called
    save_kwargs = recipe.save_checkpoint_if_needed.call_args_list[0].kwargs
    assert save_kwargs["iteration"] == 4
    assert save_kwargs["extra_metadata"] == {"eval_finished": True}
    assert save_kwargs["metadata_only"] is True

    assert recipe.evaluator.cleanup_all_eval_checkpoints.called
    recipe.evaluator.cleanup_all_eval_checkpoints.assert_called_with(4)
