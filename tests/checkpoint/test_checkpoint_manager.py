import os
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from optimus_dl.modules.checkpoint.checkpoint_manager import (
    CheckpointManager,
    CheckpointManagerConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective


def test_save_checkpoint_if_needed_logic():
    cfg = CheckpointManagerConfig(_name="base")
    mgr = CheckpointManager(cfg)
    collective = FakeCollective(rank=0, world_size=1)

    with patch.object(mgr, "save_checkpoint") as mock_save:
        # 1. save_freq=10, last_save_freq=None (default)
        mgr.save_checkpoint_if_needed(
            10, collective, "tmp", save_freq=10, last_save_freq=None
        )
        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        assert kwargs.get("is_save_persistent") is True
        assert kwargs.get("is_save_last") is True
        mock_save.reset_mock()

        # Iteration 5: neither
        mgr.save_checkpoint_if_needed(
            5, collective, "tmp", save_freq=10, last_save_freq=None
        )
        mock_save.assert_not_called()

        # 2. save_freq=10, last_save_freq=5
        # Iteration 5: only last
        mgr.save_checkpoint_if_needed(
            5, collective, "tmp", save_freq=10, last_save_freq=5
        )
        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        assert kwargs.get("is_save_persistent") is False
        assert kwargs.get("is_save_last") is True
        mock_save.reset_mock()


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


def test_save_checkpoint_persistent_paths(tmp_checkpoint_dir):
    cfg = CheckpointManagerConfig(_name="base")
    mgr = CheckpointManager(cfg)
    collective = FakeCollective(rank=0, world_size=1)

    model = MagicMock()
    optimizer = MagicMock()

    with (
        patch(
            "torch.distributed.checkpoint.state_dict.get_model_state_dict",
            return_value={},
        ),
        patch(
            "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
            return_value={},
        ),
        patch("optimus_dl.modules.checkpoint.checkpoint_manager.dcp_save"),
        patch("torch.save"),
        patch("optimus_dl.modules.metrics.state_dict", return_value={}),
    ):
        iteration = 10
        expected_checkpoint_id = tmp_checkpoint_dir / f"checkpoint_{iteration:09d}"
        expected_metadata_path = tmp_checkpoint_dir / f"metadata_{iteration:09d}.pt"
        expected_per_rank = (
            tmp_checkpoint_dir / f"per_rank_metadata_0_{iteration:09d}.pt"
        )

        # Pre-create files to satisfy symlink logic
        expected_checkpoint_id.mkdir(parents=True, exist_ok=True)
        expected_metadata_path.touch()
        expected_per_rank.touch()

        mgr.save_checkpoint(
            checkpoint_path=tmp_checkpoint_dir,
            model=model,
            optimizer=optimizer,
            collective=collective,
            full_config={},
            is_save_persistent=True,
            is_save_last=True,
            iteration=iteration,
        )

        assert (tmp_checkpoint_dir / "checkpoint_latest").is_symlink()
        assert (tmp_checkpoint_dir / "metadata_latest.pt").is_symlink()
        assert (tmp_checkpoint_dir / "per_rank_metadata_0_latest.pt").is_symlink()

        # Verify symlink target
        assert os.readlink(tmp_checkpoint_dir / "checkpoint_latest") == str(
            expected_checkpoint_id.absolute()
        )


def test_save_checkpoint_last_only_paths(tmp_checkpoint_dir):
    cfg = CheckpointManagerConfig(_name="base")
    mgr = CheckpointManager(cfg)
    collective = FakeCollective(rank=0, world_size=1)

    model = MagicMock()
    optimizer = MagicMock()

    latest_cp = tmp_checkpoint_dir / "checkpoint_latest"
    # Pre-create as dir
    latest_cp.mkdir()

    with (
        patch(
            "torch.distributed.checkpoint.state_dict.get_model_state_dict",
            return_value={},
        ),
        patch(
            "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
            return_value={},
        ),
        patch("optimus_dl.modules.checkpoint.checkpoint_manager.dcp_save") as mock_dcp,
        patch("torch.save"),
        patch("optimus_dl.modules.metrics.state_dict", return_value={}),
    ):
        # Mock dcp_save to create the directory since we are testing existence
        mock_dcp.side_effect = lambda *args, **kwargs: latest_cp.mkdir(exist_ok=True)

        mgr.save_checkpoint(
            checkpoint_path=tmp_checkpoint_dir,
            model=model,
            optimizer=optimizer,
            collective=collective,
            full_config={},
            is_save_persistent=False,
            is_save_last=True,
            iteration=15,
        )

        # Should have targeted "latest" directly
        numbered_dirs = list(tmp_checkpoint_dir.glob("checkpoint_0*"))
        assert len(numbered_dirs) == 0

        assert latest_cp.exists()
        assert not latest_cp.is_symlink()


def test_save_checkpoint_atomic_restore_on_failure(tmp_checkpoint_dir):
    cfg = CheckpointManagerConfig(_name="base")
    mgr = CheckpointManager(cfg)
    collective = FakeCollective(rank=0, world_size=1)

    model = MagicMock()
    optimizer = MagicMock()

    # Pre-create "latest" checkpoint
    latest_dir = tmp_checkpoint_dir / "checkpoint_latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / "data.txt").write_text("old data")

    latest_meta = tmp_checkpoint_dir / "metadata_latest.pt"
    latest_meta.write_text("old meta")

    latest_per_rank = tmp_checkpoint_dir / "per_rank_metadata_0_latest.pt"
    latest_per_rank.write_text("old per rank")

    with (
        patch(
            "torch.distributed.checkpoint.state_dict.get_model_state_dict",
            return_value={},
        ),
        patch(
            "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
            return_value={},
        ),
        patch(
            "optimus_dl.modules.checkpoint.checkpoint_manager.dcp_save",
            side_effect=RuntimeError("Save failed"),
        ),
        patch("torch.save"),
        patch("optimus_dl.modules.metrics.state_dict", return_value={}),
    ):
        with pytest.raises(RuntimeError, match="Save failed"):
            mgr.save_checkpoint(
                checkpoint_path=tmp_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                collective=collective,
                full_config={},
                is_save_persistent=False,
                is_save_last=True,
                iteration=20,
            )

    assert (latest_dir / "data.txt").read_text() == "old data"
    assert latest_meta.read_text() == "old meta"
    assert latest_per_rank.read_text() == "old per rank"


def test_save_checkpoint_atomic_restore_on_metadata_failure(tmp_checkpoint_dir):
    cfg = CheckpointManagerConfig(_name="base")
    mgr = CheckpointManager(cfg)
    collective = FakeCollective(rank=0, world_size=1)

    model = MagicMock()
    optimizer = MagicMock()

    # Pre-create "latest" checkpoint
    latest_dir = tmp_checkpoint_dir / "checkpoint_latest"
    latest_dir.mkdir(parents=True)

    latest_meta = tmp_checkpoint_dir / "metadata_latest.pt"
    latest_meta.write_text("old meta")

    # Mock dcp_save to succeed, but torch.save to fail
    with (
        patch(
            "torch.distributed.checkpoint.state_dict.get_model_state_dict",
            return_value={},
        ),
        patch(
            "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
            return_value={},
        ),
        patch("optimus_dl.modules.checkpoint.checkpoint_manager.dcp_save"),
        patch("torch.save", side_effect=RuntimeError("Metadata Save failed")),
        patch("optimus_dl.modules.metrics.state_dict", return_value={}),
    ):
        with pytest.raises(RuntimeError, match="Metadata Save failed"):
            mgr.save_checkpoint(
                checkpoint_path=tmp_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                collective=collective,
                full_config={},
                is_save_persistent=False,
                is_save_last=True,
                iteration=20,
            )

    assert latest_meta.exists(), "metadata_latest.pt should have been restored"
    assert latest_meta.read_text() == "old meta"
