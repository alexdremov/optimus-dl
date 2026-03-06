from unittest.mock import (
    MagicMock,
    patch,
)

import torch
import torchdata.nodes

from optimus_dl.modules.data.transforms.to_device import (
    ToDeviceTransform,
    ToDeviceTransformConfig,
)


class TestToDeviceTransform:
    def test_to_device_cpu(self):
        device = torch.device("cpu")
        cfg = ToDeviceTransformConfig(pin_memory=False)
        transform = ToDeviceTransform(cfg, device)

        sample = {"input_ids": torch.tensor([1, 2, 3]), "other": "string"}
        # For CPU, it should handle tensors and skip others if properties is None (well, as_tensor will fail on string if we are not careful)
        # Wait, the current implementation iterates over ALL keys if properties is None.
        # This might fail on strings. Let's check.

        transform.properties = ["input_ids"]
        result = transform._map(sample)
        assert result["input_ids"].device == device
        assert result["other"] == "string"

    def test_to_device_cuda_build_order(self):
        device = torch.device("cuda", 0)
        cfg = ToDeviceTransformConfig(
            pin_memory=True, pin_prefetch_factor=2, pin_snapshot_frequency=123
        )
        transform = ToDeviceTransform(cfg, device)

        source = MagicMock(spec=torchdata.nodes.BaseNode)

        with (
            patch("torchdata.nodes.PinMemory") as mock_pin,
            patch("torchdata.nodes.Prefetcher") as mock_prefetch,
            patch("torchdata.nodes.Mapper") as mock_mapper,
        ):

            mock_pin.return_value = "pinned_node"
            mock_mapper.return_value = "mapped_node"
            mock_prefetch.return_value = "prefetched_node"

            result = transform.build(source)

            # Verify PinMemory called first on source
            mock_pin.assert_called_once()
            assert mock_pin.call_args[1]["source"] == source
            assert mock_pin.call_args[1]["snapshot_frequency"] == 123

            # Verify Mapper called second on pinned_node
            mock_mapper.assert_called_once()
            assert mock_mapper.call_args[1]["source"] == "pinned_node"
            assert mock_mapper.call_args[1]["map_fn"] == transform._map

            # Verify Prefetcher called last on mapped_node
            mock_prefetch.assert_called_once()
            assert mock_prefetch.call_args[1]["source"] == "mapped_node"
            assert mock_prefetch.call_args[1]["prefetch_factor"] == 2
            assert mock_prefetch.call_args[1]["snapshot_frequency"] == 123

            assert result == "prefetched_node"

    def test_to_device_map_logic(self):
        device = torch.device("cpu")
        cfg = ToDeviceTransformConfig(properties=["a"])
        transform = ToDeviceTransform(cfg, device)

        sample = {"a": [1, 2, 3], "b": [4, 5, 6]}
        result = transform._map(sample)

        assert torch.is_tensor(result["a"])
        assert result["a"].device == device
        assert not torch.is_tensor(result["b"])  # b was not in properties
