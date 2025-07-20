import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torchdata.nodes

from optimus_dl.modules.data.datasets.txt_lines import (
    TxtLinesDataset,
    TxtLinesDatasetConfig,
)


def create_temp_file(content):
    t = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    t.write(content)
    t.close()
    return t.name

    def test_basic_local_file():
        content = "line1\nline2\nline3\n"
        path = create_temp_file(content)
        try:
            cfg = TxtLinesDatasetConfig(
                file_link=path, cache_dir=tempfile.gettempdir(), skip_empty_lines=True
            )
            dataset = TxtLinesDataset(cfg, rank=0, world_size=1)
            loader = torchdata.nodes.Loader(dataset)
            results = list(loader)

            assert len(results) == 3
            assert results[0]["text"] == "line1"
            assert results[1]["text"] == "line2"
            assert results[2]["text"] == "line3"
        finally:
            os.remove(path)

    def test_skip_empty_lines():
        content = "line1\n\nline2\n   \nline3"
        path = create_temp_file(content)
        try:
            cfg = TxtLinesDatasetConfig(
                file_link=path, cache_dir=tempfile.gettempdir(), skip_empty_lines=True
            )
            dataset = TxtLinesDataset(cfg, rank=0, world_size=1)
            loader = torchdata.nodes.Loader(dataset)
            results = list(loader)

            assert len(results) == 3
            assert results[0]["text"] == "line1"
            assert results[1]["text"] == "line2"
            assert results[2]["text"] == "line3"

            # Test with skip_empty_lines=False
            cfg.skip_empty_lines = False
            dataset = TxtLinesDataset(cfg)
            loader = torchdata.nodes.Loader(dataset)
            results = list(loader)
            # line1, empty, line2, spaces, line3
            # Note: implementation uses rstrip('\n') so "line1\n" -> "line1".
            # Empty line "\n" -> "".
            # "   \n" -> "   ".
            assert len(results) == 5
            assert results[1]["text"] == ""
            assert results[3]["text"] == "   "
        finally:
            os.remove(path)


def test_sharding():
    content = "\n".join([f"line{i}" for i in range(10)])
    path = create_temp_file(content)
    try:
        cfg = TxtLinesDatasetConfig(
            file_link=path, cache_dir=tempfile.gettempdir(), skip_empty_lines=True
        )

        # World size 2
        # Rank 0 should get lines 0-4
        ds0 = TxtLinesDataset(cfg, rank=0, world_size=2)
        res0 = list(torchdata.nodes.Loader(ds0))
        assert len(res0) == 5
        assert res0[0]["text"] == "line0"
        assert res0[-1]["text"] == "line4"

        # Rank 1 should get lines 5-9
        ds1 = TxtLinesDataset(cfg, rank=1, world_size=2)
        res1 = list(torchdata.nodes.Loader(ds1))
        assert len(res1) == 5
        assert res1[0]["text"] == "line5"
        assert res1[-1]["text"] == "line9"

    finally:
        os.remove(path)


@patch("requests.get")
def test_download(mock_get):
    # Mock response
    content = b"web_line1\nweb_line2"
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [content]
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as temp_dir:
        url = "http://example.com/data.txt"
        cfg = TxtLinesDatasetConfig(
            file_link=url, cache_dir=temp_dir, skip_empty_lines=True
        )

        dataset = TxtLinesDataset(cfg, rank=0, world_size=1)
        loader = torchdata.nodes.Loader(dataset)
        results = list(loader)

        assert len(results) == 2
        assert results[0]["text"] == "web_line1"
        assert results[1]["text"] == "web_line2"

        # Verify file was created in cache
        # Filename should be hash
        files = os.listdir(temp_dir)
        assert len(files) == 1
        assert files[0].endswith(".txt")  # Check extension
