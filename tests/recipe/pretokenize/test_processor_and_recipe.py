import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from optimus_dl.modules.tokenizer import register_tokenizer
from optimus_dl.modules.tokenizer.implementations.tiktoken import (
    TiktokenConfig,
    TiktokenTokenizer,
)
from optimus_dl.recipe.pretokenize.config import (
    DataPrepConfig,
    DatasetConfig,
    OutputConfig,
    ProcessingConfig,
)
from optimus_dl.recipe.pretokenize.recipe import DataPrepRecipe

# Configure logging to capture output during tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:

    @register_tokenizer("slow_tiktoken", TiktokenConfig)
    class SlowTiktokenTokenizer(TiktokenTokenizer):
        def __init__(self, config: TiktokenConfig, **kwargs):
            super().__init__(config, **kwargs)
            # We use a simple counter to trigger interruption
            self.process_docs = getattr(config, "process_docs", 1)
            self.processed = 0

        def encode(self, text: str) -> list[int]:
            if self.processed >= self.process_docs:
                logger.info(f"Stopping tokenizer after {self.processed} docs")
                raise KeyboardInterrupt
            res = super().encode(text)
            self.processed += 1
            return res

except ValueError:
    # Already registered
    pass


def _run_recipe_process(config):
    # Running in a separate process
    recipe = DataPrepRecipe(config)
    recipe.run()


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "pretokenized_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    yield output_dir


@pytest.mark.parametrize(
    "num_proc",
    [0, 1] if os.environ.get("CI") == "true" else [0, 1, 2],
)
def test_end_to_end_pretokenization_wikitext(temp_output_dir, num_proc):
    """
    Tests the DataPrepRecipe with a real dataset (wikitext) and tiktoken tokenizer.
    Ensures that the pipeline runs to completion and produces valid artifacts.
    """
    # 1. Setup Configuration - Using wikitext-2-raw-v1 validation split (very small)
    dataset_config = DatasetConfig(
        repo_id="Salesforce/wikitext",
        split="validation",
        config_name="wikitext-2-raw-v1",
    )

    processing_config = ProcessingConfig(
        shard_size_mb=0.1,
        shuffle_buffer_size=10,
        text_column="text",
        seed=42,
        dtype="uint16",
        num_proc=num_proc,
    )

    output_config = OutputConfig(
        dir=str(temp_output_dir),
        name="wikitext_test",
    )

    tokenizer_config = TiktokenConfig(
        _name="tiktoken", name="gpt2", add_bos=True, add_eos=True
    )

    config = DataPrepConfig(
        dataset=dataset_config,
        processing=processing_config,
        output=output_config,
        tokenizer=tokenizer_config,
    )

    # 2. Run Recipe
    recipe = DataPrepRecipe(config)
    recipe.run()

    # 3. Validation
    index_path = temp_output_dir / "index.json"
    assert index_path.exists(), "index.json should be created"

    with open(index_path) as f:
        index_data = json.load(f)

    assert index_data["total_tokens"] > 0
    assert len(index_data["files"]) > 0

    shard_info = index_data["files"][0]
    shard_file = temp_output_dir / shard_info["file"]
    assert shard_file.exists()

    tokens = np.load(shard_file)
    assert tokens.dtype == np.uint16
    logger.info(f"Successfully processed {index_data['total_tokens']} tokens.")


@pytest.fixture(scope="module")
def ref_temp_output_dir():
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="module")
def reference_tokenization(ref_temp_output_dir):
    # Using a small real dataset for reference
    dataset = DatasetConfig(
        repo_id="Salesforce/wikitext",
        split="validation",
        config_name="wikitext-2-raw-v1",
    )

    processing_config = ProcessingConfig(
        shard_size_mb=0.05,
        shuffle_buffer_size=5,
        text_column="text",
        seed=42,
        dtype="uint16",
        num_proc=0,  # Sequential for reliability in reference
    )

    reference_dir = ref_temp_output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    output_config_ref = OutputConfig(dir=str(reference_dir), name="wikitext")
    tokenizer_config_ref = TiktokenConfig(
        _name="tiktoken", name="gpt2", add_bos=True, add_eos=True
    )

    config_ref = DataPrepConfig(
        dataset=dataset,
        processing=processing_config,
        output=output_config_ref,
        tokenizer=tokenizer_config_ref,
    )

    logger.info("Starting real reference run...")
    DataPrepRecipe(config_ref).run()

    with open(reference_dir / "index.json") as f:
        ref_index = json.load(f)

    return {
        "dataset_config": dataset,
        "processing_config": processing_config,
        "ref_total_tokens": ref_index["total_tokens"],
        "tokenizer_config_ref": tokenizer_config_ref,
        "ref_index": ref_index,
    }


@pytest.mark.slow
@pytest.mark.parametrize("interrupt_at", [1, 10, 50])
def test_resumption_logic(temp_output_dir, reference_tokenization, interrupt_at):
    """
    Tests that the recipe can be interrupted and resumed correctly using real data.
    """
    dataset_config = reference_tokenization["dataset_config"]
    ref_total_tokens = reference_tokenization["ref_total_tokens"]
    processing_config = reference_tokenization["processing_config"]
    tokenizer_config_ref = reference_tokenization["tokenizer_config_ref"]

    resume_dir = temp_output_dir / f"resume_{interrupt_at}"
    resume_dir.mkdir()

    output_config_res = OutputConfig(dir=str(resume_dir), name="wikitext")

    # We use a custom config that our SlowTiktokenTokenizer will recognize
    tokenizer_config_slow = TiktokenConfig(_name="slow_tiktoken", name="gpt2")
    # Patch the config to include process_docs for our custom tokenizer
    tokenizer_config_slow.process_docs = interrupt_at

    config_res = DataPrepConfig(
        dataset=dataset_config,
        processing=processing_config,
        output=output_config_res,
        tokenizer=tokenizer_config_slow,
    )

    logger.info(f"Starting interrupted run at {interrupt_at} docs...")
    p = multiprocessing.Process(target=_run_recipe_process, args=(config_res,))
    p.start()
    p.join()

    # Resume
    logger.info("Resuming processing...")
    config_res.tokenizer = tokenizer_config_ref
    DataPrepRecipe(config_res).run()

    with open(resume_dir / "index.json") as f:
        res_index = json.load(f)

    assert (
        res_index["total_tokens"] == ref_total_tokens
    ), f"Token count mismatch! Expected {ref_total_tokens}, got {res_index['total_tokens']}"
    logger.info(
        f"Resumed run matched reference with {res_index['total_tokens']} tokens."
    )
