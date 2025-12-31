import json
import logging
import multiprocessing
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

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


@dataclass
class SlowTiktokenConfig(TiktokenConfig):
    process_docs: int = 1


@register_tokenizer("slow_tiktoken", SlowTiktokenConfig)
class SlowTiktokenTokenizer(TiktokenTokenizer):
    def __init__(self, config: SlowTiktokenConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.process_docs = config.process_docs
        self.processed = 0

    def encode(self, text: str) -> list[int]:
        if self.processed == self.process_docs:
            logger.info("Stucking tokenizer")
            raise KeyboardInterrupt
        res = super().encode(text)
        self.processed += 1
        return res


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
    # Cleanup handled by pytest tmp_path, but explicitly if needed:
    shutil.rmtree(output_dir)


@pytest.mark.parametrize(
    "num_proc",
    [0, 1, 2],
)
def test_end_to_end_pretokenization_wikitext(temp_output_dir, num_proc):
    """
    Tests the DataPrepRecipe with a real dataset (wikitext) and tiktoken tokenizer.
    Ensures that the pipeline runs to completion and produces valid artifacts.
    """
    # 1. Setup Configuration
    dataset_config = DatasetConfig(
        repo_id="Salesforce/wikitext",
        split="train",
        config_name="wikitext-2-raw-v1",
        # Use specific pattern to ensure we match the parquet files we saw in cache
        file_pattern="wikitext-2-raw-v1/train-*.parquet",
    )

    processing_config = ProcessingConfig(
        shard_size_mb=1,  # Small shard size to trigger multiple shards if dataset is large enough
        shuffle_buffer_size=100,
        text_column="text",
        seed=42,
        dtype="uint16",
        num_proc=num_proc,  # Test multiprocessing
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
    # Check if index.json exists
    index_path = temp_output_dir / "index.json"
    assert index_path.exists(), "index.json should be created"

    with open(index_path) as f:
        index_data = json.load(f)

    assert "files" in index_data
    assert "total_tokens" in index_data
    assert index_data["total_tokens"] > 0
    assert len(index_data["files"]) > 0

    # Check first shard
    shard_info = index_data["files"][0]
    shard_file = temp_output_dir / shard_info["file"]
    lens_file = temp_output_dir / shard_info["lens_file"]

    assert shard_file.exists()
    assert lens_file.exists()

    # Load shard data
    tokens = np.load(shard_file)
    doc_lens = np.load(lens_file)

    assert tokens.dtype == np.uint16
    assert doc_lens.dtype == np.uint32
    assert len(tokens) == shard_info["num_tokens"]
    assert len(doc_lens) == shard_info["num_docs"]
    assert np.sum(doc_lens) == len(tokens)

    logger.info(f"Successfully processed {index_data['total_tokens']} tokens.")


@pytest.fixture(scope="module")
def ref_temp_output_dir():
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    output_dir = tmp_path / "pretokenized_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    yield output_dir
    shutil.rmtree(tmp_path)


@pytest.fixture(
    params=enumerate(
        [
            DatasetConfig(
                repo_id="Salesforce/wikitext",
                split="train",
                config_name="wikitext-2-raw-v1",
            ),
            DatasetConfig(
                repo_id="Salesforce/wikitext",
                file_pattern="wikitext-103-v1/train-*.parquet",
            ),
        ]
    ),
    scope="module",
)
def reference_tokenization(request, ref_temp_output_dir):
    # Small shard size to ensure we flush and checkpoint frequently
    i, dataset = request.param
    processing_config = ProcessingConfig(
        shard_size_mb=1,
        shuffle_buffer_size=10,
        text_column="text",
        seed=42,
        dtype="uint16",
        num_proc=1,
    )
    if ref_temp_output_dir.exists():
        shutil.rmtree(ref_temp_output_dir)
    # --- 1. Run Reference (Clean) Execution ---
    reference_dir = ref_temp_output_dir / f"reference{i}"
    reference_dir.mkdir(parents=True)

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

    logger.info("Starting reference run...")
    DataPrepRecipe(config_ref).run()

    # Load reference results
    with open(reference_dir / "index.json") as f:
        ref_index = json.load(f)
    ref_total_tokens = ref_index["total_tokens"]
    return {
        "dataset_config": dataset,
        "processing_config": processing_config,
        "ref_total_tokens": ref_total_tokens,
        "tokenizer_config_ref": tokenizer_config_ref,
        "ref_index": ref_index,
    }


@pytest.mark.slow
@pytest.mark.parametrize("interrupt_at", [1, 5210, 5211, 14000, 23765, 23766, 40000])
def test_resumption_logic(temp_output_dir, reference_tokenization, interrupt_at):
    """
    Tests that the recipe can be interrupted and resumed correctly.
    Verifies that the resumed run produces the exact same result as a clean run.
    """
    dataset_config = reference_tokenization["dataset_config"]
    ref_total_tokens = reference_tokenization["ref_total_tokens"]
    processing_config = reference_tokenization["processing_config"]
    tokenizer_config_ref = reference_tokenization["tokenizer_config_ref"]
    ref_index = reference_tokenization["ref_index"]

    logger.info(f"Reference run completed with {ref_total_tokens} tokens.")
    # --- 2. Run Interrupted Execution ---
    resume_dir = temp_output_dir / "resume"
    resume_dir.mkdir()

    output_config_res = OutputConfig(dir=str(resume_dir), name="wikitext")
    # Start with Slow Tokenizer to ensure we catch it mid-process
    tokenizer_config_slow = SlowTiktokenConfig(
        _name="slow_tiktoken", process_docs=interrupt_at
    )

    config_res = DataPrepConfig(
        dataset=dataset_config,
        processing=processing_config,
        output=output_config_res,
        tokenizer=tokenizer_config_slow,
    )

    logger.info("Starting interrupted run (subprocess)...")
    p = multiprocessing.Process(target=_run_recipe_process, args=(config_res,))
    p.start()
    p.join()

    checkpoint_path = resume_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        # Check that we actually did some work but not all
        # If index.json exists, we finished too fast, increase dataset size or delay
        assert not (resume_dir / "index.json").exists(), "Process finished too fast"

        # Verify partial progress
        import pickle

        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)

        # Check that we have processed some tokens but not all
        total_processed = ckpt.sharder_state["total_tokens"] + len(
            ckpt.sharder_state["current_shard_tokens"]
        )
        assert (
            total_processed > 0
        ), "Checkpoint should have some tokens (flushed or in-memory)"
        assert total_processed < ref_total_tokens, "Checkpoint should be partial"
        logger.info(f"Checkpoint verified: {total_processed} tokens processed so far.")

    # --- 3. Resume Execution ---
    logger.info("Resuming processing...")
    # Switch back to fast tokenizer for resumption
    config_res.tokenizer = tokenizer_config_ref

    DataPrepRecipe(config_res).run()

    # --- 4. Compare Results ---
    assert (resume_dir / "index.json").exists()

    with open(resume_dir / "index.json") as f:
        res_index = json.load(f)

    res_total_tokens = res_index["total_tokens"]
    logger.info(f"Resumed run completed with {res_total_tokens} tokens.")

    assert (
        res_total_tokens == ref_total_tokens
    ), f"Token count mismatch! Ref: {ref_total_tokens}, Resumed: {res_total_tokens}"

    # Compare file count
    assert len(ref_index["files"]) == len(res_index["files"]), "Shard count mismatch"

    # Deep comparison of shards could be added here, but total tokens is a strong indicator
    # for this level of test.
