import torchdata.nodes
from omegaconf import OmegaConf

from optimus_dl.modules.data.transforms.base import MapperConfig
from optimus_dl.modules.data.transforms.tokenize import (
    TokenizeTransform,
    TokenizeTransformConfig,
)
from optimus_dl.modules.tokenizer.implementations.char import CharTokenizerConfig


def test_basic_tokenize():
    tokenizer_cfg = CharTokenizerConfig(
        _name="char_tokenize", add_bos=False, add_eos=False
    )
    cfg = TokenizeTransformConfig(tokenizer_config=tokenizer_cfg)
    cfg = OmegaConf.structured(cfg)

    transform_factory = TokenizeTransform(cfg)

    item = {"text": "abc"}
    result = transform_factory._map(item)

    # 'a' -> 97, 'b' -> 98, 'c' -> 99
    expected = [97, 98, 99]
    assert result["input_ids"] == expected


def test_bos_eos():
    tokenizer_cfg = CharTokenizerConfig(
        _name="char_tokenize",
        add_bos=True,
        add_eos=True,
        bos_token_id=256,
        eos_token_id=257,
    )
    cfg = TokenizeTransformConfig(tokenizer_config=tokenizer_cfg)
    cfg = OmegaConf.structured(cfg)
    transform_factory = TokenizeTransform(cfg)

    item = {"text": "abc"}
    result = transform_factory._map(item)

    expected = [256, 97, 98, 99, 257]
    assert result["input_ids"] == expected


def test_utf8_encoding():
    tokenizer_cfg = CharTokenizerConfig(_name="char_tokenize")
    cfg = TokenizeTransformConfig(tokenizer_config=tokenizer_cfg)
    cfg = OmegaConf.structured(cfg)
    transform_factory = TokenizeTransform(cfg)

    # '€' -> \xe2\x82\xac -> [226, 130, 172]
    item = {"text": "€"}
    result = transform_factory._map(item)

    expected = [256, 226, 130, 172, 257]
    assert result["input_ids"] == expected


def test_build():
    # Verify build returns a ParallelMapper
    tokenizer_cfg = CharTokenizerConfig(_name="char_tokenize")
    cfg = TokenizeTransformConfig(
        tokenizer_config=tokenizer_cfg, worker_cfg=MapperConfig(num_workers=1)
    )
    cfg = OmegaConf.structured(cfg)
    transform_factory = TokenizeTransform(cfg)

    source = torchdata.nodes.IterableWrapper([{"text": "test"}])
    pipeline = transform_factory.build(source)

    assert isinstance(pipeline, torchdata.nodes.ParallelMapper)
    results = list(pipeline)
    assert len(results) == 1
    assert results[0]["input_ids"] == [256, 116, 101, 115, 116, 257]  # 'test'
