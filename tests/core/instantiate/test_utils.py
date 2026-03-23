import pytest
from omegaconf import OmegaConf

from optimus_dl.core.instantiate import (
    compose,
    concat_lists,
    cond,
    get_item,
    merge_dicts,
    repeat,
)


def test_merge_dicts():
    d1 = {"a": 1, "b": {"c": 2}}
    d2 = {"b": {"d": 3}, "e": 4}
    d3 = OmegaConf.create({"a": 10, "f": 5})

    result = merge_dicts(d1, d2, d3)

    assert result == {
        "a": 10,
        "b": {"c": 2, "d": 3},
        "e": 4,
        "f": 5,
    }


def test_cond():
    # True branch
    assert cond(True, "foo", "bar") == "foo"
    # False branch
    assert cond(False, "foo", "bar") == "bar"

    # With complex types
    assert cond(True, {"a": 1}, [1, 2]) == {"a": 1}


def test_repeat():
    base = {"_target_": "module", "layer": 1}
    result = repeat(3, base)

    assert len(result) == 3
    assert result[0] == base
    assert result[1] == base
    assert result[2] == base

    # Verify they are deep copies!
    result[0]["layer"] = 2
    assert result[1]["layer"] == 1
    assert result[2]["layer"] == 1


def test_concat_lists():
    l1 = [1, 2, 3]
    l2 = OmegaConf.create([4, 5])
    l3 = [6]

    result = concat_lists(l1, l2, l3)
    assert result == [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError):
        concat_lists(l1, {"a": 1})


def test_get_item():
    mapping = {"adam": "AdamW", "sgd": "SGD"}

    # Basic get
    assert get_item("adam", mapping) == "AdamW"

    # OmegaConf wrapped
    om_mapping = OmegaConf.create(mapping)
    assert get_item("sgd", om_mapping) == "SGD"

    # Default
    assert get_item("rmsprop", mapping, default="Missing") == "Missing"

    # Raise KeyError
    with pytest.raises(KeyError):
        get_item("rmsprop", mapping)


def test_compose():
    def add_1(x):
        return x + 1

    def mult_2(x):
        return x * 2

    def to_str(x):
        return str(x)

    # Test single argument flow
    pipeline = compose(add_1, mult_2, to_str)
    assert pipeline(3) == "8"  # (3+1) * 2

    # Test empty compose
    empty_pipeline = compose()
    assert empty_pipeline(42) == 42

    # Test kwargs on first function
    def make_dict(a, b=2):
        return {"a": a, "b": b}

    def extract_a(d):
        return d["a"]

    pipeline_kwargs = compose(make_dict, extract_a)
    assert pipeline_kwargs(5, b=10) == 5
