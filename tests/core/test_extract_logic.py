from optimus_dl.core.omegaconf import _extract_interpolations


def test_extract_interpolations_cases() -> None:
    assert _extract_interpolations("${eval:'\"}\"'}") == ["${eval:'\"}\"'}"]
    assert _extract_interpolations('${eval:\'"} " + "${.x}" + " {"\'}') == [
        '${eval:\'"} " + "${.x}" + " {"\'}'
    ]
    assert _extract_interpolations("foo ${bar} baz") == ["${bar}"]
    assert _extract_interpolations("foo '${bar}'") == ["${bar}"]
    assert _extract_interpolations("${eval:'${foo}'}") == ["${eval:'${foo}'}"]
