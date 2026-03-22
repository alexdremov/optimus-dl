def _extract_interpolations(s: str) -> list[str]:
    inters = []
    stack = []
    start = -1
    i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue

        if s[i : i + 2] == "${":
            if not stack:
                start = i
            stack.append("INTER")
            i += 2
            continue

        if s[i] == "}":
            if stack and stack[-1] == "INTER":
                stack.pop()
                if not stack:
                    inters.append(s[start : i + 1])
        elif s[i] == "'":
            if stack:
                if stack[-1] == "SQUOTE":
                    stack.pop()
                elif stack[-1] != "DQUOTE":
                    stack.append("SQUOTE")
        elif s[i] == '"':
            if stack:
                if stack[-1] == "DQUOTE":
                    stack.pop()
                elif stack[-1] != "SQUOTE":
                    stack.append("DQUOTE")
        i += 1
    return inters


def test_extract_interpolations_cases() -> None:
    assert _extract_interpolations("${eval:'\"}\"'}") == ["${eval:'\"}\"'}"]
    assert _extract_interpolations(
        '${eval:\'"} " + "${.x}" + " {"\'}'
    ) == ['${eval:\'"} " + "${.x}" + " {"\'}']
    assert _extract_interpolations("foo ${bar} baz") == ["${bar}"]
    assert _extract_interpolations("foo '${bar}'") == ["${bar}"]
    assert _extract_interpolations("${eval:'${foo}'}") == ["${eval:'${foo}'}"]
