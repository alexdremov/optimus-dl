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


print(_extract_interpolations("${eval:'\"}\"'}") == ["${eval:'\"}\"'}"])
print(
    _extract_interpolations('${eval:\'"} " + "${.x}" + " {"\'}')
    == ['${eval:\'"} " + "${.x}" + " {"\'}']
)
print(_extract_interpolations("foo ${bar} baz") == ["${bar}"])
print(_extract_interpolations("foo '${bar}'") == ["${bar}"])
print(_extract_interpolations("${eval:'${foo}'}") == ["${eval:'${foo}'}"])
