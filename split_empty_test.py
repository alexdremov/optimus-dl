from omegaconf._utils import split_key


def _split_path(path: str) -> list[str]:
    # if path is empty, split_key("") returns ['']
    # we might want to return [] instead
    return split_key(path) if path else []


def _join_path(parts: list[str]) -> str:
    return ".".join(parts)


def _to_absolute_key(inter_key: str, parent_path: str) -> str:
    if not inter_key.startswith("."):
        return inter_key
    dots = 0
    temp_key = inter_key
    while temp_key.startswith("."):
        dots += 1
        temp_key = temp_key[1:]
    parts = _split_path(parent_path)
    pop_count = dots - 1
    if pop_count > 0:
        parts = parts[:-pop_count]
    if temp_key:
        parts.extend(_split_path(temp_key))
    return _join_path(parts)


print("split_key empty:", split_key(""))
print("orig:", _to_absolute_key(".foo", ""))
