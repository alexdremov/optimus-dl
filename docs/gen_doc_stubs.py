"""Generate the code reference pages and navigation."""

import re
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()


def get_short_docstring(py_file: Path) -> str:
    """Extracts the first non-empty line of the module docstring."""
    if not py_file.exists():
        return ""
    try:
        content = py_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

    # Use a non-greedy match to find the content of the first docstring at module level
    match = re.search(
        r'^\s*("""([\s\S]*?)"""|\'\'\'([\s\S]*?)\'\'\')', content, re.MULTILINE
    )
    if not match:
        return ""

    # group(2) will be from """ and group(3) from '''. One will be None.
    docstring = (match.group(2) or match.group(3) or "").strip()
    if not docstring:
        return ""

    # Find the first non-empty line
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


src_root = Path("optimus_dl")

for path in sorted(src_root.rglob("*.py")):
    # For python identifier (e.g., optimus_dl.core.registry)
    if "_version" in str(path):
        continue
    module_path = path.relative_to(".").with_suffix("")
    ident_parts = tuple(module_path.parts)

    # For doc file path (e.g., reference/core/registry.md)
    doc_path = path.relative_to(src_root).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # For navigation tree (e.g., ('core', 'registry'))
    nav_parts = ident_parts[1:]

    if ident_parts[-1] == "__main__":
        continue

    is_init = ident_parts[-1] == "__init__"
    if is_init:
        ident_parts = ident_parts[:-1]
        nav_parts = nav_parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Special case for the root optimus_dl package
    if not nav_parts and is_init:
        full_doc_path = Path("reference/index.md")
    else:
        nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(ident_parts)
        fd.write(f"::: {ident}")

        if is_init:
            fd.write("\n")
            package_dir = path.parent

            sub_items = []
            for item in sorted(package_dir.iterdir()):
                if item.name.startswith("_"):
                    continue
                # is it a python module?
                if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                    name = item.stem
                    docstring = get_short_docstring(item)
                    link = f"{name}.md"
                    sub_items.append((name, link, docstring))
                # is it a python package?
                elif item.is_dir() and (item / "__init__.py").exists():
                    name = item.name
                    docstring = get_short_docstring(item / "__init__.py")
                    link = f"{name}/"
                    sub_items.append((name, link, docstring))

            if sub_items:
                fd.write("\n## Modules and Sub-packages\n\n")
                for name, link, doc in sub_items:
                    fd.write(f"- [`{name}`]({link}): {doc}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
